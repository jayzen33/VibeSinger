from __future__ import annotations

import importlib
from os import PathLike
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from vocos import Vocos

from singer.decoder.modules import MelSpec
from singer.decoder.utils import (
    convert_char_to_pinyin,
    default,
    exists,
    get_epss_timesteps,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
)


class YingSinger(nn.Module):
    def __init__(
        self,
        sigma=0.0,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        with open(f"{Path(__file__).parent}/config/vocab.txt", "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        self.vocab_char_map = vocab_char_map
        self.vocab_size = len(vocab_char_map)

        _cfg = OmegaConf.load(f"{Path(__file__).parent}/config/beta.yaml")

        decoder_cls_path = _cfg.decoder.backbone
        module_path, class_name = decoder_cls_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        decoder_cls = getattr(module, class_name)

        decoder_arc = _cfg.decoder.arch

        melody_encoder_cls_path = _cfg.melody_encoder.backbone
        module_path, class_name = melody_encoder_cls_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        melody_encoder_cls = getattr(module, class_name)

        mel_dim = _cfg.decoder.mel_spec.n_mel_channels
        mel_spec_kwargs = dict(_cfg.decoder.mel_spec)

        self.transformer = decoder_cls(**decoder_arc, text_num_embeds=self.vocab_size, mel_dim=mel_dim)
        self.melody_extractor = melody_encoder_cls(in_dim=mel_dim)
        self.vocoder = self._load_vocoder(
            vocoder_path=_cfg.decoder.vocoder.path,
            is_local=_cfg.decoder.vocoder.is_local,
            hf_cache_dir="./.cache/huggingface",
        )

        self.sigma = sigma

        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        self.resampler_cache = dict()

    def _load_checkpoint(
        self,
        ckpt_path: str | PathLike,
        map_location: str | torch.device = "cpu",
    ):
        ckpt_path = str(ckpt_path)
        stats = torch.load(ckpt_path, map_location=map_location)
        if "model" in stats:
            self.load_state_dict(stats["model"], strict=False)
        else:
            self.load_state_dict(stats, strict=False)

    def _load_vocoder(
        self,
        vocoder_path: str | PathLike,
        is_local: bool = False,
        hf_cache_dir: str | PathLike = "./.cache/huggingface",
    ):
        if is_local:
            assert vocoder_path != "", "local_path must be provided when is_local is True"
            print(f"Load vocos from local path {vocoder_path}")
            config_path = f"{vocoder_path}/config.yaml"
            model_path = f"{vocoder_path}/pytorch_model.bin"
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            repo_id = "charactr/vocos-mel-24khz"
            config_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="config.yaml")
            model_path = hf_hub_download(repo_id=repo_id, cache_dir=hf_cache_dir, filename="pytorch_model.bin")

        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        from vocos.feature_extractors import EncodecFeatures

        if isinstance(vocoder.feature_extractor, EncodecFeatures):
            encodec_parameters = {
                "feature_extractor.encodec." + key: value
                for key, value in vocoder.feature_extractor.encodec.state_dict().items()
            }
            state_dict.update(encodec_parameters)
        vocoder.load_state_dict(state_dict)
        vocoder = vocoder.eval()

        return vocoder

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"] | None = None,  # noqa: F821
        *,
        melody_in: float["b n d"] | None = None,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        use_epss=True,
    ):
        self.eval()

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        duration = default(duration, cond.shape[1] * 2)

        cond = cond.to(next(self.parameters()).dtype)

        melody, _ = self.melody_extractor(melody_in)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        assert batch == 1, "Currently only batch size 1 is supported."

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        cond_mask = lens_to_mask(lens)

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration)
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        melody = F.pad(melody, (0, 0, 0, max_duration - melody.shape[1]), value=0.0)
        melody = torch.where(cond_mask, torch.zeros_like(melody), melody)

        def fn(t, x):
            pred_cfg = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                melody=melody,
                time=t,
                mask=None,
                cache=False,
            )
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            return pred + (pred - null_pred) * cfg_strength

        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        if t_start == 0 and use_epss:  # use Empirically Pruned Step Sampling for low NFE
            t = get_epss_timesteps(steps, device=self.device, dtype=step_cond.dtype)
        else:
            t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=step_cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        trajectory = odeint(fn, y0, t, method="euler")
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def _load_audio(self, path, target_sr):
        audio_ori, sr = torchaudio.load(path)
        if audio_ori.shape[0] > 1:
            audio_ori = torch.mean(audio_ori, dim=0, keepdim=True)

        if sr != target_sr:
            if (sr, target_sr) not in self.resampler_cache:
                self.resampler_cache[(sr, target_sr)] = torchaudio.transforms.Resample(sr, target_sr)
            audio = self.resampler_cache[(sr, target_sr)](audio_ori)
        else:
            audio = audio_ori

        return audio

    def _tokenize_lyrics(self, lyrics_str: str):
        all_tokens = []
        assert "|" not in lyrics_str, "Lyrics string should not contain '|'"

        lyric_clean = lyrics_str.strip()
        for punct in "，。！？、；：,.!?;:":
            lyric_clean = lyric_clean.replace(punct, " ")

        if not lyric_clean.strip():
            return None

        if lyric_clean.strip():
            text = convert_char_to_pinyin([lyric_clean.strip()], polyphone=True)[0]
            tokens = [self.vocab_char_map.get(c, 0) for c in text]
            all_tokens.extend(tokens)

        return torch.tensor(all_tokens, dtype=torch.long)

    def inference(
        self,
        timbre_audio_path: str,
        timbre_audio_content: str,
        melody_audio_path: str | None = None,
        lyrics: str | None = None,
        cfg_strength: float = 2.0,
        nfe_steps: int = 32,
        seed: int | None = 2025,
    ):
        _device = self.device

        timbre_audio = self._load_audio(timbre_audio_path, 24000)
        melody_audio = self._load_audio(melody_audio_path, 24000)

        melody_in = torch.cat([timbre_audio, melody_audio], dim=1)  # dummy midi input
        melody_in = self.mel_spec(melody_in).transpose(1, 2).to(_device)

        dummy_len = timbre_audio.shape[1] / self.mel_spec.hop_length + melody_audio.shape[1] / self.mel_spec.hop_length
        dummy_len = int(dummy_len)

        timbre_audio = timbre_audio.to(_device)
        timbre_audio_len = timbre_audio.shape[1] // self.mel_spec.hop_length

        assert isinstance(timbre_audio_content, str) and len(timbre_audio_content) > 0, (
            "Timbre audio content must be provided as a non-empty string."
        )
        assert isinstance(lyrics, str) and len(lyrics) > 0, "Lyrics must be provided as a non-empty string."

        text_list = [timbre_audio_content] + [lyrics]
        text_tokens = torch.zeros(1, dummy_len, dtype=torch.long).to(_device) - 1  # -1 for padding

        if melody_in.shape[1] > dummy_len:
            melody_in = melody_in[:, :dummy_len, :]
        elif melody_in.shape[1] < dummy_len:
            melody_in = F.pad(melody_in, (0, 0, 0, dummy_len - melody_in.shape[1]), value=0.0)

        for idx, item in enumerate(text_list):
            lyric = item

            if idx < 1:
                timestamp_in_frame = 1
            else:
                timestamp_in_frame = timbre_audio_len + 1

            tokenized = self._tokenize_lyrics(lyric)
            if tokenized is None:
                continue

            token_len = len(tokenized)
            text_tokens[0, timestamp_in_frame : timestamp_in_frame + token_len] = tokenized

        generated, _ = self.sample(
            cond=timbre_audio,
            melody_in=melody_in,
            text=text_tokens,
            duration=dummy_len,
            steps=nfe_steps,
            cfg_strength=cfg_strength,
            sway_sampling_coef=-1,
            seed=seed,
        )
        del _

        generated = generated.to(torch.float32)  # generated mel spectrogram
        generated = generated[:, timbre_audio_len:, :]
        generated = generated.permute(0, 2, 1)

        generated_wave = self.vocoder.decode(generated)
        generated_wave = generated_wave.squeeze().cpu().numpy()

        return generated_wave


if __name__ == "__main__":
    import argparse

    import soundfile as sf

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, default="", help="Path to model checkpoint.")
    parser.add_argument("--timbre_audio_path", type=str, default="", help="Path to timbre audio.")
    parser.add_argument("--timbre_audio_content", type=str, default="", help="Content of timbre audio.")
    parser.add_argument("--melody_audio_path", type=str, default="", help="Path to melody audio.")
    parser.add_argument("--out_path", type=str, default="", help="Path to save output audio.")

    parser.add_argument("--lyrics", type=str, default="", help="Lyrics text.")
    parser.add_argument("--cfg_strength", type=float, default=2.0, help="Classifier-free guidance strength.")
    parser.add_argument("--nfe_steps", type=int, default=32, help="Number of function evaluations (NFE) steps.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")

    args = parser.parse_args()

    singer = YingSinger()
    singer._load_checkpoint(args.ckpt_path)

    gen_wav = singer.inference(
        timbre_audio_path=args.timbre_audio_path,
        timbre_audio_content=args.timbre_audio_content,
        melody_audio_path=args.melody_audio_path,
        lyrics=args.lyrics,
        cfg_strength=2.0,
        nfe_steps=100,
        seed=2025,
    )
    sf.write(args.out_path, gen_wav, 24000)
