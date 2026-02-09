from __future__ import annotations

import json
import math
import os
import random
from os import PathLike
from pathlib import Path
from typing import Optional, Union

import hydra
import numpy as np
import pretty_midi
import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

from singer.decoder.utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
)
from singer.rmvpe.inference import RMVPE
from singer.rmvpe.spec import MelSpectrogram as RMVPEMelSpectrogram
from singer.tokenizer.g2p import PhonemeBpeTokenizer
from singer.vae.autoencoders import create_autoencoder_from_config


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Constants
# =============================================================================

# Audio sample rates
SAMPLE_RATE_16K = 16000
SAMPLE_RATE_24K = 24000
SAMPLE_RATE_44K = 44100
SAMPLE_RATE_48K = 48000

# Model parameters
DEFAULT_FEATURE_DIM = 64
DEFAULT_FRAME_HZ = 100
DEFAULT_SEED = 2025

# Mel spectrogram configuration for RMVPE
RMVPE_MEL_CONFIG = {
    "n_mel_channels": 80,
    "sampling_rate": SAMPLE_RATE_44K,
    "win_length": 2048,
    "hop_length": 512,
}

# =============================================================================
# Audio Utilities
# =============================================================================

_resampler_cache: dict[str, torchaudio.transforms.Resample] = {}


def get_time_span(
    method: str,
    t_start: float,
    t_end: float,
    steps: int,
    device: torch.device,
    power: float = 2.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate time span tensor for diffusion sampling.

    Args:
        method: Scheduling method. One of "linear", "cosine", or "power".
        t_start: Start time (typically 0).
        t_end: End time (typically 1).
        steps: Number of diffusion steps.
        device: Target device for the tensor.
        power: Exponent for power scheduling (only used when method="power").
        dtype: Data type for the tensor.

    Returns:
        Time span tensor of shape (steps + 1,) with values from t_start to t_end.

    Raises:
        ValueError: If method is not recognized.
    """
    t = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)

    if method == "linear":
        pass
    elif method == "cosine":
        t = 1 - torch.cos(t * math.pi / 2)
    elif method == "power":
        t = torch.pow(t, power)
    else:
        raise ValueError(f"Unknown schedule method: {method}. Expected 'linear', 'cosine', or 'power'.")

    return t_start + t * (t_end - t_start)


def load_audio(
    path: Union[str, PathLike],
    target_sr: int,
) -> tuple[torch.Tensor, int]:
    """Load and resample audio file.

    Args:
        path: Path to the audio file.
        target_sr: Target sample rate for resampling.

    Returns:
        Tuple of (audio_tensor, sample_rate) where audio_tensor has shape
        (channels, samples) and sample_rate equals target_sr.
    """
    audio, sr = torchaudio.load(path)

    # Convert to mono if stereo
    if audio.dim() == 2 and audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Duplicate mono to stereo for 48kHz (required by some models)
    if audio.shape[0] == 1 and target_sr == SAMPLE_RATE_48K:
        audio = audio.repeat(2, 1)

    # Resample if necessary
    if sr != target_sr:
        cache_key = f"{sr}_{target_sr}"
        if cache_key not in _resampler_cache:
            _resampler_cache[cache_key] = torchaudio.transforms.Resample(sr, target_sr)
        audio = _resampler_cache[cache_key](audio)

    return audio, target_sr


def load_midi_extractor(
    midi_extractor: nn.Module,
    midi_extractor_path: str | None = None,
    device: str = "cpu",
) -> nn.Module:
    """Load pretrained weights into MIDI extractor model.

    Args:
        midi_extractor: The MIDI extractor model instance.
        midi_extractor_path: Path to the checkpoint file.
        device: Target device for the model.

    Returns:
        The loaded MIDI extractor model.

    Raises:
        ValueError: If midi_extractor_path is None.
    """
    from collections import OrderedDict

    if midi_extractor_path is None:
        raise ValueError("midi_extractor_path is required")

    state_dict = torch.load(midi_extractor_path, map_location="cpu")["state_dict"]
    prefix_in_ckpt = "model.model"
    state_dict = OrderedDict(
        {
            k.replace(f"{prefix_in_ckpt}.", "midi_conform."): v
            for k, v in state_dict.items()
            if k.startswith(f"{prefix_in_ckpt}.")
        }
    )
    midi_extractor.load_state_dict(state_dict, strict=True)
    midi_extractor.to(device)
    return midi_extractor


def get_notemidi_from_file(midi_file: Union[str, PathLike]) -> np.ndarray:
    """Extract note MIDI sequence from a MIDI file.

    Converts a MIDI file into a frame-by-frame MIDI pitch sequence at 100Hz.

    Args:
        midi_file: Path to the MIDI file.

    Returns:
        NumPy array of MIDI pitch values at each frame (100Hz frame rate).
        Frames without notes have value 0.
    """
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = []

    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append(
                {
                    "pitch": note.pitch,
                    "start": note.start,
                    "end": note.end,
                    "velocity": note.velocity,
                    "instrument": instrument.name,
                }
            )

    total_frames = int(midi_data.get_end_time() * DEFAULT_FRAME_HZ)
    notemidi_seq = np.zeros(total_frames, dtype=float)

    for note in notes:
        start_frame = int(note["start"] * DEFAULT_FRAME_HZ)
        end_frame = int(note["end"] * DEFAULT_FRAME_HZ)
        notemidi_seq[start_frame:end_frame] = note["pitch"]

    return notemidi_seq


def load_vae_model(
    ckpt_path: Union[str, PathLike],
    config_path: Union[str, PathLike],
) -> nn.Module:
    """Load a pretrained VAE model from checkpoint.

    Args:
        ckpt_path: Path to the VAE checkpoint file.
        config_path: Path to the VAE configuration JSON file.

    Returns:
        The loaded VAE model in evaluation mode.
    """
    with open(config_path) as f:
        config = json.load(f)

    vae_model = create_autoencoder_from_config(config)
    vae_model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    vae_model.eval()
    return vae_model


# =============================================================================
# Main Model
# =============================================================================


class YingSinger(nn.Module):
    """Zero-shot singing voice synthesis and editing model.

    YingSinger is a unified framework for zero-shot singing voice synthesis (SVS)
    and editing, driven by annotation-free melody guidance. It uses a Diffusion
    Transformer (DiT) based generative model with pre-trained melody extraction.

    Args:
        singer_path: Path to model checkpoint directory. If None, downloads from
            HuggingFace Hub.
        device: Target device for model inference ("cpu", "cuda", or specific GPU).
        cache_dir: Directory for caching downloaded model files.

    Example:
        >>> singer = YingSinger(device="cuda")
        >>> audio = singer.inference(
        ...     timbre_audio_path="reference.wav",
        ...     timbre_audio_content="hello world",
        ...     melody_audio_path="melody.wav",
        ...     lyrics="new lyrics here",
        ... )
    """

    # HuggingFace model repository
    HF_REPO_ID = "GiantAILab/YingMusic-Singer"

    def __init__(
        self,
        singer_path: Union[str, PathLike, None] = None,
        device: Union[str, torch.device] = "cpu",
        cache_dir: Union[str, PathLike, None] = "./.cache/huggingface",
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self._device = device

        self._init_model(singer_path, device=device)

        self.tokenizer = PhonemeBpeTokenizer(g2p_device=device)
        self.mel_spectrogram = RMVPEMelSpectrogram(**RMVPE_MEL_CONFIG)
        self.feat_dim = DEFAULT_FEATURE_DIM

    def _init_model(
        self,
        singer_path: Union[str, PathLike, None] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """Initialize model components from checkpoint.

        Args:
            singer_path: Path to checkpoint directory.
            device: Target device for model.
        """
        if singer_path is None:
            singer_path = hf_hub_download(
                repo_id=self.HF_REPO_ID,
                cache_dir=self.cache_dir,
            )

        config_path = Path(__file__).parent / "config" / "singer.yaml"
        cfg = OmegaConf.load(config_path)

        # Initialize decoder
        decoder = hydra.utils.instantiate(cfg.decoder)
        self.singer = nn.Module()
        self.singer.decoder = decoder

        # Initialize melody extractor
        self.melody_extractor = hydra.utils.instantiate(cfg.melody_encoder)

        # Load singer model weights
        stats = torch.load(
            os.path.join(singer_path, "model_485000_ema_fixed.pt"),
            map_location="cpu",
            weights_only=True,
        )
        self.singer.load_state_dict(stats, strict=True)
        self.singer.to(device)

        # Load VAE
        vae = load_vae_model(
            ckpt_path=os.path.join(singer_path, "autoencoder_music_dsp1920.ckpt"),
            config_path=os.path.join(singer_path, "stable_audio_1920_vae.json"),
        )
        self.vae = vae.to(device)

        # Load F0 extractor
        self.f0_extractor = RMVPE(model_path=os.path.join(singer_path, "rmvpe.pt"), device=device)

        # Load melody extractor
        self.melody_extractor = load_midi_extractor(
            midi_extractor=self.melody_extractor,
            midi_extractor_path=os.path.join(singer_path, "some.pt"),
            device=device,
        )
        print("Model initialized.")

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        return next(self.singer.decoder.parameters()).device

    def extract_melody(self, audio_path: Union[str, PathLike]) -> torch.Tensor:
        """Extract MIDI melody sequence from audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Tensor of MIDI note values for each frame.
        """
        audio_44k, _ = load_audio(audio_path, SAMPLE_RATE_44K)
        audio_16k, _ = load_audio(audio_path, SAMPLE_RATE_16K)

        audio_mel = self.mel_spectrogram(audio_44k).permute(0, 2, 1).to(self.device)
        audio_f0 = self.f0_extractor.infer_from_audio(audio_16k)

        # Align F0 length with mel spectrogram
        if audio_f0.shape[1] != audio_mel.shape[1]:
            audio_f0 = F.interpolate(audio_f0.unsqueeze(1), size=audio_mel.shape[1], mode="nearest").squeeze(1)

        audio_masks = (audio_f0 > 1).bool()
        note_midi = self.melody_extractor.get_notemidi_seq(audio_mel, masks=audio_masks)

        return note_midi

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        text: torch.Tensor | list[str],
        duration: int | torch.Tensor | None = None,
        *,
        melody_in: torch.Tensor | None = None,
        task_tag: str = "singing",
        lens: torch.Tensor | None = None,
        steps: int = 32,
        text_strength: float = 1.0,
        seed: int | None = None,
        max_duration: int = 4096,
        no_ref_audio: bool = False,
        edit_mask: torch.Tensor | None = None,
        sde_strength: float = 0,
        sde_window: list[float] | None = None,
        disable_pbar: bool = False,
        schedule_method: str = "cosine",
        schedule_power: float = 2.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from the diffusion model.

        Generates audio features using the diffusion transformer with optional
        classifier-free guidance and SDE noise injection.

        Args:
            cond: Conditioning audio features of shape (batch, seq_len, feat_dim).
            text: Text tokens or list of tokenized strings.
            duration: Target output duration in frames. Defaults to 2x conditioning length.
            melody_in: Optional melody MIDI sequence for conditioning.
            task_tag: Task identifier ("singing" or "speech").
            lens: Actual lengths of each sequence in batch.
            steps: Number of diffusion sampling steps.
            text_strength: Classifier-free guidance strength for text.
            seed: Random seed for reproducible sampling.
            max_duration: Maximum allowed duration in frames.
            no_ref_audio: If True, ignore reference audio conditioning.
            edit_mask: Optional mask for selective editing.
            sde_strength: Stochastic noise strength (0 for pure ODE).
            sde_window: Time window [start, end] for SDE noise injection.
            disable_pbar: If True, disable progress bar.
            schedule_method: Time schedule method ("linear", "cosine", or "power").
            schedule_power: Exponent for power scheduling.

        Returns:
            Tuple of (generated_features, trajectory) where generated_features
            is the final sample and trajectory contains all intermediate steps.
        """
        if sde_window is None:
            sde_window = [0.0, 0.5]

        self.eval()

        cond = cond.to(self.device)
        cond = cond.to(next(self.parameters()).dtype)

        duration = default(duration, cond.shape[1] * 2)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text

        text = list_str_to_idx(text, self.tokenizer.vocab).to(device)

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)

        if no_ref_audio:
            cond = torch.zeros_like(cond)

        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))

        if no_ref_audio:
            step_cond = torch.zeros((1, duration, self.feat_dim), device=device, dtype=cond.dtype)

        melody = melody_in

        if melody is not None:
            if melody.shape[1] != cond.shape[1]:
                melody_cond = F.interpolate(melody.unsqueeze(1), size=cond.shape[1], mode="nearest")
                melody_cond = melody_cond.squeeze(1)
            else:
                melody_cond = melody

            melody_cond = melody_cond.round().long()

        else:
            melody_cond = None

        # diffusion process
        def fn(t, x):
            if text_strength < 1e-5:
                pred = self.singer.decoder(
                    x=x,
                    cond=step_cond,
                    text=text,
                    melody=melody_cond,
                    tags=[task_tag],
                    time=t,
                    mask=None,
                    drop_audio_cond=False,
                    drop_text=False,
                    drop_melody=False,
                    cache=False,
                )
                return pred

            pred_cfg = self.singer.decoder(
                x=x,
                cond=step_cond,
                text=text,
                melody=melody_cond,
                tags=[task_tag],
                time=t,
                mask=None,
                cfg_infer=True,
                cache=False,
            )
            pred, pred_wo_content = torch.chunk(pred_cfg, 2, dim=0)
            ret = pred
            if text_strength > 1e-5:
                ret = ret + (pred - pred_wo_content) * text_strength

            return ret

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.feat_dim, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_span = get_time_span(
            method=schedule_method,
            t_start=0.0,
            t_end=1.0,
            steps=steps,
            device=self.device,
            power=schedule_power,
            dtype=step_cond.dtype,
        )

        x = y0
        trajectory = [x]

        progress_bar = tqdm(range(steps), desc="Sampling", total=steps, disable=disable_pbar)
        for i in range(steps):
            t_curr = t_span[i]
            t_next = t_span[i + 1]
            dt = t_next - t_curr  # time step

            # drift (Velocity)
            v_pred = fn(t_curr, x)

            # ODE : x = x + v * dt
            # SDE : x = x + v * dt + sigma * noise * sqrt(dt)

            # drift
            d_x = v_pred * dt

            # diffusion
            if sde_strength > 0:
                noise = torch.randn_like(x)
                # diffusion_term = sigma * dW, dW ~ N(0, dt) = N(0,1) * sqrt(dt)
                sigma = sde_strength

                if exists(sde_window):
                    if t_curr < sde_window[0] or t_curr > sde_window[1]:
                        sigma = 0.0

                diffusion_term = sigma * noise * torch.sqrt(dt)
                d_x = d_x + diffusion_term

            x = x + d_x
            trajectory.append(x)
            progress_bar.update()

        trajectory = torch.stack(trajectory)
        return trajectory[-1], trajectory

    def encode_audio(
        self,
        audio_path: Union[str, PathLike],
        target_sr: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode audio file to VAE latent features.

        Processes audio in 60-second chunks to manage memory for long files.

        Args:
            audio_path: Path to the audio file.
            target_sr: Target sample rate for loading.
            device: Device for encoding.

        Returns:
            Encoded audio features tensor.
        """
        audio, _ = load_audio(audio_path, target_sr)
        audio_features = []
        chunk_size = target_sr * 60  # 60 seconds per chunk
        total_samples = audio.shape[-1]

        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk_audio = audio[:, start_idx:end_idx].unsqueeze(0).to(device)
            chunk_feat = self.vae.encode_audio(chunk_audio)
            audio_features.append(chunk_feat.cpu())

        return torch.cat(audio_features, dim=-1)

    @torch.inference_mode()
    def inference(
        self,
        timbre_audio_path: Union[str, PathLike],
        timbre_audio_content: str,
        lyrics: str,
        *,
        melody_audio_path: Optional[Union[str, PathLike]] = None,
        midi_file: Optional[Union[str, PathLike]] = None,
        cfg_strength: float = 4.0,
        nfe_steps: int = 32,
        seed: Optional[int] = DEFAULT_SEED,
    ) -> torch.Tensor:
        """Generate singing voice from reference audio and lyrics.

        Performs zero-shot singing voice synthesis by combining timbre from
        reference audio with melody from either a melody audio file or MIDI file.

        Args:
            timbre_audio_path: Path to audio file for voice timbre reference.
            timbre_audio_content: Text content of the timbre reference audio.
            lyrics: Target lyrics to synthesize.
            melody_audio_path: Path to audio file for melody reference.
                Either this or midi_file must be provided.
            midi_file: Path to MIDI file for melody reference.
                Either this or melody_audio_path must be provided.
            cfg_strength: Classifier-free guidance strength for text conditioning.
            nfe_steps: Number of function evaluations (diffusion steps).
            seed: Random seed for reproducibility.

        Returns:
            Generated audio waveform tensor at 48kHz.

        Raises:
            ValueError: If neither melody_audio_path nor midi_file is provided.
        """
        seed_everything(seed)

        # Encode timbre reference
        timbre_audio_feat = self.encode_audio(timbre_audio_path, SAMPLE_RATE_48K, self.device)
        timbre_audio_feat = timbre_audio_feat.to(self.device).permute(0, 2, 1)
        timbre_melody = self.extract_melody(timbre_audio_path)
        timbre_audio_len = timbre_audio_feat.shape[1]

        # Prepare text input
        text = timbre_audio_content.strip() + " " + lyrics.strip()
        tokenized_text = self.tokenizer.tokenize(text, "", language="auto")[0]
        text_tokens = tokenized_text.split("|")

        # Process melody source
        if melody_audio_path is not None:
            melody_audio_feat = self.encode_audio(melody_audio_path, SAMPLE_RATE_48K, self.device)
            melody = self.extract_melody(melody_audio_path)
            melody_len = melody_audio_feat.shape[-1]

        elif midi_file is not None:
            notemidi_seq = get_notemidi_from_file(midi_file)
            melody = torch.from_numpy(notemidi_seq).unsqueeze(0).to(torch.float32)
            melody_len = melody.shape[1] // 4
        else:
            raise ValueError("Either melody_audio_path or midi_file must be provided.")

        # Combine timbre and melody
        total_duration = timbre_audio_len + melody_len
        combined_melody = torch.cat([timbre_melody, melody], dim=1).to(self.device)

        # Generate
        generated, _ = self.sample(
            cond=timbre_audio_feat,
            text=[text_tokens],
            duration=total_duration,
            melody_in=combined_melody,
            task_tag="singing",
            steps=nfe_steps,
            text_strength=cfg_strength,
            seed=seed,
        )

        # Decode to waveform
        generated = generated.to(torch.float32)
        generated = generated[:, timbre_audio_len:, :]  # Remove reference portion
        generated = generated.permute(0, 2, 1).squeeze(0)
        generated_wave = self.vae.decode_audio(generated.to(self.device)).cpu().squeeze(0)

        return generated_wave


# =============================================================================
# CLI Interface
# =============================================================================


def main():
    """Command-line interface for YingSinger inference."""
    import argparse

    parser = argparse.ArgumentParser(
        description="YingSinger: Zero-shot Singing Voice Synthesis and Editing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--timbre_audio_path",
        type=str,
        required=True,
        help="Path to timbre reference audio file.",
    )
    parser.add_argument(
        "--timbre_audio_content",
        type=str,
        required=True,
        help="Text content of the timbre reference audio.",
    )
    parser.add_argument(
        "--lyrics",
        type=str,
        required=True,
        help="Target lyrics to synthesize.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to save output audio file.",
    )

    # Melody source (one required)
    melody_group = parser.add_mutually_exclusive_group(required=True)
    melody_group.add_argument(
        "--melody_audio_path",
        type=str,
        default=None,
        help="Path to melody reference audio file.",
    )
    melody_group.add_argument(
        "--midi_file",
        type=str,
        default=None,
        help="Path to MIDI file for melody.",
    )

    # Optional arguments
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to model checkpoint directory. Downloads from HuggingFace if not provided.",
    )
    parser.add_argument(
        "--cfg_strength",
        type=float,
        default=3.0,
        help="Classifier-free guidance strength.",
    )
    parser.add_argument(
        "--nfe_steps",
        type=int,
        default=32,
        help="Number of diffusion steps (NFE).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    singer = YingSinger(singer_path=args.ckpt_path, device=device)

    # Run inference
    generated_audio = singer.inference(
        timbre_audio_path=args.timbre_audio_path,
        timbre_audio_content=args.timbre_audio_content,
        melody_audio_path=args.melody_audio_path,
        midi_file=args.midi_file,
        lyrics=args.lyrics,
        cfg_strength=args.cfg_strength,
        nfe_steps=args.nfe_steps,
        seed=args.seed,
    )

    # Save output
    torchaudio.save(args.out_path, generated_audio, SAMPLE_RATE_48K)
    print(f"Saved output to: {args.out_path}")


if __name__ == "__main__":
    main()
