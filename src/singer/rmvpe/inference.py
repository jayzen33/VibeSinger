import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from singer.rmvpe.constants import MEL_FMAX, MEL_FMIN, N_MELS, SAMPLE_RATE, WINDOW_LENGTH
from singer.rmvpe.model import E2E0
from singer.rmvpe.spec import MelSpectrogram
from singer.rmvpe.utils import to_local_average_cents, to_local_average_cents_optimized, to_viterbi_cents


class RMVPE:
    def __init__(self, model_path, device=None, dtype=torch.float32, hop_length=160):
        self.resample_kernel = {}
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        model = E2E0(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location=torch.device(self.device))
        model.load_state_dict(ckpt["model"])
        model = model.to(dtype).to(self.device)
        model.eval()
        self.model = model
        self.dtype = dtype
        self.mel_extractor = MelSpectrogram(N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX)  # noqa: F405
        self.resample_kernel = {}

    def mel2hidden(self, mel):
        with torch.inference_mode():
            n_frames = mel.shape[-1]
            mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="constant")
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        if use_viterbi:
            cents_pred = to_viterbi_cents(hidden, thred=thred)
        else:
            cents_pred = to_local_average_cents(hidden, thred=thred)
        f0 = torch.Tensor([10 * (2 ** (cent_pred / 1200)) if cent_pred else 0 for cent_pred in cents_pred]).to(
            self.device
        )
        return f0

    def infer_from_audio(self, audio, sample_rate=16000, thred=0.05, use_viterbi=False):
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        audio = audio.to(self.dtype).to(self.device)

        if sample_rate == 16000:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, 16000, lowpass_filter_width=128)
            self.resample_kernel[key_str] = self.resample_kernel[key_str].to(self.dtype).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        mel_extractor = self.mel_extractor.to(self.device)
        mel = mel_extractor(audio_res, center=True).to(self.dtype)
        hidden = self.mel2hidden(mel)

        f0 = []
        for bib in range(hidden.shape[0]):
            f0.append(self.decode(hidden[bib], thred=thred, use_viterbi=use_viterbi))
        f0 = torch.stack(f0)

        return f0

    def infer_from_mel_legacy(self, mel, thred=0.05, use_viterbi=False):
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        mel = mel.to(self.dtype).to(self.device)
        hidden = self.mel2hidden(mel)

        f0 = []
        for bib in range(hidden.shape[0]):
            f0.append(self.decode(hidden[bib], thred=thred, use_viterbi=use_viterbi))
        f0 = torch.stack(f0)

        return f0

    def infer_from_mel_optimized(self, mel, thred=0.05):
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

        hidden = self.mel2hidden(mel)
        B, T, C = hidden.shape
        salience = hidden.reshape(B * T, C)

        cents_pred = to_local_average_cents_optimized(salience, thred=thred)
        cents_pred = cents_pred.reshape(B, T)

        f0 = torch.zeros_like(cents_pred)
        mask = cents_pred > 0
        f0[mask] = 10 * torch.pow(2, cents_pred[mask] / 1200)

        return f0


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torchaudio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Running test on device: {device}, with dtype: {dtype}")

    BATCH_SIZE = 16
    N_FRAMES = 1000

    # mock_mel = torch.rand(BATCH_SIZE, N_MELS, N_FRAMES, device=device, dtype=dtype)
    # print(f"\nCreated mock Mel Spectrogram with shape: {mock_mel.shape}")

    audio, sr = torchaudio.load("data/jaychou.wav")
    if audio.dim() == 2:
        audio = audio.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        audio = resampler(audio)

    rmvpe = RMVPE("opt/models/sovitssvc/rmvpe.pt", device=device, dtype=dtype, hop_length=160)

    mock_mel = rmvpe.mel_extractor(audio, center=True).to(dtype).to(device)

    print("\n--- Testing Legacy (Loop-based) Method ---")
    f0_legacy = rmvpe.infer_from_mel_legacy(mock_mel)

    print("\n--- Testing Optimized (Vectorized) Method ---")
    f0_optimized = rmvpe.infer_from_mel_optimized(mock_mel)

    print("\n--- Results Comparison ---")
    f0_legacy_cpu = f0_legacy.cpu().to(torch.float32).numpy()
    f0_optimized_cpu = f0_optimized.cpu().to(torch.float32).numpy()

    are_close = np.allclose(f0_legacy_cpu, f0_optimized_cpu, atol=1e-5)
    print(f"Outputs are numerically close: {are_close}")

    absolute_difference = np.abs(f0_legacy_cpu - f0_optimized_cpu)
    max_diff = np.max(absolute_difference)
    mean_diff = np.mean(absolute_difference)
    print(f"Max absolute difference: {max_diff:.10f} Hz")
    print(f"Mean absolute difference: {mean_diff:.10f} Hz")

    plt.figure(figsize=(18, 8))

    plt.subplot(2, 1, 1)
    plt.plot(f0_legacy_cpu[0, :], label="Legacy F0", color="blue", linewidth=3, alpha=0.7)
    plt.plot(f0_optimized_cpu[0, :], label="Optimized F0", color="red", linestyle="--", linewidth=1.5)
    plt.title("F0 Comparison (First sample in batch)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid(True, linestyle=":")

    plt.subplot(2, 1, 2)
    plt.plot(absolute_difference[0, :], label="Absolute Difference", color="green")
    plt.title("Absolute Difference between Legacy and Optimized")
    plt.xlabel("Frames")
    plt.ylabel("Frequency Difference (Hz)")
    plt.legend()
    plt.grid(True, linestyle=":")

    plt.tight_layout()
    plt.savefig("rmvpe_comparison.png", dpi=300)
