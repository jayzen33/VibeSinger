import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from singer.melody.Gconform import Gmidi_conform, conform_blocke

# midi decoding utils


def decode_gaussian_blurred_probs(probs, vmin, vmax, deviation, threshold):
    num_bins = int(probs.shape[-1])
    interval = (vmax - vmin) / (num_bins - 1)
    width = int(3 * deviation / interval)  # 3 * sigma
    idx = torch.arange(num_bins, device=probs.device)[None, None, :]  # [1, 1, N]
    idx_values = idx * interval + vmin
    center = torch.argmax(probs, dim=-1, keepdim=True)  # [B, T, 1]
    start = torch.clip(center - width, min=0)  # [B, T, 1]
    end = torch.clip(center + width + 1, max=num_bins)  # [B, T, 1]
    idx_masks = (idx >= start) & (idx < end)  # [B, T, N]
    weights = probs * idx_masks  # [B, T, N]
    product_sum = torch.sum(weights * idx_values, dim=2)  # [B, T]
    weight_sum = torch.sum(weights, dim=2)  # [B, T]
    values = product_sum / (weight_sum + (weight_sum == 0))  # avoid dividing by zero, [B, T]
    rest = probs.max(dim=-1)[0] < threshold  # [B, T]
    return values, rest


def decode_bounds_to_alignment(bounds, use_diff=True):
    bounds_step = bounds.cumsum(dim=1).round().long()
    if use_diff:
        bounds_inc = (
            torch.diff(
                bounds_step,
                dim=1,
                prepend=torch.full(
                    (bounds.shape[0], 1),
                    fill_value=-1,
                    dtype=bounds_step.dtype,
                    device=bounds_step.device,
                ),
            )
            > 0
        )
    else:
        bounds_inc = F.pad((bounds_step[:, 1:] > bounds_step[:, :-1]), [1, 0], value=True)
    frame2item = bounds_inc.long().cumsum(dim=1)
    return frame2item


def decode_note_sequence(frame2item, values, masks, threshold=0.5):
    """

    :param frame2item: [1, 1, 1, 1, 2, 2, 3, 3, 3]
    :param values:
    :param masks:
    :param threshold: minimum ratio of unmasked frames required to be regarded as an unmasked item
    :return: item_values, item_dur, item_masks
    """
    b = frame2item.shape[0]
    space = frame2item.max() + 1

    item_dur = frame2item.new_zeros(b, space, dtype=frame2item.dtype).scatter_add(
        1, frame2item, torch.ones_like(frame2item)
    )[:, 1:]
    item_unmasked_dur = frame2item.new_zeros(b, space, dtype=frame2item.dtype).scatter_add(
        1, frame2item, masks.long()
    )[:, 1:]
    item_masks = item_unmasked_dur / item_dur >= threshold

    values_quant = values.round().long()
    histogram = (
        frame2item.new_zeros(b, space * 128, dtype=frame2item.dtype)
        .scatter_add(1, frame2item * 128 + values_quant, torch.ones_like(frame2item) * masks)
        .unflatten(1, [space, 128])[:, 1:, :]
    )
    item_values_center = histogram.float().argmax(dim=2).to(dtype=values.dtype)
    values_center = torch.gather(F.pad(item_values_center, [1, 0]), 1, frame2item)
    values_near_center = masks & (values >= values_center - 0.5) & (values <= values_center + 0.5)
    item_valid_dur = frame2item.new_zeros(b, space, dtype=frame2item.dtype).scatter_add(
        1, frame2item, values_near_center.long()
    )[:, 1:]
    item_values = values.new_zeros(b, space, dtype=values.dtype).scatter_add(
        1, frame2item, values * values_near_center
    )[:, 1:] / (item_valid_dur + (item_valid_dur == 0))

    return item_values, item_dur, item_masks


def expand_batch_padded(feature_tensor, counts_tensor, padding_value=0.0):
    assert feature_tensor.dim() == 2 and counts_tensor.dim() == 2

    lengths = torch.sum(counts_tensor, dim=1)

    feature_tensor = feature_tensor.reshape(-1)
    counts_tensor = counts_tensor.reshape(-1)
    expanded_flat = torch.repeat_interleave(feature_tensor, counts_tensor)

    ragged_list = torch.split(expanded_flat, lengths.tolist())

    padded_tensor = pad_sequence(ragged_list, batch_first=True, padding_value=padding_value)

    return padded_tensor, lengths


class midi_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, x, target):
        midiout, cutp = x
        midi_target, cutp_target = target

        cutploss = self.loss(cutp, cutp_target)
        midiloss = self.loss(midiout, midi_target)
        return midiloss, cutploss


class MIDIExtractor(nn.Module):
    def __init__(
        self,
        indim=80,
        outdim=128,
        attention_drop=0.1,
        attention_heads=8,
        attention_heads_dim=64,
        conv_drop=0.1,
        dim=512,
        ffn_latent_drop=0.1,
        ffn_out_drop=0.1,
        kernel_size=31,
        lay=8,
        use_lay_skip=True,
        **kwargs,
    ):
        super().__init__()

        _cfg = {
            "indim": indim,
            "outdim": outdim,
            "attention_drop": attention_drop,
            "attention_heads": attention_heads,
            "attention_heads_dim": attention_heads_dim,
            "conv_drop": conv_drop,
            "dim": dim,
            "ffn_latent_drop": ffn_latent_drop,
            "ffn_out_drop": ffn_out_drop,
            "kernel_size": kernel_size,
            "lay": lay,
            "use_lay_skip": use_lay_skip,
        }

        self.midi_conform = Gmidi_conform(**_cfg)

        self.midi_min = 0
        self.midi_max = 127
        self.midi_deviation = 1.0
        self.rest_threshold = 0.1

    @property
    def device(self):
        return next(self.parameters()).device

    def _load_form_ckpt(self, ckpt_path, device="cpu"):
        from collections import OrderedDict

        if ckpt_path is None:
            raise ValueError("midi_extractor_path is required")

        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        prefix_in_ckpt = "model.model"
        state_dict = OrderedDict(
            {
                k.replace(f"{prefix_in_ckpt}.", "midi_conform."): v
                for k, v in state_dict.items()
                if k.startswith(f"{prefix_in_ckpt}.")
            }
        )
        self.load_state_dict(state_dict, strict=True)
        self.to(device)

    def forward(self, x, mask=None):
        midi, bound = self.midi_conform(x, mask)

        return midi, bound

    def postprocess(self, midi_probs):
        probs = torch.sigmoid(midi_probs)

        # Avoid in-place ops on tensors needed for autograd (outputs of SigmoidBackward)
        midi_pred, rest_pred = decode_gaussian_blurred_probs(
            probs,
            vmin=self.midi_min,
            vmax=self.midi_max,
            deviation=self.midi_deviation,
            threshold=self.rest_threshold,
        )

        return midi_pred

    def get_melody(self, x, mask=None):
        midi, bounds = self.forward(x, None)
        melody = self.postprocess(midi)
        return melody

    def get_notemidi_seq(self, x, masks=None):
        probs, _ = self.forward(x, None)
        probs = torch.sigmoid(probs)

        if masks is None:
            masks = torch.ones(probs.shape[0], probs.shape[1], device=probs.device).bool()
        probs *= masks[..., None]
        midi_pred, rest_pred = decode_gaussian_blurred_probs(
            probs,
            vmin=self.midi_min,
            vmax=self.midi_max,
            deviation=self.midi_deviation,
            threshold=self.rest_threshold,
        )
        return midi_pred


class MelodyEncoder(nn.Module):
    def __init__(
        self,
        indim=80,
        outdim=128,
        dim=512,
        lay=6,
        kernel_size=31,
        conv_drop=0.1,
        ffn_latent_drop=0.1,
        ffn_out_drop=0.1,
        attention_drop=0.1,
        attention_heads=8,
        attention_heads_dim=64,
        **kwargs,
    ):
        super().__init__()
        self.input_proj = nn.Linear(indim, dim)
        self.layers = nn.ModuleList(
            [
                conform_blocke(
                    dim=dim,
                    kernel_size=kernel_size,
                    conv_drop=conv_drop,
                    ffn_latent_drop=ffn_latent_drop,
                    ffn_out_drop=ffn_out_drop,
                    attention_drop=attention_drop,
                    attention_heads=attention_heads,
                    attention_heads_dim=attention_heads_dim,
                )
                for _ in range(lay)
            ]
        )
        self.output_proj = nn.Linear(dim, outdim)

        self.midi_min = 0
        self.midi_max = 127
        self.midi_deviation = 1.0
        self.rest_threshold = 0.1

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)

        for layer in self.layers:
            x = layer(x, mask)
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), 0)

        return self.output_proj(x)

    def postprocess(self, midi_probs):
        probs = torch.sigmoid(midi_probs)

        # Avoid in-place ops on tensors needed for autograd (outputs of SigmoidBackward)
        midi_pred, rest_pred = decode_gaussian_blurred_probs(
            probs,
            vmin=self.midi_min,
            vmax=self.midi_max,
            deviation=self.midi_deviation,
            threshold=self.rest_threshold,
        )

        return midi_pred

    def get_notemidi_seq(self, x, masks=None):
        probs = self.forward(x, None)
        probs = torch.sigmoid(probs)

        if masks is None:
            masks = torch.ones(probs.shape[0], probs.shape[1], device=probs.device).bool()
        probs *= masks[..., None]
        midi_pred, rest_pred = decode_gaussian_blurred_probs(
            probs,
            vmin=self.midi_min,
            vmax=self.midi_max,
            deviation=self.midi_deviation,
            threshold=self.rest_threshold,
        )
        return midi_pred
