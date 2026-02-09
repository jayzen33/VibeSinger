"""
DiT (Diffusion Transformer) implementation for VibeSinger.

This module contains the implementation of the Diffusion Transformer (DiT) used in VibeSinger,
including text embeddings, input embeddings, and the main DiT backbone.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from univoice.decoder.modules import (
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    Head,
    WanAttentionBlock,
    get_pos_embed_indices,
    precompute_freqs_cis,
    rope_params,
    sinusoidal_embedding_1d,
)

# Text embedding


class TextEmbedding(nn.Module):
    def __init__(
        self,
        text_num_embeds: int,
        text_dim: int,
        mask_padding: bool = False,
        conv_layers: int = 0,
        conv_mult: int = 2,
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        self.mask_padding = mask_padding  # mask filler and batch padding tokens or not

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer(
                "freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False
            )
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(
        self,
        text: Tensor,
        seq_len: int,
        drop_text: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)  # (opt.) if not self.average_upsampling:
        if self.mask_padding:
            text_mask = text == 0
        else:
            text_mask = torch.zeros((batch, seq_len), device=text.device, dtype=torch.bool)

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), device=text.device, dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            if self.mask_padding:
                text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
                for block in self.text_blocks:
                    text = block(text)
                    text = text.masked_fill(text_mask.unsqueeze(-1).expand(-1, -1, text.size(-1)), 0.0)
            else:
                text = self.text_blocks(text)

        return text, text_mask


# noised input audio and context mixing embedding


class InputEmbedding(nn.Module):
    def __init__(
        self,
        mel_dim: int,
        text_dim: int,
        out_dim: int,
        melody_num_embeds: int = 128,
        melody_dim: int = 128,
        tag_dim: int = 4,
    ):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim + melody_dim + tag_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

        self.melody_embed = nn.Embedding(melody_num_embeds + 1, melody_dim)
        self.melody_proj = nn.Linear(melody_dim, melody_dim)
        self.null_melody = nn.Parameter(torch.randn(1, 1, melody_dim))
        self.melody_dim = melody_dim

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        text_embed: Tensor,
        melody: Optional[Tensor],
        tag_embedding: Tensor,
        drop_audio_cond: bool = False,
        drop_melody: bool = False,
    ) -> Tensor:
        _batch, _seq_len, _ = x.shape

        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        if melody is None:
            # melody = torch.zeros((_batch, _seq_len, self.melody_dim), device=x.device, dtype=x.dtype)
            melody = self.null_melody.expand(x.size(0), x.size(1), 128)
        else:
            melody = melody + 1
            melody = self.melody_embed(melody)
            melody = self.melody_proj(melody)
            if drop_melody:  # cfg for melody
                melody = self.null_melody.expand(x.size(0), x.size(1), 128)

        tag_embedding = tag_embedding.unsqueeze(1).expand(-1, _seq_len, -1)

        x = self.proj(torch.cat((x, cond, text_embed, melody, tag_embedding), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int = 8,
        heads: int = 8,
        ff_mult: int = 4,
        freq_dim: int = 256,
        feat_dim: int = 100,
        text_num_embeds: int = 256,
        text_dim: Optional[int] = None,
        melody_num_embeds: int = 128,
        melody_dim: int = 128,
        tag_dim: int = 4,
        text_mask_padding: bool = True,
        qk_norm: Optional[bool] = None,
        conv_layers: int = 0,
    ):
        super().__init__()

        self.freq_dim = freq_dim
        self.time_embedding = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        if text_dim is None:
            text_dim = feat_dim

        self.text_embed = TextEmbedding(
            text_num_embeds,
            text_dim,
            mask_padding=text_mask_padding,
            conv_layers=conv_layers,
        )
        self.text_cond, self.text_uncond = None, None  # text cache
        self.input_embed = InputEmbedding(
            feat_dim, text_dim, dim, melody_num_embeds=melody_num_embeds, melody_dim=melody_dim, tag_dim=tag_dim
        )

        self.speech_tag = nn.Parameter(torch.randn(tag_dim))
        self.singing_tag = nn.Parameter(torch.randn(tag_dim))

        self.register_buffer("freqs", rope_params(4096, dim // heads), persistent=False)

        self.feat_dim = feat_dim
        self.dim = dim
        self.depth = depth

        if qk_norm is None:
            qk_norm = True

        self.transformer_blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    dim=dim,
                    ffn_dim=dim * ff_mult,
                    num_heads=heads,
                    window_size=(-1, -1),
                    qk_norm=qk_norm,
                    cross_attn_norm=False,
                    eps=1e-6,
                    task_dim=tag_dim,
                )
                for _ in range(depth)
            ]
        )

        # final modulation
        self.head = Head(dim, feat_dim, patch_size=(1,), eps=1e-6)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights for the model."""
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.text_embed.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)

        # zero init melody proj
        if hasattr(self.input_embed, "melody_proj"):
            nn.init.zeros_(self.input_embed.melody_proj.weight)
            nn.init.zeros_(self.input_embed.melody_proj.bias)

    def get_input_embed(
        self,
        x: Tensor,
        cond: Tensor,
        text: Tensor,
        melody: Optional[Tensor],
        tag_embedding: Tensor,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        drop_melody: bool = False,
        cache: bool = True,
        audio_mask: Optional[Tensor] = None,
    ) -> Tensor:
        seq_len = x.shape[1]

        if cache:
            if drop_text:
                if self.text_uncond is None:
                    self.text_uncond, _ = self.text_embed(text, seq_len, drop_text=True)
                text_embed = self.text_uncond
            else:
                if self.text_cond is None:
                    self.text_cond, _ = self.text_embed(text, seq_len, drop_text=False)
                text_embed = self.text_cond
        else:
            text_embed, text_mask = self.text_embed(text, seq_len, drop_text=drop_text)

        x = self.input_embed(
            x,
            cond,
            text_embed,
            melody,
            tag_embedding=tag_embedding,
            drop_audio_cond=drop_audio_cond,
            drop_melody=drop_melody,
        )

        return x

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        text: Tensor,
        time: Tensor | float,
        melody: Optional[Tensor] = None,
        tags: List[str] = None,
        mask: Optional[Tensor] = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        drop_melody: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
    ) -> Tensor:
        batch, seq_len = x.shape[0], x.shape[1]
        if isinstance(time, (int, float)):
            time = torch.tensor([time], device=x.device).repeat(batch)
        elif time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, text: text, x: noised audio + cond audio + text
        # time embeddings
        with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, time).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        if tags is None:
            tags = ["speech"] * batch

        tag_embeddings = []
        for tag in tags:
            if tag == "speech":
                tag_embeddings.append(self.speech_tag)
            elif tag == "singing":
                tag_embeddings.append(self.singing_tag)
            else:
                raise ValueError(f"Unknown tag: {tag}")
        tag_embedding = torch.stack(tag_embeddings, dim=0)

        if cfg_infer:  # pack cond & uncond forward: b n d -> 3b n d
            x_cond = self.get_input_embed(
                x,
                cond,
                text,
                melody,
                tag_embedding,
                drop_audio_cond=False,
                drop_text=False,
                drop_melody=False,
                cache=cache,
                audio_mask=mask,
            )
            x_content_uncond = self.get_input_embed(
                x,
                cond,
                text,
                melody,
                tag_embedding,
                drop_audio_cond=False,
                drop_text=True,
                drop_melody=False,
                cache=cache,
                audio_mask=mask,
            )

            x = torch.cat((x_cond, x_content_uncond), dim=0)
            e = torch.cat((e, e), dim=0)
            e0 = torch.cat((e0, e0), dim=0)
            tag_embedding = torch.cat((tag_embedding, tag_embedding), dim=0)

            mask = torch.cat((mask, mask), dim=0) if mask is not None else None
        else:
            x = self.get_input_embed(
                x,
                cond,
                text,
                melody,
                tag_embedding,
                drop_audio_cond=drop_audio_cond,
                drop_text=drop_text,
                drop_melody=drop_melody,
                cache=cache,
                audio_mask=mask,
            )

        if mask is not None:
            seq_lens = mask.sum(dim=1).to(dtype=torch.int32)
        else:
            seq_lens = torch.tensor([seq_len] * x.shape[0], device=x.device, dtype=torch.int32)

        for block in self.transformer_blocks:
            x = block(x, e0, seq_lens, self.freqs, task_embedding=tag_embedding)

        output = self.head(x, e)

        return output
