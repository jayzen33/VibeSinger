from __future__ import annotations

import os
import random
from collections import defaultdict

import jieba
import torch
import torch.nn.functional as F
from pypinyin import Style, lazy_pinyin
from torch.nn.utils.rnn import pad_sequence

# seed everything


def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# helpers


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def is_package_available(package_name: str) -> bool:
    try:
        import importlib

        package_exists = importlib.util.find_spec(package_name) is not None
        return package_exists
    except Exception:
        return False


# tensor helpers


def lens_to_mask(t: int["b"], length: int | None = None) -> bool["b n"]:  # noqa: F722 F821
    if not exists(length):
        length = t.amax()

    seq = torch.arange(length, device=t.device)
    return seq[None, :] < t[:, None]


def mask_from_start_end_indices(seq_len: int["b"], start: int["b"], end: int["b"]):  # noqa: F722 F821
    max_seq_len = seq_len.max().item()
    seq = torch.arange(max_seq_len, device=start.device).long()
    start_mask = seq[None, :] >= start[:, None]
    end_mask = seq[None, :] < end[:, None]
    return start_mask & end_mask


def mask_from_frac_lengths(seq_len: int["b"], frac_lengths: float["b"]):  # noqa: F722 F821
    lengths = (frac_lengths * seq_len).long()
    max_start = seq_len - lengths

    rand = torch.rand_like(frac_lengths)
    start = (max_start * rand).long().clamp(min=0)
    end = start + lengths

    return mask_from_start_end_indices(seq_len, start, end)


def maybe_masked_mean(t: float["b n d"], mask: bool["b n"] = None) -> float["b d"]:  # noqa: F722
    if not exists(mask):
        return t.mean(dim=1)

    t = torch.where(mask[:, :, None], t, torch.tensor(0.0, device=t.device))
    num = t.sum(dim=1)
    den = mask.float().sum(dim=1)

    return num / den.clamp(min=1.0)


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")]) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    list_idx_tensors = [torch.tensor([vocab_char_map.get(c, 0) for c in t]) for t in text]  # pinyin or char style
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


# Get tokenizer


def get_tokenizer(dataset_name: str, tokenizer: str = "pinyin") -> tuple[dict[str, int] | None, int]:
    """Load a tokenizer vocabulary from file.

    Args:
        dataset_name: Path to vocab file when tokenizer="custom", otherwise ignored.
        tokenizer: Tokenizer type. One of:
            - "pinyin": G2P for Chinese characters, requires vocab.txt
            - "char": Character-wise tokenizer, requires vocab.txt
            - "byte": UTF-8 byte tokenizer (vocab_size=256)
            - "custom": Load vocab from path specified in dataset_name

    Returns:
        Tuple of (vocab_char_map, vocab_size) where vocab_char_map maps
        characters to indices, or None for byte tokenizer.

    Raises:
        FileNotFoundError: If vocab file doesn't exist.
        ValueError: If vocab file is invalid (space must be at index 0).
    """
    if tokenizer in ["pinyin", "char"]:
        # For pinyin/char tokenizers, vocab file should be provided via dataset_name
        # or use the custom tokenizer type with explicit path
        raise ValueError(
            f"Tokenizer '{tokenizer}' requires a vocab file. "
            "Please use tokenizer='custom' with dataset_name pointing to your vocab.txt file."
        )
        # Legacy code for reference:
        # tokenizer_path = os.path.join(data_dir, f"{dataset_name}_{tokenizer}/vocab.txt")
        # with open(tokenizer_path, "r", encoding="utf-8") as f:
        #     vocab_char_map = {}
        #     for i, char in enumerate(f):
        #         vocab_char_map[char[:-1]] = i
        # vocab_size = len(vocab_char_map)
        # assert vocab_char_map[" "] == 0, "Space must be at index 0 in vocab.txt"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    elif tokenizer == "custom":
        with open(dataset_name, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char[:-1]] = i
        vocab_size = len(vocab_char_map)

    return vocab_char_map, vocab_size


# convert char to pinyin


def convert_char_to_pinyin(text_list, polyphone=True, with_tone=True):
    if with_tone:
        style = Style.TONE3  # with tone number
    else:
        style = Style.NORMAL  # no tone

    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)  # CRITICAL
        jieba.initialize()

    final_text_list = []
    custom_trans = str.maketrans(
        {";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"}
    )  # add custom trans here, to address oov

    def is_chinese(c):
        return (
            "\u3100" <= c <= "\u9fff"  # common chinese characters
        )

    for text in text_list:
        char_list = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # if pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # if pure east asian characters
                seg_ = lazy_pinyin(seg, style=style, tone_sandhi=True)
                for i, c in enumerate(seg):
                    if is_chinese(c):
                        char_list.append(" ")
                    char_list.append(seg_[i])
            else:  # if mixed characters, alphabets and symbols
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif is_chinese(c):
                        char_list.append(" ")
                        char_list.extend(lazy_pinyin(c, style=style, tone_sandhi=True))
                    else:
                        char_list.append(c)

        if with_tone is False:
            for idx, item in enumerate(char_list):
                char_list[idx] = "__" + item

        final_text_list.append(char_list)

    return final_text_list


# filter func for dirty data with many repetitions


def repetition_found(text, length=2, tolerance=10):
    pattern_count = defaultdict(int)
    for i in range(len(text) - length + 1):
        pattern = text[i : i + length]
        pattern_count[pattern] += 1
    for pattern, count in pattern_count.items():
        if count > tolerance:
            return True
    return False


# get the empirically pruned step for sampling


def get_epss_timesteps(n, device, dtype):
    dt = 1 / 32
    predefined_timesteps = {
        5: [0, 2, 4, 8, 16, 32],
        6: [0, 2, 4, 6, 8, 16, 32],
        7: [0, 2, 4, 6, 8, 16, 24, 32],
        10: [0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32],
        16: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32],
    }
    t = predefined_timesteps.get(n, [])
    if not t:
        return torch.linspace(0, 1, n + 1, device=device, dtype=dtype)
    return dt * torch.tensor(t, device=device, dtype=dtype)


def calculate_similarity_matrix_with_mask(vectors: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    if valid_mask is None:
        valid_mask = torch.ones(vectors.shape[:-1], dtype=torch.bool, device=vectors.device)

    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.bool()

    vectors = vectors * valid_mask.unsqueeze(-1).float()

    vectors_normalized = F.normalize(vectors, p=2, dim=-1, eps=1e-8)

    # (B, N, D) * (B, D, N) -> (B, N, N)
    similarity_matrix = torch.bmm(vectors_normalized, vectors_normalized.transpose(1, 2))

    # (B, N, 1) & (B, 1, N) -> (B, N, N)
    combined_mask = valid_mask.unsqueeze(2) & valid_mask.unsqueeze(1)

    similarity_matrix.masked_fill_(~combined_mask, 0.0)

    return similarity_matrix


def _center_gram(gram: torch.Tensor) -> torch.Tensor:
    gram = gram - torch.mean(gram, dim=1, keepdim=True)
    gram = gram - torch.mean(gram, dim=0, keepdim=True)
    return gram


def cka_loss(sim_x: torch.Tensor, sim_y: torch.Tensor) -> torch.Tensor:
    losses = []
    for i in range(sim_x.shape[0]):
        sim_x_i = _center_gram(sim_x[i])
        sim_y_i = _center_gram(sim_y[i])

        hsic = torch.sum(sim_x_i * sim_y_i)
        norm_x = torch.sqrt(torch.sum(sim_x_i**2))
        norm_y = torch.sqrt(torch.sum(sim_y_i**2))

        cka_similarity = hsic / ((norm_x * norm_y) + 1e-6)

        losses.append(1.0 - cka_similarity)

    return torch.mean(torch.stack(losses))
