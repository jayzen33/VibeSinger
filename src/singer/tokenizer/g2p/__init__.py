# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib.resources as importlib_resources
import json
import os
import threading
from functools import partial
from typing import Dict, Iterable, List, Optional, Tuple, Union

import LangSegment

from singer.tokenizer.g2p import cleaners
from singer.tokenizer.g2p.text_tokenizers import TextTokenizer

__all__ = ["PhonemeBpeTokenizer"]


_LANGSEGMENT_LOCK = threading.Lock()


class _LazyTokenizerDict(dict):
    """Create TextTokenizer instances on first access.

    Keys use short language tags (e.g., 'en', 'zh').
    """

    def __init__(self, lang2backend: Dict[str, str], tokenizer_kwargs: Optional[dict] = None) -> None:
        super().__init__()
        self._lang2backend = lang2backend
        self._tokenizer_kwargs = tokenizer_kwargs or {}

    def __missing__(self, key: str) -> TextTokenizer:
        if key not in self._lang2backend:
            raise KeyError(f"Unknown language key: {key}")
        backend_lang = self._lang2backend[key]
        tok = TextTokenizer(language=backend_lang, **self._tokenizer_kwargs)
        self[key] = tok
        return tok


class PhonemeBpeTokenizer:
    """Convert text to phoneme BPE tokens with optional auto language splitting.

    Parameters
    - vocab_path: Optional path to vocab.json. If None, uses the packaged file.
    - cleaner_names: Cleaner pipeline to apply (default: ["cjekfd_cleaners"]).
    - g2p_device: Optional device specifier for Chinese polyphone predictor (e.g., 0, "cuda:1", "cpu").
    """

    def __init__(
        self,
        vocab_path: Optional[str] = None,
        cleaner_names: Optional[Iterable[str]] = None,
        enable_cache: bool = False,
        tokenizer_njobs: Optional[int] = None,
        g2p_device: Optional[Union[str, int]] = None,
    ) -> None:
        self.lang2backend: Dict[str, str] = {
            "zh": "cmn",
            "ja": "ja",
            "en": "en-us",
            "fr": "fr-fr",
            "ko": "ko",
            "de": "de",
        }
        # Determine phonemizer njobs from param or env
        if tokenizer_njobs is None:
            env_njobs = os.getenv("F5TTS_PHONEMIZER_NJOBS")
            if env_njobs is not None and env_njobs.isdigit():
                tokenizer_njobs = max(1, int(env_njobs))

        tokenizer_kwargs = {"njobs": tokenizer_njobs} if tokenizer_njobs else {}

        # Lazy-create tokenizers to reduce startup cost, passing kwargs
        self.text_tokenizers: Dict[str, TextTokenizer] = _LazyTokenizerDict(self.lang2backend, tokenizer_kwargs)

        # Load vocab.json
        self.vocab: Dict[str, int] = self._load_vocab(vocab_path)

        self._g2p_device: Optional[Union[str, int]] = g2p_device

        # Pre-resolve cleaner functions
        if cleaner_names is None:
            cleaner_names = ("cjekfd_cleaners",)
        self._cleaner_funcs = self._resolve_cleaners(list(cleaner_names))

        # Configure language segmenter once
        LangSegment.setfilters(["en", "zh", "ja", "ko", "fr", "de"])

        # Small memo cache to avoid recomputing identical requests in a session
        if enable_cache:
            self._cache: Dict[Tuple[str, str, str], str] = {}

    def _load_vocab(self, vocab_path: Optional[str]) -> Dict[str, int]:
        if vocab_path is None:
            # Use the packaged vocab.json co-located with this module
            try:
                with (
                    importlib_resources.files("singer.tokenizer.g2p")
                    .joinpath("vocab.json")
                    .open("r", encoding="utf-8") as f
                ):
                    return json.load(f)["vocab"]
            except Exception as e:
                raise FileNotFoundError("Failed to load packaged vocab.json. Provide vocab_path explicitly.") from e
        else:
            with open(vocab_path, "r", encoding="utf-8") as f:
                return json.load(f)["vocab"]

    def _resolve_cleaners(self, cleaner_names: List[str]):
        funcs = []
        for name in cleaner_names:
            fn = getattr(cleaners, name, None)
            if fn is None:
                raise ValueError(f"Unknown cleaner: {name}")
            funcs.append(partial(fn, tokenizer=self))
        return funcs

    def int_text_tokenizers(self):
        """Backwards-compatible no-op; tokenizers are created lazily now."""
        # Maintain compatibility if external code calls this method
        for key, value in self.lang2backend.items():
            _ = self.text_tokenizers[key]  # triggers lazy creation

    def tokenize(self, text: str, sentence: str, language: str):
        """Return tuple (phoneme string, list[int] or list[list[int]]).

        language: 'auto' or one of {zh, ja, en, fr, ko, de}
        """
        # 1) text -> phonemes (cache for repeated requests)
        phonemes = self._get_phonemes(text, sentence, language)

        # 2) phonemes -> token ids
        phoneme_tokens = self.phoneme2token(phonemes)
        return phonemes, phoneme_tokens

    def _get_phonemes(self, text: str, sentence: str, language: str) -> str:
        """Return phoneme string for the given text and language."""
        # This method can be used to get phonemes without tokenizing to IDs
        cache_key = (text, sentence, language)
        cached = self._cache.get(cache_key) if hasattr(self, "_cache") else None
        if cached is not None:
            return cached

        if language == "auto":
            with _LANGSEGMENT_LOCK:
                seglist = LangSegment.getTexts(text)
            parts: List[str] = []
            for seg in seglist:
                if seg["lang"] == "ja":
                    seg["lang"] = "zh"  # Map Japanese to 'zh' backend
                parts.append(self._clean_text(seg["text"], sentence, seg["lang"]))
            phonemes = "|_|".join(parts)
        else:
            phonemes = self._clean_text(text, sentence, language)

        if hasattr(self, "_cache"):
            self._cache[cache_key] = phonemes

        return phonemes

    def _clean_text(self, text: str, sentence: str, language: str) -> str:
        # Apply each cleaner in order
        cleaned = text
        for cleaner in self._cleaner_funcs:
            cleaned = cleaner(cleaned, sentence, language, self.text_tokenizers)
        return cleaned

    @property
    def g2p_device(self) -> Optional[Union[str, int]]:
        return self._g2p_device

    def set_g2p_device(self, device: Optional[Union[str, int]]) -> None:
        self._g2p_device = device

    def phoneme2token(self, phonemes):
        # Fast path using dict.get to avoid repeated membership checks
        vocab = self.vocab
        if isinstance(phonemes, list):
            out: List[List[int]] = []
            for phone in phonemes:
                phone = phone.split("\t")[0]
                ids: List[int] = []
                for p in phone.split("|"):
                    tid = vocab.get(p)
                    if tid is not None:
                        ids.append(tid)
                out.append(ids)
            return out
        else:
            phonemes = phonemes.split("\t")[0]
            ids2: List[int] = []
            for p in phonemes.split("|"):
                tid = vocab.get(p)
                if tid is not None:
                    ids2.append(tid)
            return ids2


if __name__ == "__main__":
    punctuations = set(
        [
            ",",
            ".",
            "!",
            "?",
            ";",
            ":",
            "-",
            "—",
            "(",
            ")",
            "[",
            "]",
            "{",
            "}",
            '"',
            "'",
            "“",
            "”",
            "‘",
            "’",
            "…",
            "，",
            "。",
            "！",
            "？",
            "；",
            "：",
            "－",
            "——",
            "（",
            "）",
            "【",
            "】",
            "『",
            "』",
            "「",
            "」",
            "、",
        ]
    )

    def deteck_lang(text: str) -> str:
        # if all characters are chinese (Remove punctuation marks), return 'zh'
        text = "".join([ch for ch in text if ch not in punctuations])
        print(f"Processed text for language detection: {text}")
        for ch in text:
            if not ("\u4e00" <= ch <= "\u9fff"):
                return "en"
        return "zh"

    text_tokenizer = PhonemeBpeTokenizer(g2p_device="cuda:1")
    # test_text = "Diffusion Models are probabilistic models that create realistic samples by simulating the diffusion process, gradually adding and removing noise from data."
    # test_text = "I wanna see that bubble yum bum badumbumbadum"
    # phonemes, tokens = text_tokenizer.tokenize(test_text, "", "en")
    # print(phonemes.replace("|", " "))
    # print(tokens)

    with open("data/TTSEvalSamples/testset.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]

    testset_ipa = []
    for line in lines:
        ref_audio, ref_text, gen_text = line.strip().split("|")

        _lang = deteck_lang(ref_text)
        ref_text_ipa, _ = text_tokenizer.tokenize(ref_text, "", _lang)
        ref_text_ipa = ref_text_ipa.replace("|", " ")

        _lang = deteck_lang(gen_text)
        gen_text_ipa, _ = text_tokenizer.tokenize(gen_text, "", _lang)
        gen_text_ipa = gen_text_ipa.replace("|", " ")

        testset_ipa.append(f"{ref_audio}|{ref_text_ipa}|{gen_text_ipa}\n")

    with open("data/TTSEvalSamples/testset_ipa.txt", "w", encoding="utf-8") as f:
        f.write("audio|ref_text_ipa|gen_text_ipa\n")
        f.writelines(testset_ipa)
