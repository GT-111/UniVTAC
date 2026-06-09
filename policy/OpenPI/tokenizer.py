"""PaliGemma tokenizer for pi05 — PyTorch-only, no JAX."""

import logging
import os
import numpy as np
import sentencepiece


_DEFAULT_TOKENIZER_PATH = "/data/temp_storage/cache/openpi/big_vision/paligemma_tokenizer.model"


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 200, model_path: str | None = None):
        self._max_len = max_len
        path = model_path or os.environ.get("PALIGEMMA_TOKENIZER_PATH", _DEFAULT_TOKENIZER_PATH)
        with open(path, "rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [0] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
        else:
            if tokens_len > self._max_len:
                logging.warning(f"Token length ({tokens_len}) exceeds max_len ({self._max_len}), truncating.")
            tokens = tokens[:self._max_len]
            mask = [True] * self._max_len
        return np.asarray(tokens, dtype=np.int32), np.asarray(mask, dtype=bool)
