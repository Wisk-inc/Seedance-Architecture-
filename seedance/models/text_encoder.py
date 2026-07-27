"""Text encoding.

``[1.0]``: "The text encoder is a fine-tuned decoder-only LLM." That is a
meaningful architectural choice rather than a detail -- a causal LM's last
hidden states carry instruction-following structure that a bidirectional T5
encoder (what Wan uses) does not, which is part of why Seedance responds to
dense multi-shot captions.

Two implementations are provided:

* :class:`DecoderOnlyLLMTextEncoder` -- any HF causal LM, taking hidden states
  from ``layer_index``. This is the real path.
* :class:`HashingTextEncoder` -- a deterministic, weightless stand-in so the
  pipelines are runnable with zero downloads. It is a *fixture*, not a model:
  it produces stable embeddings for identical strings and nothing more.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..config import TextEncoderConfig

__all__ = ["TextEncoderOutput", "DecoderOnlyLLMTextEncoder", "HashingTextEncoder", "build_text_encoder"]


class TextEncoderOutput:
    """Embeddings plus the padding mask, with a couple of conveniences."""

    __slots__ = ("embeddings", "mask")

    def __init__(self, embeddings: torch.Tensor, mask: torch.Tensor):
        self.embeddings = embeddings  # [B, L, D]
        self.mask = mask  # [B, L] bool

    @property
    def dim(self) -> int:
        return self.embeddings.shape[-1]

    def to(self, *args, **kwargs) -> "TextEncoderOutput":
        return TextEncoderOutput(self.embeddings.to(*args, **kwargs), self.mask.to(self.embeddings.device))

    def pooled(self) -> torch.Tensor:
        m = self.mask.to(self.embeddings.dtype).unsqueeze(-1)
        return (self.embeddings * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)

    def concat(self, other: "TextEncoderOutput") -> "TextEncoderOutput":
        """Concatenate along the batch axis, padding to the longer length."""
        a, b = self.embeddings, other.embeddings
        if a.shape[1] != b.shape[1]:
            target = max(a.shape[1], b.shape[1])
            a = _pad_to(a, target)
            b = _pad_to(b, target)
            am = _pad_mask(self.mask, target)
            bm = _pad_mask(other.mask, target)
        else:
            am, bm = self.mask, other.mask
        return TextEncoderOutput(torch.cat([a, b], 0), torch.cat([am, bm], 0))


def _pad_to(x: torch.Tensor, length: int) -> torch.Tensor:
    if x.shape[1] >= length:
        return x[:, :length]
    pad = x.new_zeros(x.shape[0], length - x.shape[1], x.shape[2])
    return torch.cat([x, pad], dim=1)


def _pad_mask(m: torch.Tensor, length: int) -> torch.Tensor:
    if m.shape[1] >= length:
        return m[:, :length]
    pad = torch.zeros(m.shape[0], length - m.shape[1], dtype=m.dtype, device=m.device)
    return torch.cat([m, pad], dim=1)


class DecoderOnlyLLMTextEncoder(nn.Module):
    """Hidden states of a decoder-only LLM used as conditioning."""

    def __init__(self, cfg: TextEncoderConfig, device: str = "cpu"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

        self.cfg = cfg
        dtype = getattr(torch, cfg.dtype, torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, trust_remote_code=cfg.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=dtype,
            trust_remote_code=cfg.trust_remote_code,
        )
        self.model.eval().requires_grad_(False)
        self.model.to(device)
        self.device = device

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], device: Optional[str] = None) -> TextEncoderOutput:
        device = device or self.device
        batch = self.tokenizer(
            list(prompts),
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        ).to(device)
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        hidden = out.hidden_states[self.cfg.layer_index]
        return TextEncoderOutput(hidden.float(), batch["attention_mask"].bool())

    def to_device(self, device: str) -> "DecoderOnlyLLMTextEncoder":
        self.model.to(device)
        self.device = device
        return self


class HashingTextEncoder(nn.Module):
    """Deterministic weightless text embedding. A test fixture, not a model.

    Words are hashed into a fixed vocabulary of random-but-seeded vectors, so
    the same prompt always yields the same embedding and different prompts
    differ. Useful for shape/plumbing tests and for exercising the sampling
    loop without a multi-gigabyte download; it carries no semantics.
    """

    def __init__(self, dim: int = 4096, max_length: int = 512, vocab: int = 65536, seed: int = 0):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        generator = torch.Generator().manual_seed(seed)
        table = torch.randn(vocab, dim, generator=generator) / dim**0.5
        self.register_buffer("table", table, persistent=False)
        self.vocab = vocab

    @property
    def hidden_size(self) -> int:
        return self.dim

    def _ids(self, prompt: str) -> List[int]:
        tokens = prompt.replace(",", " , ").replace(".", " . ").split()
        ids = []
        for token in tokens[: self.max_length]:
            digest = hashlib.blake2b(token.lower().encode("utf-8"), digest_size=8).digest()
            ids.append(int.from_bytes(digest, "little") % self.vocab)
        return ids or [0]

    @torch.no_grad()
    def forward(self, prompts: Sequence[str], device: Optional[str] = None) -> TextEncoderOutput:
        device = device or "cpu"
        batch = [self._ids(p) for p in prompts]
        length = min(self.max_length, max(len(b) for b in batch))
        embeddings = torch.zeros(len(batch), length, self.dim)
        mask = torch.zeros(len(batch), length, dtype=torch.bool)
        for i, ids in enumerate(batch):
            ids = ids[:length]
            embeddings[i, : len(ids)] = self.table[torch.tensor(ids)]
            mask[i, : len(ids)] = True
        return TextEncoderOutput(embeddings.to(device), mask.to(device))


def build_text_encoder(
    cfg: TextEncoderConfig, device: str = "cpu", dim: int = 4096, allow_fallback: bool = True
) -> Tuple[nn.Module, str]:
    """Load the configured LLM, falling back to the hashing fixture.

    Returns ``(encoder, kind)`` where ``kind`` is ``"llm"`` or ``"hashing"`` so
    callers can report honestly which one is in use.
    """
    if cfg.model_id:
        try:
            return DecoderOnlyLLMTextEncoder(cfg, device=device), "llm"
        except Exception as exc:  # network, missing transformers, OOM ...
            if not allow_fallback:
                raise
            import logging

            logging.warning(
                "falling back to HashingTextEncoder (no semantics): %s", exc
            )
    return HashingTextEncoder(dim=dim, max_length=cfg.max_length).to(device), "hashing"
