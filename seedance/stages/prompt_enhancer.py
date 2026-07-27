"""Stage 1: prompt engineering.

``[1.0]``: "A separate prompt-engineering (PE) model, initialized from
Qwen2.5-14B, rewrites user prompts into the dense-caption format used in DiT
training via a two-stage process: SFT on synthesized prompt->dense-caption
pairs (with separate handling of I2V vs T2V prompt styles), then DPO (with
LoRA) to reduce hallucination and improve semantic fidelity."

This is the single highest-leverage stage to copy. Wan is trained on dense
captions too, so feeding it a five-word prompt wastes most of the model. The
rewriter here has three tiers, tried in order:

1. A local Qwen2.5 instruct model (the closest analog to Seedance's PE stage).
2. Wan's own ``prompt_extend`` (DashScope API or a local model), if configured.
3. A deterministic template expander that needs no weights at all.

Tier 3 is not a language model and does not invent content -- it restructures
what you wrote into the caption *shape* the model was trained on (subject,
action, setting, camera, lighting, quality tail), which is most of the benefit.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = ["PromptEnhancer", "DenseCaption", "SYSTEM_PROMPT_T2V", "SYSTEM_PROMPT_I2V"]


SYSTEM_PROMPT_T2V = """You rewrite short video prompts into dense captions for a video generation model.

Rules:
- One paragraph, 60-120 words, present tense, no bullet points, no preamble.
- Describe in this order: main subject and its appearance, the action over time,
  the setting, camera framing and movement, lighting and colour, overall style.
- Keep every concrete detail the user gave. Do not contradict them.
- Add only detail that is strongly implied. Never invent named people, brands or text.
- End with a short quality clause (e.g. "sharp focus, natural motion, cinematic lighting").
Output the caption only."""

SYSTEM_PROMPT_I2V = """You rewrite short prompts into dense captions for an image-to-video model.

The first frame already exists and must not be re-described in contradiction.
Rules:
- One paragraph, 50-100 words, present tense.
- Focus on what CHANGES: the motion of the subject, the camera move, evolving light.
- Refer to what is in the image only to anchor the motion.
- Do not describe a different scene, a cut, or a new subject.
Output the caption only."""


@dataclass
class DenseCaption:
    """A rewritten prompt plus what the rewriter thinks it contains."""

    text: str
    source: str = "template"  # "qwen" | "wan_prompt_extend" | "template" | "passthrough"
    original: str = ""
    subject: str = ""
    camera: str = ""
    warnings: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return self.text


_CAMERA_TERMS = (
    "dolly", "pan", "tilt", "zoom", "tracking", "handheld", "crane", "orbit",
    "close-up", "closeup", "wide shot", "aerial", "drone", "static", "steadicam",
    "push in", "pull out", "low angle", "high angle", "over the shoulder",
)
_LIGHT_TERMS = (
    "golden hour", "backlit", "rim light", "neon", "overcast", "candlelit",
    "moonlight", "sunlight", "studio light", "soft light", "harsh", "silhouette",
)
_STYLE_TERMS = (
    "cinematic", "documentary", "anime", "3d render", "claymation", "film grain",
    "vhs", "35mm", "photorealistic", "watercolor", "oil painting", "pixel art",
)
_QUALITY_TAIL = "sharp focus, stable natural motion, consistent lighting, high detail"


class PromptEnhancer:
    """Rewrites user prompts into dense captions."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 220,
        enabled: bool = True,
        offline_only: bool = False,
        wan_prompt_extend: Optional[str] = None,  # "dashscope" | "local_qwen"
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.enabled = enabled
        self.offline_only = offline_only
        self.wan_prompt_extend = wan_prompt_extend
        self._pipe = None
        self._wan_expander = None
        self._tried_load = False

    # ------------------------------------------------------------------ load
    def _load(self) -> bool:
        if self._pipe is not None:
            return True
        if self._tried_load or self.offline_only:
            return False
        self._tried_load = True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=getattr(torch, self.dtype, torch.bfloat16)
            )
            model.eval().requires_grad_(False).to(self.device)
            self._pipe = (tokenizer, model)
            return True
        except Exception as exc:
            logger.warning("prompt enhancer LLM unavailable (%s); using templates", exc)
            return False

    def _load_wan_expander(self):
        if self._wan_expander is not None or not self.wan_prompt_extend:
            return self._wan_expander
        try:
            from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander

            if self.wan_prompt_extend == "dashscope":
                self._wan_expander = DashScopePromptExpander(is_vl=False)
            else:
                self._wan_expander = QwenPromptExpander(
                    model_name=self.model_id, is_vl=False, device=0
                )
        except Exception as exc:
            logger.warning("wan prompt_extend unavailable: %s", exc)
            self._wan_expander = None
        return self._wan_expander

    # --------------------------------------------------------------- rewrite
    def enhance(
        self,
        prompt: str,
        *,
        task: str = "t2v",
        image=None,
        shot_index: int = 0,
        num_shots: int = 1,
        seed: int = -1,
    ) -> DenseCaption:
        """Rewrite one prompt. Never raises -- worst case it passes through."""
        original = prompt.strip()
        if not self.enabled or not original:
            return DenseCaption(text=original, source="passthrough", original=original)

        expander = self._load_wan_expander()
        if expander is not None:
            try:
                result = expander(original, image=image, seed=seed)
                if getattr(result, "status", False):
                    return DenseCaption(
                        text=result.prompt, source="wan_prompt_extend", original=original
                    )
                logger.warning("wan prompt_extend failed: %s", getattr(result, "message", ""))
            except Exception as exc:
                logger.warning("wan prompt_extend raised: %s", exc)

        if self._load():
            try:
                return self._enhance_llm(original, task=task, shot_index=shot_index, num_shots=num_shots)
            except Exception as exc:
                logger.warning("LLM rewrite failed (%s); using templates", exc)

        return self.template_expand(original, task=task)

    def _enhance_llm(
        self, prompt: str, *, task: str, shot_index: int, num_shots: int
    ) -> DenseCaption:
        import torch

        tokenizer, model = self._pipe  # type: ignore[misc]
        system = SYSTEM_PROMPT_I2V if task in ("i2v", "flf2v") else SYSTEM_PROMPT_T2V
        user = prompt
        if num_shots > 1:
            user = (
                f"This is shot {shot_index + 1} of {num_shots} in one continuous scene. "
                f"Keep the subject's appearance identical to the other shots.\n\n{prompt}"
            )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=True,
                temperature=0.7, top_p=0.9,
            )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        completion = _strip_preamble(completion)
        if len(completion) < len(prompt):
            return self.template_expand(prompt, task=task)
        return DenseCaption(
            text=completion,
            source="qwen",
            original=prompt,
            camera=_find_terms(completion, _CAMERA_TERMS),
        )

    # -------------------------------------------------------------- template
    def template_expand(self, prompt: str, *, task: str = "t2v") -> DenseCaption:
        """Deterministic restructuring into the dense-caption shape.

        Adds no facts. It supplies the *slots* the training captions have --
        camera, lighting, style, quality tail -- only where the user left them
        empty, and normalises punctuation so the text encoder sees clean input.
        """
        text = re.sub(r"\s+", " ", prompt.strip().rstrip(".")).strip()
        warnings: List[str] = []
        camera = _find_terms(text, _CAMERA_TERMS)
        light = _find_terms(text, _LIGHT_TERMS)
        style = _find_terms(text, _STYLE_TERMS)

        parts = [text]
        if not camera:
            parts.append(
                "The camera holds a steady medium shot with a slow push in"
                if task in ("i2v", "flf2v")
                else "The camera holds a steady medium shot"
            )
            warnings.append("no camera direction given; a neutral one was added")
        if not light:
            parts.append("natural directional lighting with soft shadows")
        if not style:
            parts.append("cinematic photorealistic look")
        parts.append(_QUALITY_TAIL)

        caption = ". ".join(p.strip().rstrip(".") for p in parts if p.strip()) + "."
        caption = caption[0].upper() + caption[1:]
        return DenseCaption(
            text=caption,
            source="template",
            original=prompt,
            subject=_first_noun_phrase(text),
            camera=camera,
            warnings=warnings,
        )

    def enhance_storyboard(
        self, prompts: Sequence[str], *, task: str = "t2v", subject_lock: str = ""
    ) -> List[DenseCaption]:
        """Rewrite a multi-shot storyboard, keeping the subject description fixed.

        Repeating the *same* subject clause verbatim in every shot caption is
        the cheapest identity-preservation trick there is: it is what the
        multi-shot training format does, and it costs nothing at inference.
        """
        out: List[DenseCaption] = []
        lock = subject_lock.strip()
        for i, prompt in enumerate(prompts):
            caption = self.enhance(prompt, task=task, shot_index=i, num_shots=len(prompts))
            if lock and lock.lower() not in caption.text.lower():
                caption.text = f"{lock}. {caption.text}"
            if not lock and i == 0:
                lock = caption.subject
            out.append(caption)
        return out


def _find_terms(text: str, terms: Sequence[str]) -> str:
    low = text.lower()
    for term in terms:
        if term in low:
            return term
    return ""


def _first_noun_phrase(text: str) -> str:
    """Very rough subject extraction: the first clause up to a verb-ish word."""
    words = text.split()
    stop = {"is", "are", "was", "were", "walks", "runs", "sits", "stands", "moves", "flies"}
    out: List[str] = []
    for word in words[:12]:
        if word.lower().strip(",.") in stop:
            break
        out.append(word)
    return " ".join(out).strip(" ,.")


def _strip_preamble(text: str) -> str:
    """Remove 'Sure, here is...' style lead-ins some models emit."""
    text = text.strip().strip('"')
    for marker in ("caption:", "Caption:", "here is", "Here is", "Here's", "here's"):
        if text.lower().startswith(marker.lower()):
            idx = text.find(":")
            if 0 <= idx < 80:
                return text[idx + 1 :].strip().strip('"')
    return text
