"""RewardDance: generative reward modelling.

``arXiv:2509.08826`` (Wu, Gao et al., ByteDance Seed) reformulates the reward
score as the VLM's probability of predicting a "yes" token to a verification
question about the sample. Two properties follow, and both matter here:

* **Scaling works.** Because the reward head is just next-token prediction,
  the reward model can be any VLM -- the paper scales to 26B (InternVL
  variants). On Seedance-1.0 T2V, alignment GSB improvement rises from +28.0%
  with a 1B RM to +49.0% with a 26B RM.
* **Reward hacking is suppressed.** Large RMs keep high reward variance during
  RL rather than collapsing to a single exploitable mode.

This module implements that scoring rule against any HF vision-language model,
plus the "context scaling" the paper describes (task instruction, reference
examples, chain-of-thought). Every piece degrades gracefully: without a VLM
installed you get :class:`~seedance.reward.heuristics.HeuristicRewardModel`
through :func:`build_reward_model`, and the RL loop is unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

__all__ = ["RewardPrompt", "RewardDance", "RewardEnsemble", "build_reward_model", "DIMENSION_PROMPTS"]

logger = logging.getLogger(__name__)


DIMENSION_PROMPTS: Dict[str, str] = {
    # The three dimensions Seedance 1.0 names for video RLHF.
    "motion_naturalness": (
        "Does the motion in this video look natural and physically plausible, with no "
        "jitter, teleporting, or limbs passing through objects?"
    ),
    "structural_coherence": (
        "Do the subjects in this video keep a consistent identity, shape and structure "
        "across all frames, with no melting, morphing or extra limbs?"
    ),
    "visual_fidelity": (
        "Is this video high quality -- sharp, correctly exposed, free of compression "
        "artifacts, banding and blur?"
    ),
    # 1.5 Pro adds audio fidelity as a third reward dimension alongside motion
    # quality and visual aesthetics.
    "audio_fidelity": (
        "Is the audio clean and well synchronised with the visible events in the video?"
    ),
    "prompt_alignment": (
        "Does this video accurately depict everything described in the prompt?"
    ),
}


@dataclass
class RewardPrompt:
    """The context scaling described in the paper, as a structured object."""

    question: str
    task_instruction: str = (
        "You are a strict video quality evaluator. Answer with a single word: yes or no."
    )
    reference_examples: List[str] = field(default_factory=list)
    chain_of_thought: bool = False
    generation_prompt: str = ""

    def render(self) -> str:
        parts = [self.task_instruction]
        if self.generation_prompt:
            parts.append(f'The video was generated from this prompt: "{self.generation_prompt}"')
        for i, example in enumerate(self.reference_examples, 1):
            parts.append(f"Reference {i}: {example}")
        parts.append(self.question)
        if self.chain_of_thought:
            parts.append("Think step by step, then answer yes or no on the final line.")
        else:
            parts.append("Answer yes or no.")
        return "\n".join(parts)


class RewardDance:
    """Reward = P(``yes``) under a vision-language model.

    Args:
        model_id: any HF VLM with a chat template and image support
            (Qwen2.5-VL, InternVL, ...). Bigger is better and the paper's whole
            point; 2B is the smallest that gives a usable signal.
        num_frames: frames sampled uniformly from the clip and passed as images.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        num_frames: int = 8,
        max_new_tokens: int = 8,
        trust_remote_code: bool = True,
    ):
        from transformers import AutoModelForVision2Seq, AutoProcessor  # lazy

        self.model_id = model_id
        self.device = device
        self.num_frames = num_frames
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype, torch.float32),
            trust_remote_code=trust_remote_code,
        )
        self.model.eval().requires_grad_(False).to(device)
        self.name = f"rewarddance:{model_id}"
        self._yes_ids, self._no_ids = self._answer_token_ids()

    # ------------------------------------------------------------------ setup
    def _answer_token_ids(self) -> tuple[List[int], List[int]]:
        tok = getattr(self.processor, "tokenizer", self.processor)
        yes, no = [], []
        for text in ("yes", " yes", "Yes", " Yes", "YES"):
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                yes.append(ids[0])
        for text in ("no", " no", "No", " No", "NO"):
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                no.append(ids[0])
        if not yes or not no:
            raise RuntimeError(
                f"tokenizer for {self.model_id} has no single-token yes/no; "
                "RewardDance needs one to read the answer probability"
            )
        return sorted(set(yes)), sorted(set(no))

    @staticmethod
    def _frames_to_pil(video: torch.Tensor, count: int) -> List:
        from PIL import Image  # lazy; only needed on the VLM path

        if video.dim() == 5:
            video = video[0]
        t = video.shape[1]
        idx = torch.linspace(0, t - 1, min(count, t)).round().long()
        frames = []
        for i in idx.tolist():
            frame = ((video[:, i].float().clamp(-1, 1) + 1) * 127.5).byte()
            frames.append(Image.fromarray(frame.permute(1, 2, 0).cpu().numpy()))
        return frames

    # ------------------------------------------------------------------ score
    @torch.no_grad()
    def score_one(self, video: torch.Tensor, prompt: RewardPrompt) -> float:
        """P(yes) for a single clip, in ``[0, 1]``."""
        images = self._frames_to_pil(video, self.num_frames)
        text = prompt.render()
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": text}],
            }
        ]
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[chat], images=images, return_tensors="pt").to(self.device)

        if prompt.chain_of_thought:
            # Let the model reason, then read the yes/no distribution at the
            # position where the verdict must appear.
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
            )
            logits = generated.scores[-1][0]
        else:
            logits = self.model(**inputs).logits[0, -1]

        yes = torch.logsumexp(logits[self._yes_ids].float(), dim=0)
        no = torch.logsumexp(logits[self._no_ids].float(), dim=0)
        return float(torch.sigmoid(yes - no))

    @torch.no_grad()
    def score(
        self,
        videos: torch.Tensor,
        prompts: Optional[Sequence[str]] = None,
        *,
        dimension: str = "prompt_alignment",
        chain_of_thought: bool = False,
        reference_examples: Optional[Sequence[str]] = None,
    ) -> torch.Tensor:
        """Reward per sample for a batch, ``[B]``."""
        if videos.dim() == 4:
            videos = videos.unsqueeze(0)
        question = DIMENSION_PROMPTS.get(dimension, dimension)
        out = []
        for i in range(videos.shape[0]):
            rp = RewardPrompt(
                question=question,
                generation_prompt=(prompts[i] if prompts and i < len(prompts) else ""),
                reference_examples=list(reference_examples or []),
                chain_of_thought=chain_of_thought,
            )
            out.append(self.score_one(videos[i], rp))
        return torch.tensor(out, device=videos.device)

    def score_dimensions(
        self, videos: torch.Tensor, prompts: Optional[Sequence[str]] = None
    ) -> Dict[str, torch.Tensor]:
        return {
            dim: self.score(videos, prompts, dimension=dim)
            for dim in ("motion_naturalness", "structural_coherence", "visual_fidelity")
        }


class RewardEnsemble:
    """Multi-dimensional reward, combined the way DanceGRPO expects.

    ``arXiv:2505.07818`` sums *normalised* advantages across up to five reward
    models rather than summing raw scores -- raw summation lets whichever model
    has the widest output range dominate the gradient, which is exactly how
    reward hacking gets a foothold.
    """

    def __init__(self, models: Sequence[object], weights: Optional[Sequence[float]] = None):
        if not models:
            raise ValueError("RewardEnsemble needs at least one reward model")
        if len(models) > 5:
            logger.warning("DanceGRPO validated up to 5 reward models; got %d", len(models))
        self.models = list(models)
        self.weights = list(weights) if weights else [1.0] * len(models)
        if len(self.weights) != len(self.models):
            raise ValueError("weights and models must be the same length")
        self.name = "ensemble(" + ",".join(getattr(m, "name", type(m).__name__) for m in self.models) + ")"

    def raw_scores(
        self, videos: torch.Tensor, prompts: Optional[Sequence[str]] = None
    ) -> List[torch.Tensor]:
        return [m.score(videos, prompts) for m in self.models]  # type: ignore[attr-defined]

    def score(
        self, videos: torch.Tensor, prompts: Optional[Sequence[str]] = None
    ) -> torch.Tensor:
        """Weighted mean of raw scores. Use :meth:`normalized_advantage` for RL."""
        scores = self.raw_scores(videos, prompts)
        total = sum(w * s for w, s in zip(self.weights, scores))
        return total / (sum(self.weights) or 1.0)

    def normalized_advantage(
        self,
        videos: torch.Tensor,
        prompts: Optional[Sequence[str]] = None,
        *,
        group_size: Optional[int] = None,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """Per-model group normalisation, then a weighted sum.

        ``group_size`` splits the batch into GRPO groups (samples that share a
        prompt); statistics are computed inside each group.
        """
        scores = self.raw_scores(videos, prompts)
        b = videos.shape[0] if videos.dim() == 5 else 1
        g = group_size or b
        if b % g:
            raise ValueError(f"batch {b} is not divisible by group size {g}")
        total = torch.zeros(b, device=videos.device)
        for weight, score in zip(self.weights, scores):
            grouped = score.view(-1, g)
            mean = grouped.mean(dim=1, keepdim=True)
            std = grouped.std(dim=1, keepdim=True).clamp(min=eps)
            total = total + weight * ((grouped - mean) / std).reshape(-1)
        return total / (sum(self.weights) or 1.0)


def build_reward_model(
    spec: str = "auto",
    *,
    device: str = "cuda",
    allow_fallback: bool = True,
    **kwargs,
):
    """Build a reward model from a spec string.

    ``"heuristic"`` -> weightless proxies; anything else is treated as a VLM
    repo id for :class:`RewardDance`. ``"auto"`` tries a 7B VLM and falls back.
    Returns an object exposing ``score(videos, prompts) -> [B]``.
    """
    from .heuristics import HeuristicRewardModel

    if spec == "heuristic":
        return HeuristicRewardModel()
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct" if spec == "auto" else spec
    try:
        return RewardDance(model_id, device=device, **kwargs)
    except Exception as exc:
        if not allow_fallback:
            raise
        logger.warning("RewardDance unavailable (%s); using heuristic proxies", exc)
        return HeuristicRewardModel()
