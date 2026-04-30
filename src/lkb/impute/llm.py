"""LLM client + PromptingImputer (LLM-backed imputer over a Prompt strategy).

HFClient wraps a HuggingFace causal-LM (AutoModelForCausalLM + AutoTokenizer)
and implements the ``LLMClient`` ABC. PromptingImputer glues a ``Prompt``
strategy (e.g., ICLPrompt, KGPrompt) to an ``LLMClient`` to produce
predictions over (language, feature) pairs.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from jinja2 import exceptions as jinja_exceptions

from lkb.interfaces import (
    Imputer,
    KnowledgeBase,
    LLMClient,
    Prediction,
    Prompt,
    PromptPayload,
)


_CONFIDENCE_TO_PROB = {"low": 0.33, "medium": 0.66, "high": 0.90}


class HFClient(LLMClient):
    """HuggingFace causal-LM backend (greedy decode, chat-template aware)."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        *,
        max_new_tokens: int = 64,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        tokenizer=None,
        model=None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.dtype = dtype
        self._tokenizer = tokenizer
        self._model = model

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        kwargs = {}
        if self.dtype is not None:
            import torch

            kwargs["torch_dtype"] = getattr(torch, self.dtype, None) or self.dtype
        if self.device is not None:
            kwargs["device_map"] = self.device
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

    def _generate(self, system: str, user: str) -> str:
        self._ensure_loaded()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except jinja_exceptions.TemplateError as exc:
                # Some chat templates (e.g., Gemma variants) reject an explicit
                # ``system`` role. Fold system guidance into the user turn.
                if "System role not supported" not in str(exc):
                    raise
                merged_user = f"System instructions:\n{system}\n\nUser request:\n{user}"
                prompt = self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": merged_user}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            prompt = f"SYSTEM:\n{system}\nUSER:\n{user}\nASSISTANT:\n"

        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self.device is not None:
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
        gen = outputs[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(gen, skip_special_tokens=True).strip()

    def complete(self, payloads: Sequence[PromptPayload]) -> List[str]:
        return [self._generate(p.system, p.user) for p in payloads]


class PromptingImputer(Imputer):
    """LLM-backed imputer: build payloads with ``prompt``, decode with ``llm``."""

    name = "prompting"

    def __init__(
        self,
        prompt: Prompt,
        llm: LLMClient,
        *,
        fallback_value: str = "0",
        fallback_confidence: str = "low",
    ) -> None:
        self.prompt = prompt
        self.llm = llm
        self.fallback_value = fallback_value
        self.fallback_confidence = fallback_confidence

    def fit(self, kb: KnowledgeBase) -> None:
        """No training: Prompt + LLMClient are configured at construction."""
        return

    def impute(
        self, kb: KnowledgeBase, pairs: Sequence[tuple[str, str]]
    ) -> List[Prediction]:
        payloads = [self.prompt.build(kb, lang, feat) for lang, feat in pairs]
        raws = self.llm.complete(payloads)
        preds: List[Prediction] = []
        for raw in raws:
            pred = self.prompt.parse(raw)
            if pred.value is None:
                pred = Prediction(
                    value=self.fallback_value,
                    confidence=pred.confidence or self.fallback_confidence,
                    rationale=pred.rationale,
                    parsed_ok=False,
                    raw=pred.raw,
                )
            preds.append(pred)
        return preds


def confidence_to_prob(confidence: Optional[str]) -> float:
    """Map ordinal confidence {low, medium, high} to a probability."""
    if confidence is None:
        return _CONFIDENCE_TO_PROB["low"]
    return _CONFIDENCE_TO_PROB.get(confidence.strip().lower(), _CONFIDENCE_TO_PROB["low"])


__all__ = ["HFClient", "PromptingImputer", "confidence_to_prob"]
