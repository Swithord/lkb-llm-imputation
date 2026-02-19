from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

VALID_CONFIDENCE = {"low", "medium", "high"}


def _read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _chunked(seq: Sequence[dict], n: int) -> Iterable[Sequence[dict]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _extract_json_fields(text: str) -> Tuple[str | None, str | None, str | None]:
    raw = text.strip()
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None
    if not isinstance(parsed, dict):
        return None, None, None
    value = parsed.get("value")
    confidence = parsed.get("confidence")
    rationale = parsed.get("rationale")
    return (
        None if value is None else str(value),
        None if confidence is None else str(confidence).lower(),
        None if rationale is None else str(rationale),
    )


def _normalize_confidence(value: str | None) -> str:
    if not value:
        return "low"
    v = value.strip().lower()
    if v in VALID_CONFIDENCE:
        return v
    if "high" in v:
        return "high"
    if "med" in v:
        return "medium"
    if "low" in v:
        return "low"
    return "low"


def _normalize_rationale(value: str | None) -> str:
    text = "" if value is None else " ".join(str(value).split())
    words = text.split()
    if not words:
        return "Insufficient direct evidence."
    return " ".join(words[:30])


def _normalize_value(parsed: str | None, raw_output: str, allowed_values: Sequence[str]) -> str | None:
    allowed = [str(v) for v in allowed_values]
    if not allowed:
        return parsed

    candidates: List[str] = []
    if parsed is not None:
        candidates.append(str(parsed).strip())
    for pattern in (
        r'"value"\s*:\s*"([^"]+)"',
        r'"value"\s*:\s*([^\s,}]+)',
    ):
        for m in re.finditer(pattern, raw_output, flags=re.IGNORECASE):
            candidates.append(m.group(1).strip().strip('"').strip("'"))

    for cand in candidates:
        if cand in allowed:
            return cand
        if re.fullmatch(r"-?\d+(?:\.0+)?", cand):
            normalized = str(int(float(cand)))
            if normalized in allowed:
                return normalized

    for val in sorted(set(allowed), key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(val)}(?![A-Za-z0-9_])", raw_output):
            return val

    return allowed[0]


def _load_model(AutoModelForCausalLM, model_name: str, dtype, use_cuda: bool):
    model_kwargs = {"dtype": dtype}
    if use_cuda:
        model_kwargs["device_map"] = "auto"
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        model_kwargs.pop("dtype", None)
        model_kwargs["torch_dtype"] = dtype
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)


def main() -> None:
    p = argparse.ArgumentParser(description="Run inference on prompt JSONL produced by benchmark/build_benchmark.py.")
    p.add_argument("--in", dest="input_jsonl", type=str, required=True)
    p.add_argument("--out", dest="output_jsonl", type=str, required=True)
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError("This script needs `torch` and `transformers` installed in the runtime.") from e

    rows = _read_jsonl(args.input_jsonl)
    Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("`--device cuda` requested but CUDA is not available.")
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"

    if args.dtype == "auto":
        if use_cuda and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif use_cuda:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = _load_model(AutoModelForCausalLM, args.model, torch_dtype, use_cuda)
    model.eval()

    do_sample = args.temperature > 0
    if not do_sample and hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    outputs: List[dict] = []

    for chunk in _chunked(rows, args.batch_size):
        prompts: List[str] = []
        for rec in chunk:
            system = rec["system"]
            user = rec["user"]
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = f"SYSTEM:\n{system}\n\nUSER:\n{user}\n\nASSISTANT:\n"
            prompts.append(prompt_text)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if use_cuda:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        with torch.no_grad():
            generated = model.generate(**inputs, **gen_kwargs)

        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        for i, rec in enumerate(chunk):
            gen_ids = generated[i, int(input_lens[i]) :]
            raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            value, confidence, rationale = _extract_json_fields(raw_output)
            normalized_value = _normalize_value(value, raw_output, rec.get("allowed_values", []))
            normalized_confidence = _normalize_confidence(confidence)
            normalized_rationale = _normalize_rationale(rationale)
            outputs.append(
                {
                    "id": rec.get("id"),
                    "resource_group": rec.get("resource_group"),
                    "language": rec.get("language"),
                    "feature": rec.get("feature"),
                    "value": normalized_value,
                    "confidence": normalized_confidence,
                    "rationale": normalized_rationale,
                    "output": raw_output,
                }
            )

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for rec in outputs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote predictions: {args.output_jsonl}")
    print(f"Num prompts: {len(rows)}")


if __name__ == "__main__":
    main()
