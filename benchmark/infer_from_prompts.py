from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

VALID_CONFIDENCE = {"low", "medium", "high"}
DEFAULT_SYSTEM = (
    "You are a linguistics expert specializing in typology.\n"
    "Infer missing typological features using:\n"
    "(1) observed facts about the target language,\n"
    "(2) evidence from phylogenetic and geographic neighbors,\n"
    "and (3) well-established linguistic universals.\n"
    "If evidence conflicts, prefer phylogenetic evidence over geographic evidence.\n"
    "If uncertainty remains, choose the most typologically common value."
)
STRICT_JSON_MARKER = "You MUST output exactly one JSON object"
STRICT_JSON_SYSTEM_BLOCK = (
    "Output policy (STRICT JSON):\n"
    "You MUST output exactly one JSON object with keys in this exact order:\n"
    '{"value":"0|1","confidence":"low|medium|high","rationale":"<max 30 words>"}\n'
    "Rules:\n"
    "- Output ONLY valid JSON.\n"
    "- No markdown.\n"
    "- No explanations outside JSON.\n"
    "- `rationale` must be at most 30 words.\n"
    "Valid examples:\n"
    '{"value":"0","confidence":"high","rationale":"Phylogenetic and geographic neighbors consistently support value 0 for this feature."}\n'
    '{"value":"1","confidence":"medium","rationale":"Evidence is mixed, but related languages slightly favor value 1 overall."}\n'
    '{"value":"0","confidence":"low","rationale":"Evidence is sparse and conflicting; defaulting to the more common value."}'
)


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


def _augment_system_prompt(system_prompt: str) -> str:
    base = str(system_prompt).rstrip()
    if STRICT_JSON_MARKER in base:
        return base
    return f"{base}\n\n{STRICT_JSON_SYSTEM_BLOCK}"


def _is_canonical_example(rec: dict) -> bool:
    return all(k in rec for k in ("example_id", "language_id", "feature_id", "resource_group", "r_L"))


def _canonical_to_prompt_record(rec: dict) -> dict:
    allowed_values = rec.get("allowed_values", ["0", "1"])
    allowed = [str(v) for v in allowed_values]
    feature = str(rec["feature_id"])
    description = str(rec.get("r_L", "")).rstrip()
    if "\nTask:" in description:
        description = description.split("\nTask:", 1)[0].rstrip()

    user = (
        f"{description}\n"
        "Task:\n"
        "Predict the missing value for the following feature:\n"
        f"- Feature: {feature}\n"
        f"- Allowed values: {' | '.join(allowed)}\n"
        "Output format (STRICT):\n"
        "Return exactly one minified JSON object on one line with keys in this exact order:\n"
        '{"value":"<one allowed value>","confidence":"low|medium|high","rationale":"<max 30 words>"}\n'
        "Use double quotes only. No Markdown, no code fences, no extra text."
    )
    return {
        "id": str(rec["example_id"]),
        "resource_group": str(rec["resource_group"]),
        "language": str(rec["language_id"]),
        "feature": feature,
        "allowed_values": allowed,
        "system": rec.get("system", DEFAULT_SYSTEM),
        "user": user,
    }


def _normalize_input_record(rec: dict) -> dict:
    if _is_canonical_example(rec):
        return _canonical_to_prompt_record(rec)
    if "system" in rec and "user" in rec:
        return {
            "id": str(rec.get("id", rec.get("example_id", ""))),
            "resource_group": rec.get("resource_group"),
            "language": rec.get("language", rec.get("language_id")),
            "feature": rec.get("feature", rec.get("feature_id")),
            "allowed_values": [str(v) for v in rec.get("allowed_values", ["0", "1"])],
            "system": rec["system"],
            "user": rec["user"],
        }
    raise ValueError(
        "Input JSONL row must be either prompt-style "
        "(`system` + `user`) or canonical example-style "
        "(`example_id`,`language_id`,`feature_id`,`resource_group`,`r_L`)."
    )


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


def _slice_to_first_json_object(text: str) -> str:
    s = text.strip()
    start = s.find("{")
    if start < 0:
        return s
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return s


def _decode_tokens(tokenizer, token_ids: list[int]) -> str:
    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
    return tokenizer.convert_tokens_to_string(tokens).strip()


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


def _resolve_eos_token_ids(tokenizer, stop_at_closing_brace: bool):
    eos_ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    brace_enabled = False
    if stop_at_closing_brace:
        for stop_seq in ("}\n", "})", "}"):
            stop_ids = tokenizer.encode(stop_seq, add_special_tokens=False)
            if len(stop_ids) == 1:
                eos_ids.append(int(stop_ids[0]))
                brace_enabled = True
                break
    eos_ids = sorted(set(eos_ids))
    if not eos_ids:
        return None, brace_enabled
    if len(eos_ids) == 1:
        return eos_ids[0], brace_enabled
    return eos_ids, brace_enabled


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run inference on prompt JSONL or canonical example JSONL."
    )
    p.add_argument("--in", dest="input_jsonl", type=str, required=True)
    p.add_argument("--out", dest="output_jsonl", type=str, required=True)
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument(
        "--strengthen_json_contract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append strict JSON schema/rules/few-shot examples to the system prompt.",
    )
    p.add_argument(
        "--stop_at_closing_brace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If tokenizer supports single-token `}`, include it as a generation stop token.",
    )
    args = p.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError("This script needs `torch` and `transformers` installed in the runtime.") from e

    input_rows = _read_jsonl(args.input_jsonl)
    rows = [_normalize_input_record(rec) for rec in input_rows]
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
    eos_token_id, brace_stop_enabled = _resolve_eos_token_ids(tokenizer, args.stop_at_closing_brace)
    if args.stop_at_closing_brace and not brace_stop_enabled:
        print("Warning: tokenizer does not expose single-token `}`; falling back to default EOS stop.")

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
            if args.strengthen_json_contract:
                system = _augment_system_prompt(system)
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
            "eos_token_id": eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        with torch.no_grad():
            generated = model.generate(**inputs, **gen_kwargs)

        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        for i, rec in enumerate(chunk):
            gen_ids = generated[i, int(input_lens[i]) :]
            raw_output = _decode_tokens(tokenizer, gen_ids.detach().cpu().tolist())
            json_view = _slice_to_first_json_object(raw_output)
            value, confidence, rationale = _extract_json_fields(json_view)
            normalized_value = _normalize_value(value, json_view, rec.get("allowed_values", []))
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
