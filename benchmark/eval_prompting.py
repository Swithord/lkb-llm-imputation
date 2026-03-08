from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

VALID_CONFIDENCE = {"low", "medium", "high"}


def _load_prompting_module(path: str):
    spec = importlib.util.spec_from_file_location("prompting_impl", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load prompting module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    return v == -1


def _rewrite_output_contract(user_prompt: str) -> str:
    marker = "Output format (STRICT):"
    prefix = user_prompt.split(marker)[0].rstrip()
    return (
        f"{prefix}\n"
        "Output format (STRICT):\n"
        "Return exactly one minified JSON object on one line with keys in this exact order:\n"
        '{"value":"<one allowed value>","confidence":"low|medium|high","rationale":"<max 30 words>"}\n'
        "Use double quotes only. No Markdown, no code fences, no extra text."
    )


def _allowed_values_from_df(df: pd.DataFrame, feature: str) -> List[str]:
    if feature not in df.columns:
        return ["0", "1"]
    vals = [v for v in df[feature].unique().tolist() if not _is_missing(v)]
    if not vals:
        return ["0", "1"]
    out: List[str] = []
    for v in vals:
        if isinstance(v, float) and v.is_integer():
            out.append(str(int(v)))
        else:
            out.append(str(v))
    return sorted(set(out))


def _pick_language_groups(
    typ_df: pd.DataFrame, high_n: int, low_n: int, min_observed_low: int
) -> Dict[str, List[str]]:
    obs_counts = (typ_df != -1).sum(axis=1).sort_values(ascending=False)
    high = obs_counts.head(high_n).index.astype(str).tolist()

    low_pool = obs_counts[obs_counts >= min_observed_low].sort_values(ascending=True)
    low = [gc for gc in low_pool.index.astype(str).tolist() if gc not in set(high)]
    low = low[:low_n]

    if len(low) < low_n:
        fill = [
            gc
            for gc in obs_counts.sort_values(ascending=True).index.astype(str).tolist()
            if gc not in set(high) and gc not in set(low)
        ]
        low.extend(fill[: (low_n - len(low))])

    return {"high": high, "low": low}


def _iter_missing_pairs(typ_df: pd.DataFrame, groups: Dict[str, List[str]]) -> Iterable[Tuple[str, str, str]]:
    for group in ("high", "low"):
        for lang in groups.get(group, []):
            if lang not in typ_df.index:
                continue
            row = typ_df.loc[lang]
            for feat, val in row.items():
                if _is_missing(val):
                    yield group, str(lang), str(feat)


def _extract_json_fields(text: str) -> Tuple[str | None, str | None, str | None]:
    raw = text.strip()
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = None

    if not isinstance(parsed, dict):
        candidates: List[dict] = []
        start = raw.find("{")
        if start >= 0:
            depth = 0
            in_string = False
            escaped = False
            seg_start = -1
            for i in range(start, len(raw)):
                ch = raw[i]
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
                    continue
                if ch == "{":
                    if depth == 0:
                        seg_start = i
                    depth += 1
                elif ch == "}":
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and seg_start >= 0:
                            segment = raw[seg_start : i + 1]
                            try:
                                obj = json.loads(segment)
                                if isinstance(obj, dict):
                                    candidates.append(obj)
                            except Exception:
                                pass
        if candidates:
            parsed = candidates[-1]
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
    if "|" in v or "<" in v or ">" in v:
        return "low"
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
    lower = text.lower()
    if "<max" in lower or "<one" in lower:
        return "Insufficient direct evidence."
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


def _build_topk_map(topk_df: pd.DataFrame) -> Dict[str, List[str]]:
    tmp: Dict[str, List[Tuple[int, str]]] = {}
    for _, row in topk_df.iterrows():
        feat = str(row["feature"])
        other = str(row["other_feature"])
        rank = int(row["rank"]) if "rank" in row and not pd.isna(row["rank"]) else 0
        tmp.setdefault(feat, []).append((rank, other))
    return {feat: [other for _, other in sorted(items, key=lambda x: x[0])] for feat, items in tmp.items()}


def _chunked(seq: Sequence[dict], n: int) -> Iterable[Sequence[dict]]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build prompts for all missing features in evaluation languages and run GPU inference."
    )
    p.add_argument("--prompting_py", type=str, default="code/prompting.py")
    p.add_argument("--typ", type=str, default="output/uriel+_typological.csv")
    p.add_argument("--meta", type=str, default="output/metadata.csv")
    p.add_argument("--topk_csv", type=str, default="out_corr/topk_per_feature.csv")
    p.add_argument("--gen", type=str, default="output/genetic_neighbours.json")
    p.add_argument("--geo", type=str, default="output/geographic_neighbours.json")
    p.add_argument("--groups_json", type=str, default=None, help="Optional groups JSON from build_benchmark.py.")
    p.add_argument("--high_n", type=int, default=50)
    p.add_argument("--low_n", type=int, default=50)
    p.add_argument("--min_observed_low", type=int, default=5)
    p.add_argument("--top_n", type=int, default=10)
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--prompts_out", type=str, default="benchmark/eval_missing_prompts.jsonl")
    p.add_argument("--pred_out", type=str, default="benchmark/eval_missing_predictions.jsonl")
    args = p.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        raise RuntimeError("This script needs `torch` and `transformers` installed in the runtime.") from e

    typ_df = pd.read_csv(args.typ, index_col=0)
    typ_df.index = typ_df.index.astype(str)
    meta_df = pd.read_csv(args.meta, index_col=0)
    meta_df.index = meta_df.index.astype(str)
    topk_df = pd.read_csv(args.topk_csv)
    with open(args.gen, "r", encoding="utf-8") as f:
        genetic = {str(k): v for k, v in json.load(f).items()}
    with open(args.geo, "r", encoding="utf-8") as f:
        geographic = {str(k): v for k, v in json.load(f).items()}
    topk_map = _build_topk_map(topk_df)

    if args.groups_json:
        groups = json.loads(Path(args.groups_json).read_text(encoding="utf-8"))
    else:
        groups = _pick_language_groups(typ_df, args.high_n, args.low_n, args.min_observed_low)

    prompting = _load_prompting_module(args.prompting_py)
    prompting.typ_df = typ_df
    prompting.metadata_df = meta_df
    prompting.genetic_neighbours = genetic
    prompting.geographic_neighbours = geographic
    prompting.top_n_features = args.top_n
    prompting.topk_map = topk_map

    records: List[dict] = []
    for i, (group, lang, feat) in enumerate(_iter_missing_pairs(typ_df, groups)):
        system, user = prompting.construct_prompt(lang, feat)
        records.append(
            {
                "id": f"{group}:{lang}:{feat}:{i}",
                "resource_group": group,
                "language": lang,
                "feature": feat,
                "allowed_values": _allowed_values_from_df(typ_df, feat),
                "system": system,
                "user": _rewrite_output_contract(user),
            }
        )

    Path(args.prompts_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.prompts_out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

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

    outputs: List[dict] = []
    do_sample = args.temperature > 0
    if not do_sample and hasattr(model, "generation_config"):
        model.generation_config.temperature = None
        model.generation_config.top_p = None
    for chunk in _chunked(records, args.batch_size):
        prompts = []
        for rec in chunk:
            messages = [
                {"role": "system", "content": rec["system"]},
                {"role": "user", "content": rec["user"]},
            ]
            if hasattr(tokenizer, "apply_chat_template"):
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = (
                    f"SYSTEM:\n{rec['system']}\n\nUSER:\n{rec['user']}\n\nASSISTANT:\n"
                )
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
                    "id": rec["id"],
                    "resource_group": rec["resource_group"],
                    "language": rec["language"],
                    "feature": rec["feature"],
                    "value": normalized_value,
                    "confidence": normalized_confidence,
                    "rationale": normalized_rationale,
                    "output": raw_output,
                }
            )

    Path(args.pred_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.pred_out, "w", encoding="utf-8") as f:
        for rec in outputs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote prompts:      {args.prompts_out}")
    print(f"Wrote predictions:  {args.pred_out}")
    print(f"Languages high/low: {len(groups.get('high', []))}/{len(groups.get('low', []))}")
    print(f"Missing prompts:    {len(records)}")


if __name__ == "__main__":
    main()
