from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from infer_from_prompts import (
    _augment_system_prompt,
    _chunked,
    _normalize_input_record,
    _read_jsonl,
    _resolve_eos_token_ids,
    _slice_to_first_json_object,
    _load_model,
)


def _device_info(torch, use_cuda: bool) -> dict[str, Any]:
    if use_cuda and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return {
            "device": "cuda",
            "gpu_model": torch.cuda.get_device_name(idx),
            "gpu_total_memory_mb": int(props.total_memory / (1024 * 1024)),
        }
    return {
        "device": "cpu",
        "gpu_model": "CPU-only",
        "gpu_total_memory_mb": None,
    }


def _decode_tokens(tokenizer, token_ids: list[int]) -> str:
    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=True)
    return tokenizer.convert_tokens_to_string(tokens).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure Stream A inference throughput.")
    parser.add_argument("--in", dest="input_jsonl", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--n_examples", type=int, default=128)
    parser.add_argument("--target_examples", type=int, default=82248)
    parser.add_argument("--pred_out", type=str, default=None)
    parser.add_argument("--report_out", type=str, default="speed_report.json")
    parser.add_argument(
        "--strengthen_json_contract",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--stop_at_closing_brace",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    report: dict[str, Any] = {
        "status": "error",
        "input_jsonl": args.input_jsonl,
        "model": args.model,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "n_examples_requested": args.n_examples,
    }

    t_total_start = time.perf_counter()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        input_rows = _read_jsonl(args.input_jsonl)
        rows = [_normalize_input_record(rec) for rec in input_rows]
        if args.n_examples > 0:
            rows = rows[: args.n_examples]

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

        t_load_start = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        eos_token_id, _ = _resolve_eos_token_ids(tokenizer, args.stop_at_closing_brace)

        model = _load_model(AutoModelForCausalLM, args.model, torch_dtype, use_cuda)
        model.eval()
        load_seconds = time.perf_counter() - t_load_start

        do_sample = args.temperature > 0
        if not do_sample and hasattr(model, "generation_config"):
            model.generation_config.temperature = None
            model.generation_config.top_p = None

        generated_tokens = 0
        collect_outputs = args.pred_out is not None
        outputs: list[dict[str, Any]] = []
        gen_seconds = 0.0

        for chunk in _chunked(rows, args.batch_size):
            prompts = []
            for rec in chunk:
                system = rec["system"]
                if args.strengthen_json_contract:
                    system = _augment_system_prompt(system)
                user = rec["user"]
                prompt_text = f"SYSTEM:\n{system}\n\nUSER:\n{user}\n\nASSISTANT:\n"
                prompts.append(prompt_text)

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
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

            t_gen_start = time.perf_counter()
            with torch.no_grad():
                generated = model.generate(**inputs, **gen_kwargs)
            gen_seconds += time.perf_counter() - t_gen_start

            prompt_len = int(inputs["input_ids"].shape[1])
            for i, rec in enumerate(chunk):
                gen_ids = generated[i, prompt_len:]
                generated_tokens += int(gen_ids.numel())
                if collect_outputs:
                    raw_output = _decode_tokens(tokenizer, gen_ids.detach().cpu().tolist())
                    json_view = _slice_to_first_json_object(raw_output)
                    outputs.append(
                        {
                            "id": rec.get("id"),
                            "output": raw_output,
                            "json_view": json_view,
                        }
                    )

        total_seconds = time.perf_counter() - t_total_start
        n = len(rows)
        report.update(
            {
                "status": "ok",
                "n_examples_measured": n,
                "target_examples": args.target_examples,
                "timing": {
                    "model_load_seconds": load_seconds,
                    "generation_seconds": gen_seconds,
                    "total_seconds": total_seconds,
                },
                "throughput": {
                    "prompts_per_sec": (n / gen_seconds) if gen_seconds > 0 else 0.0,
                    "tokens_per_sec": (generated_tokens / gen_seconds) if gen_seconds > 0 else 0.0,
                    "overall_prompts_per_sec": (n / total_seconds) if total_seconds > 0 else 0.0,
                    "generated_tokens_total": generated_tokens,
                    "avg_generated_tokens_per_prompt": (generated_tokens / n) if n > 0 else 0.0,
                },
                "dtype": str(torch_dtype).replace("torch.", ""),
            }
        )
        report.update(_device_info(torch, use_cuda))
        pps = report["throughput"]["prompts_per_sec"]
        est_seconds = (args.target_examples / pps) if pps > 0 else None
        if est_seconds is None:
            decision = "insufficient_data"
            reason = "Unable to estimate full-run time from measured throughput."
        elif report["device"] != "cuda":
            decision = "do_not_scale_now"
            reason = "CPU-only environment; full-scale run should wait for GPU execution."
        elif est_seconds <= 3600:
            decision = "scale_now"
            reason = "Estimated full MCAR20 run is within ~1 hour on current setup."
        else:
            decision = "do_not_scale_now"
            reason = "Estimated full MCAR20 run exceeds ~1 hour on current setup."

        report["scale_decision"] = {
            "decision": decision,
            "reason": reason,
            "estimated_seconds_for_target_examples": est_seconds,
            "estimated_hours_for_target_examples": (est_seconds / 3600.0) if est_seconds is not None else None,
        }

        if collect_outputs:
            pred_out = Path(args.pred_out)
            pred_out.parent.mkdir(parents=True, exist_ok=True)
            with pred_out.open("w", encoding="utf-8") as f:
                for rec in outputs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            report["pred_out"] = str(pred_out)

    except Exception as e:
        report["error"] = f"{type(e).__name__}: {e}"
        report["timing"] = {"total_seconds": time.perf_counter() - t_total_start}

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote speed report: {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
