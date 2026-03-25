#!/usr/bin/env python3
"""
Run DCoT-style generation on a test set and save per-example predictions.

Output JSONL fields:
- id
- question
- response_rationale
- response_ans
- gold
- correct
- raw_output
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

MULTIPLE_CHOICE_DATASETS = {"AQUA", "AI2ARC", "GPQA"}


def read_records(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    rows: List[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def resolve_id(row: dict, fallback_index: int) -> str:
    for key in ("id", "index", "qid"):
        if key in row and row[key] is not None:
            return str(row[key])
    return str(fallback_index)


def format_options(raw_options: Any) -> Optional[str]:
    if raw_options is None:
        return None
    if isinstance(raw_options, str):
        txt = raw_options.strip()
        return txt if txt else None
    if isinstance(raw_options, list):
        out = []
        for i, item in enumerate(raw_options):
            letter = chr(ord("A") + i)
            out.append(f"{letter}) {item}")
        return " ".join(out) if out else None
    if isinstance(raw_options, dict):
        out = [f"{k}) {v}" for k, v in raw_options.items()]
        return " ".join(out) if out else None
    return str(raw_options)


def build_prompt(row: dict, num_cots: int) -> str:
    question = row.get("question", "")
    context = row.get("context")
    options = format_options(row.get("options"))
    k = max(1, int(num_cots))

    parts = []
    if question:
        parts.append(f"[Question] {question}")
    if context:
        parts.append(f"[Context] {context}")
    if options:
        parts.append(f"[Options] {options}")
    parts.append(f"[Number of answers] {k}")
    parts.append("[Answer 1] ")
    return "\n".join(parts)


def normalize_answer(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value

    text = str(value).strip()
    if not text:
        return ""
    if "####" in text:
        text = text.split("####", 1)[1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        m = re.search(r"(?i)\bfinal\s*answer\s*[:\-]\s*(.+)$", last)
        if m:
            text = m.group(1).strip()
        else:
            m = re.search(r"(?i)\banswer\s*[:\-]\s*(.+)$", last)
            if m:
                text = m.group(1).strip()

    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"yes", "no"}:
        return lowered

    m = re.match(r"^\s*([A-Z])(?:\)|\.|:|-|\s|$)", text)
    if m is not None:
        return m.group(1)

    numeric = re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if numeric is not None:
        n = text.replace(",", "")
        if "." in n:
            try:
                return float(n)
            except ValueError:
                return n
        try:
            return int(n)
        except ValueError:
            return n
    return text


def normalize_for_compare(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip().lower()


def extract_numeric_candidates(value: Any) -> List[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, int):
        try:
            return [float(value)]
        except OverflowError:
            return []
    if isinstance(value, float):
        return [value]
    text = str(value).strip()
    if not text:
        return []
    text = text.replace(",", "")
    matches = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", text)
    out: List[float] = []
    for token in matches:
        try:
            out.append(float(token))
        except (ValueError, OverflowError):
            continue
    return out


def compare_answers(pred: Any, gold: Any, dataset: Optional[str] = None) -> bool:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    p_norm = normalize_for_compare(p)
    g_norm = normalize_for_compare(g)

    if p_norm == g_norm:
        return True

    dataset_key = str(dataset or "").upper()
    if dataset_key in MULTIPLE_CHOICE_DATASETS or dataset_key == "STRATEGYQA":
        return False

    if isinstance(p, (int, float)) and isinstance(g, (int, float)):
        return abs(float(p) - float(g)) < 1e-9

    p_nums = extract_numeric_candidates(p)
    g_nums = extract_numeric_candidates(g)
    if p_nums and g_nums:
        if abs(float(p_nums[-1]) - float(g_nums[-1])) < 1e-9:
            return True
        g_set = {round(x, 12) for x in g_nums}
        for x in p_nums:
            if round(x, 12) in g_set:
                return True

    return p_norm == g_norm


def get_gold(row: dict) -> Any:
    for key in ("gold", "answer", "label", "gold_ans", "ans", "response_ans"):
        if key in row and row[key] is not None:
            return normalize_answer(row[key])
    return ""


def parse_rationale(raw_text: str) -> Any:
    pairs = re.findall(
        r"(?is)\[answer\s*([0-9]+)\]\s*(.*?)(?=(?:\[\s*answer\s*[0-9]+\s*\]|\[\s*final\s*answer\s*\]|$))",
        raw_text,
    )
    if pairs:
        out: Dict[str, str] = {}
        for idx, content in pairs:
            txt = str(content).strip()
            if txt:
                out[f"Answer{int(idx)}"] = txt
        if out:
            return out

    m = re.search(r"(?is)(.*?)(?:\[\s*final\s*answer\s*\]|$)", raw_text)
    if m:
        txt = m.group(1).strip()
        return txt if txt else raw_text.strip()
    return raw_text.strip()


def parse_final_answer(raw_text: str, dataset: str) -> Any:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    candidate = lines[-1] if lines else ""
    for line in reversed(lines):
        m = re.search(r"(?i)\[final\s*answer\]\s*(.+)$", line)
        if m:
            candidate = m.group(1).strip()
            break
        m = re.search(r"(?i)\bfinal\s*answer\s*[:\-]\s*(.+)$", line)
        if m:
            candidate = m.group(1).strip()
            break

    d = dataset.upper()
    if d in MULTIPLE_CHOICE_DATASETS:
        patterns = (
            r"(?i)\bfinal\s*answer\s*[:\-]?\s*\(?([A-Z])\)?\b",
            r"(?i)\bcorrect\s*answer\s*[:\-]?\s*\(?([A-Z])\)?\b",
            r"(?i)\banswer\s*[:\-]?\s*\(?([A-Z])\)?\b",
            r"(?i)\boption\s*[:\-]?\s*\(?([A-Z])\)?\b",
        )
        for text in (candidate, raw_text):
            for pattern in patterns:
                m = re.search(pattern, text)
                if m:
                    return m.group(1).upper()
        m = re.search(r"\b([A-Z])\b", candidate.upper())
        if m:
            return m.group(1)
    elif d == "STRATEGYQA":
        low = candidate.lower()
        if "true" in low:
            return True
        if "false" in low:
            return False
        if "yes" in low:
            return True
        if "no" in low:
            return False
    else:  # numeric-answer datasets such as GSM8K
        m = re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", candidate)
        if m:
            return normalize_answer(m.group(0))

    return normalize_answer(candidate)


def load_existing_ids(output_file: Path) -> Set[str]:
    if not output_file.exists():
        return set()
    ids: Set[str] = set()
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("id") is not None:
                ids.add(str(obj["id"]))
    return ids


def resolve_dtype_name(dtype_name: str):
    return dtype_name


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["AQUA", "GSM8K", "StrategyQA", "AI2ARC", "GPQA", "MATH"], required=True)
    p.add_argument("--test-file", type=Path, required=True)
    p.add_argument("--output-file", type=Path, required=True)

    p.add_argument("--sft-type", choices=["lora", "full"], required=True)
    p.add_argument("--base-model-path", type=str, default=None)
    p.add_argument("--adapter-path", type=str, default=None)
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--tokenizer-name", type=str, default=None)

    p.add_argument("--num-cots", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--torch-dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")

    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--start-entry", type=int, default=None)
    p.add_argument("--stop-entry", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()

    if not args.test_file.exists():
        raise SystemExit(f"Missing test file: {args.test_file}")
    rows = read_records(args.test_file)
    if not rows:
        raise SystemExit(f"No rows loaded from test file: {args.test_file}")

    if args.sft_type == "lora":
        if not args.base_model_path or not args.adapter_path:
            raise SystemExit("For --sft-type lora, set --base-model-path and --adapter-path.")
    else:
        if not args.model_path:
            raise SystemExit("For --sft-type full, set --model-path.")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output_file.exists():
        args.output_file.unlink()
    existing_ids = load_existing_ids(args.output_file)

    if args.dry_run:
        n = 0
        for i, row in enumerate(rows):
            if args.start_entry is not None and i < args.start_entry:
                continue
            if args.stop_entry is not None and i > args.stop_entry:
                break
            sid = resolve_id(row, i)
            if sid in existing_ids:
                continue
            n += 1
            if args.max_samples is not None and n >= args.max_samples:
                break
        print(f"[DRY-RUN] would_generate={n} output={args.output_file}")
        return

    # Lazy imports so --help/--dry-run can work without full ML deps.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_name = resolve_dtype_name(args.torch_dtype)
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": None,
    }
    dtype = mapping[dtype_name]
    tokenizer_name = args.tokenizer_name

    if args.sft_type == "lora":
        tokenizer_name = tokenizer_name or args.base_model_path
        base_kwargs = {"device_map": "auto", "trust_remote_code": True}
        if dtype is not None:
            base_kwargs["dtype"] = dtype
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, **base_kwargs)
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
    else:
        tokenizer_name = tokenizer_name or args.model_path
        model_kwargs = {"device_map": "auto", "trust_remote_code": True}
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"
    tokenizer.padding_side = "left"

    model.eval()
    n_written = 0
    n_seen = 0

    with args.output_file.open("a", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            if args.start_entry is not None and i < args.start_entry:
                continue
            if args.stop_entry is not None and i > args.stop_entry:
                break

            sid = resolve_id(row, i)
            if sid in existing_ids:
                continue

            prompt = build_prompt(row, num_cots=max(1, args.num_cots))
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": bool(args.do_sample),
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if args.do_sample:
                gen_kwargs["temperature"] = args.temperature
            else:
                # Override sampling-oriented model defaults during greedy decoding
                # to avoid generation-config warnings from some instruct models.
                gen_kwargs["temperature"] = 1.0
                gen_kwargs["top_p"] = 1.0
                gen_kwargs["top_k"] = 50

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            generated = outputs[0][inputs["input_ids"].shape[1]:]
            raw_output = tokenizer.decode(generated, skip_special_tokens=True).strip()

            pred = parse_final_answer(raw_output, args.dataset)
            gold = get_gold(row)
            rationale = parse_rationale(raw_output)
            correct = compare_answers(pred, gold, dataset=args.dataset)

            item = {
                "id": sid,
                "question": row.get("question", ""),
                "response_rationale": rationale,
                "response_ans": pred,
                "gold": gold,
                "correct": bool(correct),
                "raw_output": raw_output,
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()

            n_written += 1
            n_seen += 1
            existing_ids.add(sid)

            if args.log_every > 0 and n_written % args.log_every == 0:
                print(f"[INFO] written={n_written} output={args.output_file}")

            if args.max_samples is not None and n_written >= args.max_samples:
                break

    print(f"[DONE] output={args.output_file} written={n_written}")


if __name__ == "__main__":
    main()
