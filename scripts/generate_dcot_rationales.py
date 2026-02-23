#!/usr/bin/env python3
"""
Build DCoT rationale JSONL files from source datasets.

Features:
- Resume-safe: skips IDs already present in output JSONL.
- Partial run control: --start-id / --stop-id.
- Works with JSON array or JSONL input.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


DEFAULTS = {
    "AQUA": (
        Path("data/aqua/cot_dataset.json"),
        Path("Baseline/data/AQUA/results_dcot.jsonl"),
    ),
    "GSM8K": (
        Path("data/gsm8k/cot_dataset.json"),
        Path("Baseline/data/GSM8K/results_dcot.jsonl"),
    ),
    "STRATEGYQA": (
        Path("data/strategyqa/cot_dataset.json"),
        Path("Baseline/data/StrategyQA/results_dcot.jsonl"),
    ),
}


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


def normalize_answer(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, int, float)):
        return value

    text = str(value).strip()
    if not text:
        return ""
    if "####" in text:
        text = text.split("####", 1)[1].strip()

    # Prefer final "Answer: ..." fragment if present.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        m = re.search(r"(?i)\banswer\s*[:\-]\s*(.+)$", lines[-1])
        if m:
            text = m.group(1).strip()

    # Collapse common MC form, e.g., "B) ...".
    m = re.match(r"^\s*([A-Z])(?:\)|\.|:|-|\s|$)", text)
    if m:
        return m.group(1)
    return text


def extract_cots(row: dict) -> List[str]:
    cots: List[str] = []
    source = row.get("correct_cots", [])
    if isinstance(source, list):
        for item in source:
            if isinstance(item, dict):
                cot = item.get("cot")
            else:
                cot = item
            if cot is None:
                continue
            cot_text = str(cot).strip()
            if cot_text:
                cots.append(cot_text)
    return cots


def resolve_id(row: dict, fallback_index: int, id_field: str) -> str:
    if id_field in row and row[id_field] is not None:
        return str(row[id_field])
    for k in ("id", "index"):
        if k in row and row[k] is not None:
            return str(row[k])
    return str(fallback_index)


def load_existing_ids(output_path: Path, output_id_field: str = "index") -> Set[str]:
    if not output_path.exists():
        return set()
    ids: Set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and output_id_field in obj and obj[output_id_field] is not None:
                ids.add(str(obj[output_id_field]))
    return ids


def make_output_row(src_row: dict, src_id: str) -> dict:
    cots = extract_cots(src_row)
    rationale = {f"Answer{i + 1}": cot for i, cot in enumerate(cots)}
    answer = normalize_answer(src_row.get("answer", src_row.get("label", "")))

    out = {
        "index": src_id,
        "question": src_row.get("question", ""),
        "response_rationale": rationale,
        "response_ans": answer,
        "prompt_strategy": "dcot",
    }
    if "options" in src_row and src_row["options"] is not None:
        out["options"] = src_row["options"]
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate DCoT rationale JSONL with resume support.")
    p.add_argument("--dataset", type=str, choices=["AQUA", "GSM8K", "StrategyQA"], help="Optional preset for input/output.")
    p.add_argument("--input", type=Path, help="Source dataset JSON/JSONL path.")
    p.add_argument("--output", type=Path, help="Output JSONL path.")
    p.add_argument("--id-field", type=str, default="id", help="Preferred ID field in source rows.")
    p.add_argument("--start-id", type=str, default=None, help="Start processing from this source ID (inclusive).")
    p.add_argument("--stop-id", type=str, default=None, help="Stop after reaching this source ID (inclusive).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output instead of resume append mode.")
    p.add_argument("--dry-run", action="store_true", help="Do not write output; only print stats.")
    return p


def resolve_paths(args) -> Tuple[Path, Path]:
    if args.dataset:
        d_in, d_out = DEFAULTS[args.dataset.upper()]
        in_path = args.input if args.input is not None else d_in
        out_path = args.output if args.output is not None else d_out
        return in_path, out_path
    if args.input is None or args.output is None:
        raise ValueError("Provide --dataset or both --input and --output.")
    return args.input, args.output


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path, output_path = resolve_paths(args)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    rows = read_records(input_path)
    if not rows:
        raise SystemExit(f"No rows loaded from: {input_path}")

    if args.overwrite and not args.dry_run and output_path.exists():
        output_path.unlink()

    existing_ids: Set[str] = set()
    if not args.overwrite:
        existing_ids = load_existing_ids(output_path, output_id_field="index")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a"
    handle = None
    if not args.dry_run:
        handle = output_path.open(mode, encoding="utf-8")

    started = args.start_id is None
    n_total = 0
    n_written = 0
    n_skipped_existing = 0
    n_skipped_empty = 0
    reached_stop = False

    try:
        for i, row in enumerate(rows):
            src_id = resolve_id(row, i, args.id_field)
            n_total += 1

            if not started:
                if src_id == args.start_id:
                    started = True
                else:
                    continue

            if src_id in existing_ids:
                n_skipped_existing += 1
                if args.stop_id is not None and src_id == args.stop_id:
                    reached_stop = True
                    break
                continue

            cots = extract_cots(row)
            if not cots:
                n_skipped_empty += 1
                if args.stop_id is not None and src_id == args.stop_id:
                    reached_stop = True
                    break
                continue

            out = make_output_row(row, src_id=src_id)
            if not args.dry_run and handle is not None:
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_written += 1
            existing_ids.add(src_id)

            if args.stop_id is not None and src_id == args.stop_id:
                reached_stop = True
                break
    finally:
        if handle is not None:
            handle.close()

    print(
        f"[DONE] input={input_path} output={output_path} total_seen={n_total} "
        f"written={n_written} skipped_existing={n_skipped_existing} skipped_empty={n_skipped_empty}"
    )
    if args.start_id is not None and not started:
        print(f"[WARN] start-id not found: {args.start_id}")
    if args.stop_id is not None and not reached_stop:
        print(f"[WARN] stop-id not reached: {args.stop_id}")


if __name__ == "__main__":
    main()
