#!/usr/bin/env python3
"""
Recompute `correct` in teacher-output JSONL files.

Supports numeric-robust matching:
- "1230 square feet" vs 1230 => correct = true
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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

    lowered = text.strip().lower()
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
    if isinstance(value, (int, float)):
        return [float(value)]
    text = str(value).strip()
    if not text:
        return []
    text = text.replace(",", "")
    matches = re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)", text)
    out: List[float] = []
    for token in matches:
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


def compare_answers(pred: Any, gold: Any) -> bool:
    p = normalize_answer(pred)
    g = normalize_answer(gold)

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

    return normalize_for_compare(p) == normalize_for_compare(g)


def resolve_field(row: Dict[str, Any], preferred: str, fallbacks: Iterable[str]) -> Any:
    if preferred in row and row[preferred] is not None:
        return row[preferred]
    for key in fallbacks:
        if key in row and row[key] is not None:
            return row[key]
    return None


def process_file(
    in_path: Path,
    out_path: Path,
    *,
    pred_field: str,
    gold_field: str,
    dry_run: bool,
) -> Dict[str, int]:
    stats = {
        "lines": 0,
        "parsed": 0,
        "updated": 0,
        "unchanged": 0,
        "missing": 0,
        "invalid_json": 0,
        "eval_total": 0,
        "eval_correct": 0,
    }

    temp_path: Optional[Path] = None
    writer = None

    if not dry_run:
        if in_path.resolve() == out_path.resolve():
            temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            writer = temp_path.open("w", encoding="utf-8", newline="\n")
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = out_path.open("w", encoding="utf-8", newline="\n")

    try:
        with in_path.open("r", encoding="utf-8") as reader:
            for line in reader:
                stats["lines"] += 1
                raw = line.rstrip("\n")
                line_out = raw

                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    stats["invalid_json"] += 1
                    if not dry_run and writer is not None:
                        writer.write(raw + "\n")
                    continue

                if not isinstance(obj, dict):
                    if not dry_run and writer is not None:
                        writer.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    continue

                stats["parsed"] += 1
                pred = resolve_field(
                    obj,
                    preferred=pred_field,
                    fallbacks=("response_ans", "pred", "prediction", "answer"),
                )
                gold = resolve_field(
                    obj,
                    preferred=gold_field,
                    fallbacks=("gold", "answer", "label", "gold_ans", "ans"),
                )

                if pred is None or gold is None:
                    stats["missing"] += 1
                else:
                    new_correct = bool(compare_answers(pred, gold))
                    stats["eval_total"] += 1
                    if new_correct:
                        stats["eval_correct"] += 1
                    old_correct = obj.get("correct")
                    obj["correct"] = new_correct
                    if old_correct == new_correct:
                        stats["unchanged"] += 1
                    else:
                        stats["updated"] += 1

                if not dry_run and writer is not None:
                    writer.write(json.dumps(obj, ensure_ascii=False) + "\n")

    finally:
        if writer is not None:
            writer.flush()
            writer.close()

    if not dry_run and temp_path is not None:
        os.replace(temp_path, out_path)

    return stats


def collect_files(inputs: List[str], patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        paths.append(Path(item))
    for pattern in patterns:
        for match in glob.glob(pattern, recursive=True):
            paths.append(Path(match))

    unique: List[Path] = []
    seen = set()
    for p in paths:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(p)
    return unique


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Post-process JSONL and recompute `correct`.")
    p.add_argument("--input", nargs="*", default=[], help="Input JSONL file(s).")
    p.add_argument("--glob", nargs="*", default=[], help="Glob pattern(s) for input JSONL files.")
    p.add_argument("--inplace", action="store_true", help="Rewrite files in-place.")
    p.add_argument("--output", type=Path, default=None, help="Output path (single input only, ignored when --inplace).")
    p.add_argument("--pred-field", type=str, default="response_ans", help="Prediction field name.")
    p.add_argument("--gold-field", type=str, default="gold", help="Gold field name.")
    p.add_argument("--dry-run", action="store_true", help="Do not write output; print stats only.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    files = collect_files(args.input, args.glob)

    if not files:
        raise SystemExit("No input files. Provide --input or --glob.")

    total_eval = 0
    total_correct = 0

    for file_path in files:
        if not file_path.exists():
            print(f"[WARN] skip missing file: {file_path}")
            continue

        if args.inplace:
            out_path = file_path
        else:
            if args.output is not None:
                if len(files) > 1:
                    raise SystemExit("--output can only be used with a single input file.")
                out_path = args.output
            else:
                out_path = file_path.with_suffix(file_path.suffix + ".post.jsonl")

        stats = process_file(
            in_path=file_path,
            out_path=out_path,
            pred_field=args.pred_field,
            gold_field=args.gold_field,
            dry_run=args.dry_run,
        )
        total_eval += stats["eval_total"]
        total_correct += stats["eval_correct"]

        acc = (stats["eval_correct"] / stats["eval_total"]) if stats["eval_total"] > 0 else 0.0

        mode = "dry-run" if args.dry_run else ("inplace" if args.inplace else "write")
        print(
            f"[DONE] mode={mode} input={file_path} output={out_path} "
            f"lines={stats['lines']} parsed={stats['parsed']} updated={stats['updated']} "
            f"unchanged={stats['unchanged']} missing={stats['missing']} invalid_json={stats['invalid_json']} "
            f"accuracy={stats['eval_correct']}/{stats['eval_total']} ({acc:.4f})"
        )

    overall_acc = (total_correct / total_eval) if total_eval > 0 else 0.0
    print(
        f"[SUMMARY] files={len(files)} accuracy={total_correct}/{total_eval} ({overall_acc:.4f})"
    )


if __name__ == "__main__":
    main()
