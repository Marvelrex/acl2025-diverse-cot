#!/usr/bin/env python3
"""
Normalize DCot/CoT datasets to the schema expected by:
- SFT/distill_rationale.py
- SFT/distill_rationale_full_sft.py

Expected output schema (per row):
{
  "index": "...",
  "question": "...",
  "response_rationale": <string|dict>,
  "response_ans": <number|bool|string|null>,
  "options": "..."               # optional
  "prompt_strategy": "..."       # optional
}
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ANSWER_LINE_RE = re.compile(r"(?i)\banswer\s*[:\-]\s*(.+)$")
BOOL_RE = re.compile(r"^(true|false)$", re.IGNORECASE)
INT_RE = re.compile(r"^[+-]?\d+$")
FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d+|\d+)$")
CHOICE_RE = re.compile(r"^\s*([A-Z])(?:\)|\.|:|-|\s|$)")


def read_records(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8")

    # Try full JSON first (array or single object), then fallback to JSONL.
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


def detect_input_format(rows: List[dict]) -> str:
    if not rows:
        return "unknown"
    sample = rows[0]
    if "correct_cots" in sample:
        return "dcot_raw"
    if (
        "response_rationale" in sample
        or "response_ans" in sample
        or "prompt_strategy" in sample
    ):
        return "method_results"
    return "unknown"


def extract_cot_text(cot_obj: Any) -> str:
    if isinstance(cot_obj, str):
        return cot_obj.strip()
    if isinstance(cot_obj, dict):
        for key in ("cot", "response_rationale", "rationale", "text"):
            val = cot_obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def normalize_answer(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value

    text = str(value).strip()
    if not text:
        return None

    if "####" in text:
        text = text.split("####", 1)[1].strip()

    # Prefer an explicit "Answer: ..." line if present.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        match = ANSWER_LINE_RE.search(last)
        if match:
            text = match.group(1).strip()

    text = text.strip().strip("`").strip()
    text = text.strip("<>").strip()
    text = text.rstrip(".").strip()

    if not text:
        return None

    bool_match = BOOL_RE.match(text)
    if bool_match:
        return bool_match.group(1).lower() == "true"

    if INT_RE.match(text):
        try:
            return int(text)
        except ValueError:
            pass

    if FLOAT_RE.match(text):
        try:
            return float(text)
        except ValueError:
            pass

    # Multi-choice labels (A, B, C, ...), e.g., "B) No".
    choice_match = CHOICE_RE.match(text)
    if choice_match and len(choice_match.group(1)) == 1:
        return choice_match.group(1)

    return text


def coerce_rationale(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return value
    return str(value)


def build_base_row(row: dict, fallback_index: str) -> dict:
    index_value = row.get("index", row.get("id", fallback_index))
    out = {
        "index": str(index_value),
        "question": str(row.get("question", "")).strip(),
    }
    if "options" in row and row.get("options") is not None:
        out["options"] = row.get("options")
    return out


def convert_method_results(rows: List[dict]) -> List[dict]:
    converted: List[dict] = []
    for i, row in enumerate(rows):
        out = build_base_row(row, fallback_index=f"row_{i:06d}")
        rationale = row.get("response_rationale", row.get("rationale"))
        answer = row.get("response_ans", row.get("ans"))
        if answer is None:
            for key in ("gold", "gold_ans", "answer", "answer_from_dataset"):
                if row.get(key) is not None:
                    answer = row.get(key)
                    break
        out["response_rationale"] = coerce_rationale(rationale)
        out["response_ans"] = normalize_answer(answer)
        if row.get("prompt_strategy") is not None:
            out["prompt_strategy"] = str(row.get("prompt_strategy"))
        converted.append(out)
    return converted


def _truncate_cots(cots: List[Any], max_cots: Optional[int]) -> List[Any]:
    if max_cots is None or max_cots <= 0:
        return cots
    return cots[:max_cots]


def _row_answer_or_fallback(row: dict, fallback: Any = None) -> Any:
    answer = row.get("answer", row.get("response_ans", row.get("ans", fallback)))
    return normalize_answer(answer)


def convert_dcot_raw(
    rows: List[dict],
    method: str,
    cot_pick: str,
    expand_cots: bool,
    max_cots: Optional[int],
    seed: int,
) -> List[dict]:
    random.seed(seed)
    converted: List[dict] = []

    for i, row in enumerate(rows):
        base = build_base_row(row, fallback_index=f"row_{i:06d}")
        cots = row.get("correct_cots") or []
        if not isinstance(cots, list):
            cots = []
        cots = _truncate_cots(cots, max_cots=max_cots)

        if method == "cot":
            cot_items = cots
            if not expand_cots:
                if not cot_items:
                    cot_items = []
                elif cot_pick == "random":
                    cot_items = [random.choice(cot_items)]
                else:
                    cot_items = [cot_items[0]]

            for j, cot in enumerate(cot_items):
                cot_text = extract_cot_text(cot)
                if not cot_text:
                    continue
                out = dict(base)
                if expand_cots:
                    out["index"] = f"{base['index']}_cot_{j:02d}"
                out["response_rationale"] = cot_text
                cot_answer = cot.get("answer") if isinstance(cot, dict) else None
                out["response_ans"] = _row_answer_or_fallback(row, fallback=cot_answer)
                out["prompt_strategy"] = "cot"
                converted.append(out)
            continue

        # method == "dcot": preserve multiple CoTs in one rationale object.
        rationale_map: Dict[str, str] = {}
        first_cot_answer = None
        for j, cot in enumerate(cots):
            cot_text = extract_cot_text(cot)
            if not cot_text:
                continue
            rationale_map[f"Answer{j + 1}"] = cot_text
            if first_cot_answer is None and isinstance(cot, dict):
                first_cot_answer = cot.get("answer")

        if not rationale_map:
            continue

        out = dict(base)
        out["response_rationale"] = rationale_map
        out["response_ans"] = _row_answer_or_fallback(row, fallback=first_cot_answer)
        out["prompt_strategy"] = "dcot"
        converted.append(out)

    return converted


def has_non_empty_rationale(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return len(value) > 0
    if isinstance(value, list):
        return len(value) > 0
    return True


def validate_rows(rows: List[dict]) -> List[str]:
    errors: List[str] = []
    required = ("index", "question", "response_rationale", "response_ans")
    for i, row in enumerate(rows):
        missing = [key for key in required if key not in row]
        if missing:
            errors.append(f"row[{i}] missing keys: {missing}")
            continue
        if not str(row.get("index", "")).strip():
            errors.append(f"row[{i}] has empty index")
        if not str(row.get("question", "")).strip():
            errors.append(f"row[{i}] has empty question")
        if not has_non_empty_rationale(row.get("response_rationale")):
            errors.append(f"row[{i}] has empty response_rationale")
    return errors


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert DCot/CoT data to BRIDGE distillation schema."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSON/JSONL path.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument(
        "--input-format",
        choices=["auto", "method_results", "dcot_raw"],
        default="auto",
        help="Input schema. Auto-detect by default.",
    )
    parser.add_argument(
        "--method",
        choices=["auto", "cot", "dcot"],
        default="auto",
        help="For dcot_raw input: output as single-CoT or multi-CoT rationale rows.",
    )
    parser.add_argument(
        "--cot-pick",
        choices=["first", "random"],
        default="first",
        help="When --method cot and --expand-cots is off, pick first or random correct CoT.",
    )
    parser.add_argument(
        "--expand-cots",
        action="store_true",
        help="When --method cot on dcot_raw, emit one row per correct CoT.",
    )
    parser.add_argument(
        "--max-cots",
        type=int,
        default=None,
        help="Optional cap on correct CoTs per raw row.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep rows with empty rationale/question; default drops them.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate converted rows without writing output.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = read_records(args.input)
    if not rows:
        raise SystemExit(f"No records found in {args.input}")

    input_format = args.input_format
    if input_format == "auto":
        input_format = detect_input_format(rows)
    if input_format == "unknown":
        raise SystemExit(
            "Could not detect input format. Use --input-format method_results|dcot_raw."
        )

    if input_format == "method_results":
        converted = convert_method_results(rows)
        selected_method = "method_results"
    else:
        method = args.method
        if method == "auto":
            method = "dcot"
        converted = convert_dcot_raw(
            rows,
            method=method,
            cot_pick=args.cot_pick,
            expand_cots=args.expand_cots,
            max_cots=args.max_cots,
            seed=args.seed,
        )
        selected_method = method

    before_filter = len(converted)
    if not args.keep_empty:
        converted = [
            row
            for row in converted
            if str(row.get("question", "")).strip()
            and has_non_empty_rationale(row.get("response_rationale"))
        ]
    dropped = before_filter - len(converted)

    errors = validate_rows(converted)
    if errors:
        print("Validation errors found:")
        for err in errors[:20]:
            print(f"- {err}")
        if len(errors) > 20:
            print(f"... and {len(errors) - 20} more")
        raise SystemExit("Converted data is not compatible with distill scripts.")

    print(
        f"[OK] input={args.input} format={input_format} method={selected_method} "
        f"rows={len(converted)} dropped={dropped}"
    )
    if converted:
        sample = converted[0]
        print(
            "[Sample] keys="
            + ", ".join(sorted(sample.keys()))
            + f" | index={sample.get('index')}"
        )

    if args.validate_only:
        return

    write_jsonl(args.output, converted)
    print(f"[Wrote] {args.output}")


if __name__ == "__main__":
    main()
