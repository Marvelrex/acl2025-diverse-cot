#!/usr/bin/env python3
"""
Generate teacher rationales with API inference and save resume-safe JSONL.

Output schema per row:
{
  "id": "...",
  "question": "...",
  "response_rationale": {"Answer1": "...", ...} | "...",
  "response_ans": "...",
  "gold": "...",
  "correct": true | false
}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


DATASET_DEFAULTS: Dict[str, Tuple[Path, Path, str]] = {
    "AQUA": (
        Path("Baseline/data/AQUA/train.jsonl"),
        Path("Baseline/data/AQUA/results_dcot_teacher.jsonl"),
        "qid",
    ),
    "GSM8K": (
        Path("Baseline/data/GSM8K/train.jsonl"),
        Path("Baseline/data/GSM8K/results_dcot_teacher.jsonl"),
        "index",
    ),
    "STRATEGYQA": (
        Path("Baseline/data/StrategyQA/train.json"),
        Path("Baseline/data/StrategyQA/results_dcot_teacher.jsonl"),
        "qid",
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


def normalize_question_text(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_question_text_strict(text: str) -> str:
    s = normalize_question_text(text)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_inline_options(question: str) -> Optional[List[str]]:
    q = str(question or "")
    if not q:
        return None

    # Typical multiple-choice format: A) ... B) ... C) ...
    matches = re.findall(
        r"(?is)\b([A-E])\)\s*(.*?)(?=(?:\s+[A-E]\)\s*)|$)",
        q,
    )
    if not matches:
        return None

    options: List[str] = []
    for letter, content in matches:
        txt = re.sub(r"\s+", " ", str(content).strip())
        if txt:
            options.append(f"{letter}) {txt}")
    if len(options) < 2:
        return None
    return options


def make_id_aliases(raw_id: str) -> List[str]:
    aliases: List[str] = []
    if raw_id is None:
        return aliases
    sid = str(raw_id).strip()
    if not sid:
        return aliases
    aliases.append(sid)

    digits = "".join(ch for ch in sid if ch.isdigit())
    if digits:
        aliases.append(digits)
        try:
            aliases.append(str(int(digits)))
        except ValueError:
            pass
    return list(dict.fromkeys(aliases))


def build_options_lookup(
    source_path: Path,
    explicit_id_field: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    rows = read_records(source_path)
    by_id: Dict[str, Any] = {}
    by_question: Dict[str, Any] = {}
    by_question_strict: Dict[str, Any] = {}

    for i, row in enumerate(rows):
        options = row.get("options")
        if options is None:
            continue

        for key in [explicit_id_field, "id", "qid", "index"]:
            if not key:
                continue
            if row.get(key) is None:
                continue
            for alias in make_id_aliases(str(row.get(key))):
                by_id.setdefault(alias, options)

        q = row.get("question")
        if q is not None:
            by_question.setdefault(normalize_question_text(q), options)
            by_question_strict.setdefault(normalize_question_text_strict(q), options)

    return {
        "by_id": by_id,
        "by_question": by_question,
        "by_question_strict": by_question_strict,
    }


def resolve_options_for_row(
    row: dict,
    src_id: str,
    options_lookup: Optional[Dict[str, Dict[str, Any]]],
) -> Optional[Any]:
    # 1) Native options in row.
    if row.get("options") is not None:
        return row.get("options")

    # 2) Parse inline options embedded inside question text.
    parsed = extract_inline_options(str(row.get("question", "")))
    if parsed:
        return parsed

    # 3) Try external lookup by id/question.
    if options_lookup:
        for alias in make_id_aliases(src_id):
            if alias in options_lookup["by_id"]:
                return options_lookup["by_id"][alias]

        q = row.get("question")
        if q is not None:
            key = normalize_question_text(q)
            if key in options_lookup["by_question"]:
                return options_lookup["by_question"][key]
            skey = normalize_question_text_strict(q)
            if skey in options_lookup["by_question_strict"]:
                return options_lookup["by_question_strict"][skey]
    return None


def parse_dotenv(dotenv_path: Path, override_keys: Optional[Set[str]] = None) -> None:
    if not dotenv_path.exists():
        return
    if override_keys is None:
        override_keys = set()
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        if key in os.environ and key not in override_keys:
            continue
        value = value.strip().strip("'").strip('"')
        os.environ[key] = value


def normalize_api_key(raw_key: str) -> str:
    key = str(raw_key or "").strip().strip("'").strip('"')

    # Common accidental paste: "OPENAI_API_KEY=sk-..."
    if key.startswith("OPENAI_API_KEY="):
        key = key.split("=", 1)[1].strip()

    # Accept "Bearer sk-..." pastes.
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


def validate_api_key(api_key: str, api_key_env: str, base_url: Optional[str]) -> None:
    if not api_key:
        raise SystemExit(
            f"Missing API key. Set env var {api_key_env} or place it in .env."
        )

    # Fast-fail for obvious placeholder/value mistakes.
    bad_literals = {
        "OPENAI_API_KEY",
        "YOUR_OPENAI_API_KEY",
        "YOUR_API_KEY",
        "API_KEY",
    }
    if api_key in bad_literals or api_key.startswith("OPENAI_"):
        raise SystemExit(
            f"Invalid API key value detected in {api_key_env}. "
            "It looks like a placeholder or wrong env var text."
        )

    # For official OpenAI, keys normally start with sk-.
    # For custom OpenAI-compatible providers, this may differ.
    if not api_key.startswith("sk-") and not base_url:
        raise SystemExit(
            f"{api_key_env} does not look like an OpenAI key (expected prefix 'sk-'). "
            "If you use an OpenAI-compatible endpoint, pass --base-url."
        )


def resolve_id(row: dict, fallback_index: int, id_field: str) -> str:
    if id_field in row and row[id_field] is not None:
        return str(row[id_field])
    for key in ("id", "index", "qid"):
        if key in row and row[key] is not None:
            return str(row[key])
    return str(fallback_index)


def load_existing_ids(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()
    out: Set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            for key in ("id", "index"):
                if key in obj and obj[key] is not None:
                    out.add(str(obj[key]))
                    break
    return out


def get_gold(row: dict) -> Any:
    for key in (
        "gold",
        "gold_ans",
        "answer",
        "label",
        "response_ans",
        "ans",
    ):
        if key in row and row[key] is not None:
            return normalize_answer(row[key])
    return ""


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

    # Keep this strict enough to avoid pulling tokens like "A)".
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

    # Numeric-robust matching for outputs like "1230 square feet" vs gold=1230.
    p_nums = extract_numeric_candidates(p)
    g_nums = extract_numeric_candidates(g)
    if p_nums and g_nums:
        pg = p_nums[-1]
        gg = g_nums[-1]
        if abs(float(pg) - float(gg)) < 1e-9:
            return True

        # Fallback: any candidate pair exact match.
        g_set = {round(x, 12) for x in g_nums}
        for x in p_nums:
            if round(x, 12) in g_set:
                return True

    return normalize_for_compare(p) == normalize_for_compare(g)


def format_options(options: Any) -> Optional[str]:
    if options is None:
        return None
    if isinstance(options, str):
        txt = options.strip()
        return txt if txt else None
    if isinstance(options, list):
        chunks = []
        for i, item in enumerate(options):
            letter = chr(ord("A") + i)
            if isinstance(item, dict):
                opt_txt = (
                    item.get("text")
                    or item.get("option")
                    or item.get("value")
                    or item.get("content")
                    or str(item)
                )
            else:
                opt_txt = str(item)
            chunks.append(f"{letter}) {opt_txt}")
        return " ".join(chunks) if chunks else None
    if isinstance(options, dict):
        chunks = []
        for key, value in options.items():
            chunks.append(f"{key}) {value}")
        return " ".join(chunks) if chunks else None
    return str(options)


def build_prompt(row: dict, dataset: str, num_chains: int) -> str:
    question = str(row.get("question", "")).strip()
    context = str(row.get("context", "")).strip() if row.get("context") is not None else ""
    options = format_options(row.get("options"))

    answer_constraint = ""
    if dataset.upper() == "AQUA":
        answer_constraint = "Final answer must be one option letter (A/B/C/D/E)."
    elif dataset.upper() == "STRATEGYQA":
        answer_constraint = "Final answer must be true or false."
    else:
        answer_constraint = "Final answer should be concise."

    parts = [
        "You are a teacher model creating training rationales.",
        f"Generate {num_chains} diverse reasoning chains for the question, then provide one final answer.",
        answer_constraint,
        "Return ONLY valid JSON with this schema:",
        '{ "rationale": {"Answer1":"...", "Answer2":"..."}, "final_answer": "..." }',
        "",
        f"Question: {question}",
    ]
    if context:
        parts.append(f"Context: {context}")
    if options:
        # Avoid duplicate "Options: Options: ..." when input already includes label.
        if re.match(r"(?i)^\s*options\s*:\s*", options):
            parts.append(options.strip())
        else:
            parts.append(f"Options: {options}")
    return "\n".join(parts).strip()


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if block_match:
        candidate = block_match.group(1).strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for start in starts:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : end + 1]
                    try:
                        obj = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(obj, dict):
                        return obj
                    break
    return None


def parse_answer_from_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    patterns = [
        r"(?i)\[final\s*answer\]\s*(.+)$",
        r"(?i)\bfinal\s*answer\s*[:\-]\s*(.+)$",
        r"(?i)\banswer\s*[:\-]\s*(.+)$",
    ]
    for line in reversed(lines):
        for pattern in patterns:
            m = re.search(pattern, line)
            if m:
                return m.group(1).strip()
    return lines[-1]


def text_to_answer_dict(text: str) -> Dict[str, str]:
    pairs = re.findall(
        r"(?is)\[answer\s*([0-9]+)\]\s*(.*?)(?=(?:\[\s*answer\s*[0-9]+\s*\]|\[\s*final\s*answer\s*\]|$))",
        text,
    )
    out: Dict[str, str] = {}
    for idx, content in pairs:
        cleaned = str(content).strip()
        if cleaned:
            out[f"Answer{int(idx)}"] = cleaned
    return out


def coerce_rationale(payload: Any, raw_text: str, num_chains: int) -> Any:
    rationale = None
    if isinstance(payload, dict):
        if payload.get("rationale") is not None:
            rationale = payload.get("rationale")
        elif payload.get("response_rationale") is not None:
            rationale = payload.get("response_rationale")

    if rationale is None:
        from_text = text_to_answer_dict(raw_text)
        if from_text:
            rationale = from_text
        else:
            rationale = raw_text.strip()

    if isinstance(rationale, dict):
        if all(re.match(r"(?i)^answer\s*[0-9]+$", str(k).strip()) for k in rationale.keys()):
            items: List[Tuple[int, str]] = []
            for k, v in rationale.items():
                m = re.match(r"(?i)^answer\s*([0-9]+)$", str(k).strip())
                if m is None:
                    continue
                items.append((int(m.group(1)), str(v).strip()))
            items.sort(key=lambda x: x[0])
            return {f"Answer{i}": txt for i, txt in items if txt}
        return {str(k): str(v).strip() for k, v in rationale.items()}

    if isinstance(rationale, list):
        out: Dict[str, str] = {}
        for i, value in enumerate(rationale, start=1):
            txt = str(value).strip()
            if txt:
                out[f"Answer{i}"] = txt
        if out:
            return out
        return ""

    txt = str(rationale).strip()
    if num_chains <= 1:
        return txt
    if not txt:
        return ""
    return {"Answer1": txt}


def coerce_final_answer(payload: Any, raw_text: str) -> Any:
    if isinstance(payload, dict):
        for key in ("final_answer", "answer", "response_ans", "final"):
            if key in payload and payload[key] is not None:
                return normalize_answer(payload[key])
    return normalize_answer(parse_answer_from_text(raw_text))


def extract_message_content(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        chunks: List[str] = []
        for chunk in message_content:
            if isinstance(chunk, str):
                chunks.append(chunk)
            elif isinstance(chunk, dict):
                if chunk.get("type") == "text" and chunk.get("text") is not None:
                    chunks.append(str(chunk["text"]))
                elif chunk.get("content") is not None:
                    chunks.append(str(chunk["content"]))
                elif chunk.get("value") is not None:
                    chunks.append(str(chunk["value"]))
        return "\n".join([x for x in chunks if x]).strip()
    return str(message_content)


def call_openai_chat(
    *,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    retries: int,
    retry_wait: float,
    base_url: Optional[str] = None,
) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'openai'. Install with: pip install openai"
        ) from exc

    client_kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    def build_completion_kwargs(use_max_completion_tokens: bool) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only valid JSON. No markdown fences.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        token_key = "max_completion_tokens" if use_max_completion_tokens else "max_tokens"
        kwargs[token_key] = max_tokens
        return kwargs

    # GPT-5.x chat completions require max_completion_tokens instead of max_tokens.
    use_max_completion_tokens = str(model).lower().startswith("gpt-5")
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                **build_completion_kwargs(use_max_completion_tokens)
            )
            if not getattr(resp, "choices", None):
                return ""
            content = resp.choices[0].message.content
            return extract_message_content(content).strip()
        except Exception as exc:  # pragma: no cover
            msg = str(exc)
            if "max_tokens" in msg and "max_completion_tokens" in msg:
                use_max_completion_tokens = True
                last_err = exc
                continue
            if "max_completion_tokens" in msg and "max_tokens" in msg:
                use_max_completion_tokens = False
                last_err = exc
                continue
            last_err = exc
            if attempt < retries:
                time.sleep(retry_wait * attempt)
            else:
                break
    if last_err is not None:
        raise last_err
    return ""


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, str, str]:
    if args.dataset:
        key = args.dataset.upper()
        in_path, out_path, default_id_field = DATASET_DEFAULTS[key]
        return (
            args.input if args.input is not None else in_path,
            args.output if args.output is not None else out_path,
            args.id_field if args.id_field else default_id_field,
            key,
        )
    if args.input is None or args.output is None:
        raise ValueError("Provide --dataset or both --input and --output.")
    dataset_name = args.dataset_name.upper() if args.dataset_name else "CUSTOM"
    id_field = args.id_field if args.id_field else "id"
    return args.input, args.output, id_field, dataset_name


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate teacher rationale JSONL with resume support.")
    p.add_argument("--dataset", choices=["AQUA", "GSM8K", "StrategyQA"], help="Use built-in input/output presets.")
    p.add_argument("--dataset-name", type=str, default=None, help="Dataset name for prompt constraints when not using --dataset.")
    p.add_argument("--input", type=Path, help="Input file (.json or .jsonl).")
    p.add_argument("--output", type=Path, help="Output JSONL path.")
    p.add_argument("--id-field", type=str, default=None, help="Preferred source ID field.")

    p.add_argument("--model", type=str, default="gpt-4o-mini", help="Teacher model name.")
    p.add_argument("--num-chains", type=int, default=2, help="Number of rationale chains to request.")
    p.add_argument("--temperature", type=float, default=0.2, help="Generation temperature.")
    p.add_argument("--max-tokens", type=int, default=900, help="Max tokens for completion.")
    p.add_argument("--retries", type=int, default=3, help="Retry attempts per sample.")
    p.add_argument("--retry-wait", type=float, default=2.0, help="Base backoff seconds.")
    p.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between successful requests.")
    p.add_argument("--base-url", type=str, default=None, help="Optional OpenAI-compatible base URL.")

    p.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY", help="Environment variable containing API key.")
    p.add_argument("--dotenv", type=Path, default=Path(".env"), help="Optional .env file to preload.")
    p.add_argument("--options-source", type=Path, default=None, help="Optional JSON/JSONL file used to backfill missing options (useful for AQUA).")
    p.add_argument("--options-id-field", type=str, default=None, help="ID field name in --options-source.")
    p.add_argument("--require-options-for-aqua", action=argparse.BooleanOptionalAction, default=True, help="For AQUA, skip samples that still have no options after backfill (default: true).")

    p.add_argument("--start-id", type=str, default=None, help="Start processing at this source ID (inclusive). If numeric and --start-entry is not set, it is treated as entry index.")
    p.add_argument("--stop-id", type=str, default=None, help="Stop processing at this source ID (inclusive). If numeric and --stop-entry is not set, it is treated as entry index.")
    p.add_argument("--start-entry", type=int, default=None, help="Start from entry number (0-based, inclusive).")
    p.add_argument("--stop-entry", type=int, default=None, help="Stop at entry number (0-based, inclusive).")
    p.add_argument("--max-samples", type=int, default=None, help="Optional cap on number of written samples.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output file.")
    p.add_argument("--dry-run", action="store_true", help="No API call, only print what would run.")
    p.add_argument("--log-every", type=int, default=20, help="Progress log interval.")
    p.add_argument("--print-prompt", action=argparse.BooleanOptionalAction, default=True, help="Print prompt text for each sample (default: true).")
    p.add_argument("--print-response", action=argparse.BooleanOptionalAction, default=True, help="Print raw model response for each sample (default: true).")
    p.add_argument("--fsync-every-write", action=argparse.BooleanOptionalAction, default=True, help="Force fsync after each output line (default: true).")
    return p


def resolve_ranges(args: argparse.Namespace) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
    start_id = args.start_id
    stop_id = args.stop_id
    start_entry = args.start_entry
    stop_entry = args.stop_entry

    # Backward-compatible convenience: numeric --start-id/--stop-id means entry index
    # when explicit entry bounds are not set.
    if start_entry is None and start_id is not None and re.fullmatch(r"\d+", start_id):
        start_entry = int(start_id)
        start_id = None
    if stop_entry is None and stop_id is not None and re.fullmatch(r"\d+", stop_id):
        stop_entry = int(stop_id)
        stop_id = None

    if start_entry is not None and start_entry < 0:
        raise ValueError("--start-entry must be >= 0.")
    if stop_entry is not None and stop_entry < 0:
        raise ValueError("--stop-entry must be >= 0.")
    if start_entry is not None and stop_entry is not None and start_entry > stop_entry:
        raise ValueError("--start-entry cannot be greater than --stop-entry.")

    return start_id, stop_id, start_entry, stop_entry


def write_jsonl_line(handle, obj: Dict[str, Any], fsync_every_write: bool) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
    handle.flush()
    if fsync_every_write:
        os.fsync(handle.fileno())


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path, output_path, id_field, dataset_name = resolve_paths(args)
    start_id, stop_id, start_entry, stop_entry = resolve_ranges(args)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    # Prefer .env value for the API key to avoid stale shell exports from other projects.
    parse_dotenv(args.dotenv, override_keys={args.api_key_env})

    api_key = normalize_api_key(os.environ.get(args.api_key_env, ""))
    os.environ[args.api_key_env] = api_key
    if not args.dry_run:
        validate_api_key(api_key=api_key, api_key_env=args.api_key_env, base_url=args.base_url)

    rows = read_records(input_path)
    if not rows:
        raise SystemExit(f"No records loaded from: {input_path}")

    options_lookup: Optional[Dict[str, Dict[str, Any]]] = None
    if args.options_source is not None:
        if not args.options_source.exists():
            raise SystemExit(f"Options source not found: {args.options_source}")
        options_lookup = build_options_lookup(
            source_path=args.options_source,
            explicit_id_field=args.options_id_field,
        )
        print(
            f"[INFO] options_source={args.options_source} "
            f"id_keys={len(options_lookup['by_id'])} "
            f"q_keys={len(options_lookup['by_question'])}"
        )
    elif dataset_name == "AQUA":
        # Best-effort default for AQUA if train rows do not carry options.
        default_candidates = [
            Path("data/aqua/cot_dataset.json"),
            Path("Baseline/data/AQUA/test.jsonl"),
        ]
        for cand in default_candidates:
            if cand.exists():
                options_lookup = build_options_lookup(cand, explicit_id_field=args.options_id_field)
                print(
                    f"[INFO] auto options_source={cand} "
                    f"id_keys={len(options_lookup['by_id'])} "
                    f"q_keys={len(options_lookup['by_question'])}"
                )
                break

    if args.overwrite and output_path.exists() and not args.dry_run:
        output_path.unlink()

    existing_ids = set() if args.overwrite else load_existing_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    handle = None
    if not args.dry_run:
        handle = output_path.open("a", encoding="utf-8")

    started = start_id is None
    reached_stop = False

    seen = 0
    written = 0
    skipped_existing = 0
    skipped_before_start = 0
    skipped_missing_options = 0
    failed = 0

    try:
        for i, row in enumerate(rows):
            src_id = resolve_id(row, i, id_field)
            seen += 1

            if start_entry is not None and i < start_entry:
                skipped_before_start += 1
                continue
            if stop_entry is not None and i > stop_entry:
                reached_stop = True
                break

            if not started:
                if src_id == start_id:
                    started = True
                else:
                    skipped_before_start += 1
                    continue

            if src_id in existing_ids:
                skipped_existing += 1
                if stop_id is not None and src_id == stop_id:
                    reached_stop = True
                    break
                continue

            question = str(row.get("question", "")).strip()
            if not question:
                failed += 1
                continue

            options_value = resolve_options_for_row(
                row=row,
                src_id=src_id,
                options_lookup=options_lookup,
            )
            if options_value is not None and row.get("options") is None:
                row = dict(row)
                row["options"] = options_value

            if dataset_name == "AQUA" and args.require_options_for_aqua and row.get("options") is None:
                skipped_missing_options += 1
                print(
                    f"[WARN] id={src_id} skipped: missing options for AQUA. "
                    "Use --options-source with a file containing options for these questions."
                )
                if stop_id is not None and src_id == stop_id:
                    reached_stop = True
                    break
                continue

            gold = get_gold(row)

            if args.dry_run:
                written += 1
                existing_ids.add(src_id)
                if args.max_samples is not None and written >= args.max_samples:
                    break
                if args.stop_id is not None and src_id == args.stop_id:
                    reached_stop = True
                    break
                continue

            prompt = build_prompt(row=row, dataset=dataset_name, num_chains=args.num_chains)
            if args.print_prompt:
                print(f"\n[PROMPT] id={src_id}\n{prompt}\n")
            try:
                raw = call_openai_chat(
                    api_key=api_key,
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    retries=max(1, args.retries),
                    retry_wait=max(0.1, args.retry_wait),
                    base_url=args.base_url,
                )
            except Exception as exc:
                failed += 1
                print(f"[WARN] id={src_id} failed: {exc}")
                if stop_id is not None and src_id == stop_id:
                    reached_stop = True
                    break
                continue
            if args.print_response:
                print(f"[RESPONSE] id={src_id}\n{raw}\n")

            payload = parse_json_object(raw)
            rationale = coerce_rationale(payload=payload, raw_text=raw, num_chains=args.num_chains)
            pred = coerce_final_answer(payload=payload, raw_text=raw)
            correct = compare_answers(pred, gold)

            out_row = {
                "id": src_id,
                "question": question,
                "response_rationale": rationale,
                "response_ans": pred,
                "gold": gold,
                "correct": bool(correct),
            }
            write_jsonl_line(handle, out_row, fsync_every_write=args.fsync_every_write)

            written += 1
            existing_ids.add(src_id)

            if args.sleep > 0:
                time.sleep(args.sleep)

            if args.log_every > 0 and written % args.log_every == 0:
                print(
                    f"[INFO] written={written} seen={seen} skipped_existing={skipped_existing} "
                    f"skipped_missing_options={skipped_missing_options} failed={failed}"
                )

            if args.max_samples is not None and written >= args.max_samples:
                break
            if stop_id is not None and src_id == stop_id:
                reached_stop = True
                break
    finally:
        if handle is not None:
            handle.close()

    print(
        f"[DONE] input={input_path} output={output_path} seen={seen} written={written} "
        f"skipped_existing={skipped_existing} skipped_before_start={skipped_before_start} "
        f"skipped_missing_options={skipped_missing_options} failed={failed}"
    )
    if start_id is not None and not started:
        print(f"[WARN] start-id not found: {start_id}")
    if stop_id is not None and not reached_stop:
        print(f"[WARN] stop-id not reached: {stop_id}")
    if stop_entry is not None and not reached_stop:
        print(f"[WARN] stop-entry not reached: {stop_entry}")


if __name__ == "__main__":
    main()
