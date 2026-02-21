# DCot -> SFT Compatibility

This repo now includes `src/bridge_distill_compat.py` to make data from this directory compatible with:

- `SFT/distill_rationale.py` (LoRA)
- `SFT/distill_rationale_full_sft.py` (Full SFT)

## 1) Convert method outputs already in `response_rationale/response_ans` format

```powershell
python src/bridge_distill_compat.py `
  --input Baseline\data\GSM8K\results_cot.jsonl `
  --output Baseline\data\GSM8K\results_cot_for_sft.jsonl `
  --input-format method_results
```

## 2) Convert raw DCoT training data (`correct_cots`) to distill schema

Single multi-CoT row per question:

```powershell
python src/bridge_distill_compat.py `
  --input data\dcot_collection\cot9_dataset.json `
  --output Baseline\data\DCot\dcot_for_sft.jsonl `
  --input-format dcot_raw `
  --method dcot `
  --max-cots 4
```

Single-CoT rows (one per correct CoT):

```powershell
python src/bridge_distill_compat.py `
  --input data\dcot_collection\cot9_dataset.json `
  --output Baseline\data\DCot\cot_for_sft.jsonl `
  --input-format dcot_raw `
  --method cot `
  --expand-cots
```

## 3) Train with your own hyperparameters

Use your own LoRA/full-SFT params on the converted file:

```powershell
python SFT\distill_rationale.py --data-file <converted.jsonl> <your LoRA args...>
python SFT\distill_rationale_full_sft.py --data-file <converted.jsonl> <your Full-SFT args...>
```

## Notes

- The converter guarantees required keys exist: `index`, `question`, `response_rationale`, `response_ans`.
- It supports both JSON arrays and JSONL input.
- It normalizes answers from formats like `#### 72`, `Answer:72`, `B) ...`, and `True/False`.
- If your target repo uses `Baselines` (plural), place converted files there accordingly.
