# Santali feedback and training workflow

This workflow keeps new model weights out of the live app until reviewers examine them. The app's exported corpus contains only lessons reviewed by both the teacher role and the community expert role. The training script separately requires an independently checked holdout file so it can compare the current base model with the candidate adapter on the same held-out Hindi sentences.

## 1. Export reviewed lesson pairs

Sign in to the app, open **Progress**, and download the approved pairs JSONL. Keep a copy of every dated export. Never put the holdout sentences in the training file.

## 2. Optional back-translation candidates

Use short, checked Santali text in the model's current Bengali-script setting, one sentence per line:

```powershell
python -m pip install -r requirements-training.txt
python training/backtranslate_santali.py .\reviewed_santali.txt --output .\training\synthetic_candidates.jsonl
```

The generated Hindi source is synthetic and may be wrong. A fluent Hindi and Santali reviewer must check each pair. Only after that review, mark the row `"human_validated": true` and include both roles in `"reviewer_roles"`. Keep these rows in a separate file. The training command rejects unreviewed synthetic rows.

## 3. Prepare a separate holdout file

Create a JSONL file of at least 25 curriculum sentences that were reviewed by both roles and held aside from this new training run, back-translation, and candidate selection. Each row uses this shape:

```json
{"hindi":"<Hindi source>","target":"<reviewed Santali reference>","language":"sat","reviewer_roles":["community_expert","teacher"]}
```

The evaluation file must not share Hindi source sentences with training data. The script checks exact source overlap, review roles, duplicate/conflicting sources, language, and token limits. This does not prevent topic-level leakage; reviewers should choose representative curriculum examples that were held aside before training. The repository does not include the training corpus used for the current adapter, so we cannot prove these sentences were never seen by that older model; its baseline score is therefore provisional.

## 4. Train and evaluate a candidate adapter

Use a new output directory for each attempt:

```powershell
python training/train_santali_adapter.py `
  --approved .\approved_pairs.jsonl `
  --evaluation .\holdout_pairs.jsonl `
  --output .\training\candidate-santali-2026-10-11
```

To include separately reviewed synthetic rows, run the command with this additional option:

```powershell
python training/train_santali_adapter.py `
  --approved .\approved_pairs.jsonl `
  --evaluation .\holdout_pairs.jsonl `
  --synthetic-reviewed .\synthetic_pairs_reviewed.jsonl `
  --output .\training\candidate-santali-with-reviewed-synthetic
```

The script continues the LoRA adapter in `lora_santali/`, defaults to batch size one for CPU, and saves the candidate plus an `evaluation.json` report without overwriting the live adapter. The report includes data fingerprints, human and synthetic pair counts, chrF scores for the base model, current adapter, and candidate, score delta from the current adapter, and settings. chrF is a corpus-level screening signal; a higher score does not prove better educational or cultural translations. Have both reviewer roles inspect candidate outputs before manually promoting any adapter.

At least 50 unique human-reviewed training pairs and 25 separate holdout pairs are required. Small corpora can overfit, especially with added synthetic data. Ho and Mundari training are not included; each language needs its own verified data, language tags, script decision, adapter, holdout set, and human evaluation. A direct check of the installed NLLB tokenizer found `sat_Beng`, but `hoc_Deva`, `unr_Deva`, and `sat_Olck` resolve to its unknown-token ID. This script does not add language tokens or initialize their embeddings, so do not enable Ho, Mundari, or Ol Chiki by changing only the target tag. First agree on the script with language experts, then design and evaluate that model extension with their reviewed data.
