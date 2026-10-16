"""Continue the Santali LoRA adapter using only independently reviewed pairs.

The output is a candidate adapter in a new directory. It never overwrites the
adapter used by the running app. Keep the evaluation file separate from all
training and synthetic-augmentation data.
"""
import argparse
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

BASE_MODEL = "facebook/nllb-200-distilled-600M"
SOURCE_LANGUAGE = "hin_Deva"
TARGET_LANGUAGE = "sat_Beng"
REQUIRED_ROLES = {"teacher", "community_expert"}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved", type=Path, required=True,
                        help="JSONL exported from the app after both roles approved each pair")
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="separate JSONL of reviewed holdout pairs, never used for training")
    parser.add_argument("--synthetic-reviewed", type=Path,
                        help="optional back-translation JSONL; every row must be human_validated and dual-reviewed")
    parser.add_argument("--adapter", type=Path, default=Path("lora_santali"),
                        help="existing LoRA adapter to continue from")
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory for candidate adapter and evaluation report")
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=1,
                        help="use 1 on CPU; increase only if memory allows")
    parser.add_argument("--gradient-accumulation", type=positive_int, default=4)
    parser.add_argument("--max-source-tokens", type=positive_int, default=120)
    parser.add_argument("--max-target-tokens", type=positive_int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def normalize(text):
    return " ".join(str(text or "").split())


def read_pairs(path, *, require_synthetic_review=False):
    if not path.is_file():
        raise SystemExit(f"Dataset not found: {path}")
    pairs = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"Invalid JSON in {path}, line {line_number}: {error.msg}")
        if not isinstance(row, dict):
            raise SystemExit(f"Expected a JSON object in {path}, line {line_number}.")
        if row.get("language", "sat") != "sat":
            raise SystemExit(f"Unexpected target language in {path}, line {line_number}.")
        roles = set(row.get("reviewer_roles") or [])
        if not REQUIRED_ROLES.issubset(roles):
            raise SystemExit(f"Line {line_number} in {path} lacks teacher and community expert review.")
        if require_synthetic_review and row.get("synthetic") is not True:
            raise SystemExit(f"Row {line_number} in {path} is not marked synthetic=true.")
        if row.get("synthetic"):
            if not require_synthetic_review:
                raise SystemExit(f"Synthetic row {line_number} belongs in --synthetic-reviewed, not the human export.")
            if row.get("human_validated") is not True:
                raise SystemExit(f"Synthetic row {line_number} in {path} is not marked human_validated=true.")
        hindi = normalize(row.get("hindi"))
        target = normalize(row.get("target"))
        if not hindi or not target:
            raise SystemExit(f"Missing Hindi or Santali text in {path}, line {line_number}.")
        pairs.append({"hindi": hindi, "target": target, "synthetic": bool(row.get("synthetic"))})
    return pairs


def deduplicate(pairs, dataset_name):
    by_source = {}
    unique = []
    for pair in pairs:
        source = pair["hindi"]
        old = by_source.get(source)
        if old and old["target"] != pair["target"]:
            raise SystemExit(f"Conflicting reviewed Santali targets in {dataset_name} for the same Hindi source: {source[:80]}")
        if old:
            continue
        by_source[source] = pair
        unique.append(pair)
    return unique


def dataset_fingerprint(pairs):
    data = "\n".join(pair["hindi"] + "\t" + pair["target"] for pair in pairs)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def collate_pairs(pairs, tokenizer, src_id, tgt_id, max_source, max_target, device, torch):
    inputs, targets = [], []
    eos = tokenizer.eos_token_id
    for pair in pairs:
        source = tokenizer(pair["hindi"], add_special_tokens=False, truncation=False)["input_ids"]
        target = tokenizer(pair["target"], add_special_tokens=False, truncation=False)["input_ids"]
        if len(source) + 2 > max_source or len(target) + 2 > max_target:
            raise ValueError("A prevalidated example exceeded the token limits; refusing silent truncation.")
        inputs.append([src_id, *source, eos])
        targets.append([tgt_id, *target, eos])

    input_width = max(map(len, inputs))
    label_width = max(map(len, targets))
    input_ids = torch.full((len(inputs), input_width), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(inputs), input_width), dtype=torch.long)
    labels = torch.full((len(targets), label_width), -100, dtype=torch.long)
    for index, (source, target) in enumerate(zip(inputs, targets)):
        input_ids[index, :len(source)] = torch.tensor(source, dtype=torch.long)
        attention_mask[index, :len(source)] = 1
        labels[index, :len(target)] = torch.tensor(target, dtype=torch.long)
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def evaluate(model, pairs, tokenizer, src_id, tgt_id, max_source, max_new_tokens, device, torch, chrf):
    hypotheses = []
    model.eval()
    for start in range(0, len(pairs), 4):
        batch = pairs[start:start + 4]
        sequences = []
        for pair in batch:
            body = tokenizer(pair["hindi"], add_special_tokens=False, truncation=False)["input_ids"]
            if len(body) + 2 > max_source:
                raise ValueError("A holdout source exceeded --max-source-tokens; refusing truncated evaluation.")
            sequences.append([src_id, *body, tokenizer.eos_token_id])
        encoded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(**encoded, forced_bos_token_id=tgt_id,
                                    max_new_tokens=max_new_tokens, num_beams=1)
        hypotheses.extend(normalize(text) for text in tokenizer.batch_decode(output, skip_special_tokens=True))
    references = [pair["target"] for pair in pairs]
    return chrf.corpus_score(hypotheses, [references]).score


def main():
    args = parse_args()
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise SystemExit("--learning-rate must be positive.")
    if args.output.exists():
        raise SystemExit(f"Output path already exists; choose a new candidate directory: {args.output}")
    if not args.adapter.is_dir() or not (args.adapter / "adapter_config.json").is_file():
        raise SystemExit(f"LoRA adapter not found: {args.adapter}")

    human = deduplicate(read_pairs(args.approved), "approved training data")
    synthetic = deduplicate(read_pairs(args.synthetic_reviewed, require_synthetic_review=True),
                            "reviewed synthetic data") if args.synthetic_reviewed else []
    training = deduplicate(human + synthetic, "combined training data")
    evaluation = deduplicate(read_pairs(args.evaluation), "holdout evaluation data")
    if len(human) < 50:
        raise SystemExit(f"Need at least 50 unique human-reviewed training pairs before augmentation; found {len(human)}.")
    if len(evaluation) < 25:
        raise SystemExit(f"Need at least 25 separate, dual-reviewed holdout pairs; found {len(evaluation)}.")
    train_sources = {pair["hindi"] for pair in training}
    overlap = train_sources.intersection(pair["hindi"] for pair in evaluation)
    if overlap:
        raise SystemExit(f"Training and evaluation sources overlap ({len(overlap)} source sentence(s)); remove the leakage.")

    import torch
    from peft import PeftModel
    from sacrebleu.metrics import CHRF
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but PyTorch cannot access a CUDA device.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, src_lang=SOURCE_LANGUAGE)
    src_id = tokenizer.convert_tokens_to_ids(SOURCE_LANGUAGE)
    tgt_id = tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE)
    if tokenizer.unk_token_id in (src_id, tgt_id):
        raise SystemExit("The local NLLB tokenizer lacks hin_Deva or sat_Beng language tags.")
    for pair in training + evaluation:
        source_length = len(tokenizer(pair["hindi"], add_special_tokens=False, truncation=False)["input_ids"])
        target_length = len(tokenizer(pair["target"], add_special_tokens=False, truncation=False)["input_ids"])
        if source_length + 2 > args.max_source_tokens or target_length + 2 > args.max_target_tokens:
            raise SystemExit("A dataset sentence is over the configured token limit; shorten/resegment and review it first.")

    chrf = CHRF()
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device).eval()
    base_score = evaluate(base_model, evaluation, tokenizer, src_id, tgt_id,
                          args.max_source_tokens, args.max_target_tokens, device, torch, chrf)
    del base_model
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    current_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
    current_model = PeftModel.from_pretrained(current_model, str(args.adapter), is_trainable=False).eval()
    current_score = evaluate(current_model, evaluation, tokenizer, src_id, tgt_id,
                             args.max_source_tokens, args.max_target_tokens, device, torch, chrf)
    del current_model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    train_base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device)
    train_base.config.use_cache = False
    model = PeftModel.from_pretrained(train_base, str(args.adapter), is_trainable=True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise SystemExit("The loaded adapter has no trainable parameters.")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    step_count = 0
    history = []
    for epoch in range(args.epochs):
        shuffled = list(training)
        random.Random(args.seed + epoch).shuffle(shuffled)
        model.train()
        running_loss, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for start in range(0, len(shuffled), args.batch_size):
            batch = shuffled[start:start + args.batch_size]
            input_ids, attention_mask, labels = collate_pairs(
                batch, tokenizer, src_id, tgt_id, args.max_source_tokens,
                args.max_target_tokens, device, torch,
            )
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became non-finite; candidate training was stopped.")
            running_loss += float(loss.detach()) * len(batch)
            seen += len(batch)
            loss.backward()
            pending += 1
            if pending == args.gradient_accumulation or start + args.batch_size >= len(shuffled):
                for parameter in trainable:
                    if parameter.grad is not None:
                        parameter.grad.div_(pending)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_count += 1
                pending = 0
        epoch_loss = running_loss / max(seen, 1)
        history.append({"epoch": epoch + 1, "training_loss": epoch_loss})
        print(f"epoch {epoch + 1}/{args.epochs}: loss={epoch_loss:.4f}")

    tuned_score = evaluate(model, evaluation, tokenizer, src_id, tgt_id,
                           args.max_source_tokens, args.max_target_tokens, device, torch, chrf)
    synthetic_used = sum(1 for pair in training if pair["synthetic"])
    args.output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    report = {
        "base_model": BASE_MODEL,
        "continued_from_adapter": str(args.adapter),
        "original_adapter_training_data_available": False,
        "target_language": "Santali (sat_Beng; Bengali-script setting)",
        "training_pairs_human": len(human),
        "reviewed_synthetic_pairs_provided": len(synthetic),
        "training_pairs_reviewed_synthetic": synthetic_used,
        "evaluation_pairs": len(evaluation),
        "training_sha256": dataset_fingerprint(training),
        "evaluation_sha256": dataset_fingerprint(evaluation),
        "metric": "sacreBLEU CHRF",
        "metric_signature": chrf.get_signature(),
        "base_model_chrf": base_score,
        "current_adapter_chrf": current_score,
        "candidate_chrf": tuned_score,
        "candidate_delta_from_current_adapter": tuned_score - current_score,
        "automatic_deployment": False,
        "evaluation_limit": "The repository does not contain the original adapter's training corpus, so holdout independence from that earlier training cannot be proven.",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "optimizer_steps": step_count,
        "epoch_history": history,
        "created_unix": time.time(),
        "review_note": "CHRF is a corpus-level screening metric, not a human quality judgment. Have both reviewer roles inspect the candidate before replacing the live adapter.",
    }
    (args.output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8")
    print(f"Saved candidate adapter and holdout report to {args.output}.")
    print(f"Base chrF: {base_score:.2f}; current adapter: {current_score:.2f}; candidate: {tuned_score:.2f}; "
          f"candidate delta from current: {tuned_score - current_score:+.2f}.")
    print("This candidate was not installed in the running app; review its translations before promotion.")


if __name__ == "__main__":
    main()
