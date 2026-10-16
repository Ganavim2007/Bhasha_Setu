"""Generate separate, unvalidated Santali-to-Hindi back-translation candidates.

Input lines must be short Santali text that a fluent speaker has checked. This
script uses the NLLB base model in the reverse direction (Santali to Hindi); it
does not add output to the app's approved corpus or retrain the adapter.
"""
import argparse
import json
from pathlib import Path

BASE_MODEL = "facebook/nllb-200-distilled-600M"
SOURCE_LANGUAGE = "sat_Beng"
TARGET_LANGUAGE = "hin_Deva"


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="UTF-8 text file, one reviewed Santali sentence per line")
    parser.add_argument("--output", type=Path, required=True, help="new JSONL file for synthetic candidates")
    parser.add_argument("--batch-size", type=positive_int, default=1, help="use 1 on CPU; increase only if memory allows")
    parser.add_argument("--max-source-tokens", type=positive_int, default=160)
    parser.add_argument("--max-new-tokens", type=positive_int, default=192)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input text file not found: {args.input}")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {args.output} (use --overwrite to replace it)")

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but PyTorch cannot access a CUDA device.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, src_lang=SOURCE_LANGUAGE)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL).to(device).eval()
    src_id = tokenizer.convert_tokens_to_ids(SOURCE_LANGUAGE)
    tgt_id = tokenizer.convert_tokens_to_ids(TARGET_LANGUAGE)
    if tokenizer.unk_token_id in (src_id, tgt_id):
        raise SystemExit("The local NLLB tokenizer does not contain sat_Beng or hin_Deva language tags.")

    lines = []
    seen = set()
    for line in args.input.read_text(encoding="utf-8-sig").splitlines():
        text = " ".join(line.split())
        if text and text not in seen:
            seen.add(text)
            lines.append(text)
    if not lines:
        raise SystemExit("No non-empty, unique Santali lines were found in the input.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    eos = tokenizer.eos_token_id
    skipped_long = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for start in range(0, len(lines), args.batch_size):
            batch = lines[start:start + args.batch_size]
            sequences = []
            kept_batch = []
            for text in batch:
                body = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
                if len(body) > args.max_source_tokens:
                    skipped_long += 1
                    continue
                kept_batch.append(text)
                sequences.append([src_id, *body, eos])
            if not sequences:
                continue
            encoded = tokenizer.pad({"input_ids": sequences}, padding=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    forced_bos_token_id=tgt_id,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=1,
                )
            hindi_lines = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for santali, hindi in zip(kept_batch, hindi_lines):
                hindi = " ".join(hindi.split())
                if not hindi:
                    continue
                record = {
                    "hindi": hindi,
                    "target": santali,
                    "language": "sat",
                    "synthetic": True,
                    "human_validated": False,
                    "source": "nllb-200-distilled-600m-backtranslation",
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote reverse-translation candidates to {args.output} on {device}; skipped {skipped_long} overlong line(s).")
    print("These synthetic pairs are not approved lesson content. Have a fluent Hindi and Santali reviewer check every pair before training.")


if __name__ == "__main__":
    main()
