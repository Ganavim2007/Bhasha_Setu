# Bhasha Setu Santali adapter

## Summary

This directory contains a PEFT LoRA adapter loaded by the Bhasha Setu FastAPI service for Hindi-to-Santali draft translation. It is a classroom prototype: translations need both a teacher-role review and a community language expert-role review before students can see them.

## Verified configuration

- Base model: `facebook/nllb-200-distilled-600M`.
- Source language tag: `hin_Deva` (Hindi in Devanagari).
- Target language tag in the app: `sat_Beng` (Santali in the Bengali-script setting).
- LoRA rank: 8; alpha: 32; dropout: 0.05.
- Target modules: `q_proj` and `v_proj`.
- Inference is on the school server. The PWA caches approved text, not the NLLB model weights.

These fields come from `adapter_config.json` and the inference code in `app.py`. The local NLLB tokenizer recognizes `sat_Beng`; it does not recognize `sat_Olck`. This adapter therefore must not be presented as Ol Chiki output.

## Training and evaluation provenance

The original parallel training corpus, preprocessing steps, and evaluation report are not included in this repository. The adapter's original scores cannot be reproduced or independently checked from the files here. Do not treat the confidence score produced by generation as a probability of correctness.

The current training workflow in `training/train_santali_adapter.py` continues this adapter only from pairs carrying both reviewer roles, keeps the holdout corpus separate, reports chrF for the base model/current adapter/candidate, and writes a candidate directory without replacing this adapter. The original adapter's training data is still unknown, so holdout independence from its earlier training cannot be proved. A metric increase is not a substitute for community review.

Back-translation output is synthetic and starts with `human_validated: false`. It must be checked by fluent reviewers before it can enter a training run. The app does not automatically train or install new weights.

## Intended use and limitations

- Use only for Hindi-to-Santali draft suggestions in the configured Bengali-script setting.
- Ho and Mundari translation models are not included in this adapter.
- The model can mistranslate educational terms, omit context, mix languages, or produce the wrong writing system.
- It is not an offline student-device model, OCR system, explanation model, or speech model.
- Do not publish a draft to students until both reviewer roles have approved it.

Before classroom deployment, collect permissioned and representative parallel data, document the community's preferred Santali writing system, create a genuinely held-out test set, and have fluent speakers assess both quality and educational meaning.
