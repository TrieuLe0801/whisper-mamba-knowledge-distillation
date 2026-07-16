# Multilingual Streaming ASR Distillation

This repository contains a Kaggle-first experiment for distilling Whisper large-v3 into a compact, multilingual streaming ASR student for edge devices.

The student is a causal ConMamba encoder with a convolutional log-mel front end. It is trained from scratch with output-level feature distillation and CTC supervision, targeting low memory use and eventual ONNX deployment.

## Repository contents

| Path | Purpose |
| --- | --- |
| `knowledge-distillation-experiment.ipynb` | End-to-end Kaggle notebook: setup, pseudo-labeling, cache generation, training, evaluation, and diagnostics. |
| `README.md` | Project overview and usage notes. |

Training artifacts, cached features, datasets, checkpoints, logs, W&B runs, and local AI-agent configuration are intentionally excluded from version control.

## Model design

- **Teacher:** Whisper large-v3 encoder.
- **Student:** causal ConMamba encoder (`18` layers, `d_model=256`) with a 4×-subsampling CNN log-mel front end.
- **Objectives:** masked cosine feature distillation (teacher features interpolated to the student frame rate) and CTC over a 5,000-piece SentencePiece vocabulary.
- **Streaming focus:** the ConMamba backbone is unidirectional. The encoder is deliberately evaluated in fp32 because selective scan can overflow in fp16 on long sequences.

## Running on Kaggle

The notebook is designed for a Kaggle GPU session, tested with 2× Tesla T4 GPUs. Run its cells in order:

1. Set up the GPU environment and dependencies.
2. Configure data/cache paths in the `Config` cell.
3. Generate or supply the pseudo-label, teacher-feature, and companion mel caches.
4. Run the smoke test.
5. Start resumable KD and CTC training.
6. Run evaluation and, if needed, the CTC-collapse diagnostic cell.

The mel cache must align one-to-one with the teacher-feature cache before training can begin. Kaggle input mounts are read-only; write intermediate outputs and checkpoints under `/kaggle/working`.

## Notes

- The notebook installs the required Kaggle-side dependencies, including `mamba-ssm`, `causal-conv1d`, and `speechbrain`.
- Effective batch size is maintained through gradient accumulation to fit the T4 memory budget.
- Checkpoints are periodically saved to make training resumable across Kaggle sessions.
- No dataset, model weight, cache, checkpoint, generated result, credential, or AI coding-agent file belongs in this repository.
