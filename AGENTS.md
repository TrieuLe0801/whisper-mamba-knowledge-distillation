# AGENTS.md

## What this project is

Knowledge-distillation experiment to build a **multilingual streaming ASR model for edge devices**. Whisper-large-v3 (teacher) supervises a small **ConMamba** student (Convolution-Augmented Mamba encoder) meant to be fast, low-memory, and eventually ONNX-exportable.

The whole project is a single Kaggle notebook: [knowledge-distillation-experiment.ipynb](knowledge-distillation-experiment.ipynb). No package/build/test — cells run top-to-bottom on Kaggle. `knowledge-distillation-experiment.log` is a prior run's captured output.

## Runtime environment (Kaggle)

Runs on **Kaggle GPU (2× Tesla T4)** under a **~45 h weekly quota**. Each session is capped (`SESSION_HOURS`, default 8 h, checkpoint every 30 min so training resumes across sessions). Verified env:

```
Python 3.12.12 | PyTorch 2.10.0+cu128 | CUDA 12.8 | 2× Tesla T4 (15.6 GB, sm_75)
```

T4 (sm_75) has **no flash-attn-2** (teacher uses `sdpa`) and **16 GB/GPU** — hence `_T4_OVERRIDES` shrinks micro-batch + max audio while holding effective batch 32 via gradient accumulation. Multi-GPU is plain `DataParallel`.

## Key design decisions (and why)

The student was pivoted from `state-spaces/mamba-130m-hf` to a **ConMamba encoder** (xi-j/Mamba-ASR, [arXiv 2407.09732](https://arxiv.org/abs/2407.09732)). Decisions, newest-first:

- **ConMamba via SpeechBrain.** Uses the *real* modules vendored from `.kdedit/conmamba-src/modules/Conmamba.py` (ConvolutionModule + macaron-FFN / Mamba / Conv / FFN layers), so **`speechbrain` is now a runtime dependency** (installed in the env cell). ("MLMA" from the original ask isn't in that repo — ConMamba is the concrete architecture.)
- **Unidirectional / causal, not bidirectional.** Streaming is the goal, and stock `mamba_ssm.Mamba` is causal-only. Going unidirectional let us **delete ~1000 lines of vendored Vim `bimamba`/`selective_scan_interface`** (mamba_ssm has no bidirectional block nor `mamba_inner_fn_no_out_proj`). `Config.cm_bidirectional=False`; a `BiMamba` sentinel raises if flipped back without restoring that code.
- **Single mel path + output-level distillation.** `mel → CNN (4× downsample) → ConMamba → {ctc_head, kl_head}`. `kl_head` regresses the teacher's encoder features at the **encoder output** via cosine loss. The old `StudentASRv2` *injected* teacher states into the backbone — ConMamba can't accept that — and this also fixes a latent train/inference frame-rate mismatch.
- **Companion mel cache.** The teacher cache stores only int8 Whisper states (no audio/mel), so a mel-input student needs a new cache. The mel-cache cell rebuilds the teacher cache's exact shard layout but stores int8 log-mel; `KDDataset` asserts the two caches' per-sample `mel_lens` match. **Must be generated before training.**
- **From-scratch → restaged.** No public ConMamba checkpoint = no backbone to freeze. Stage 1 = **KD-only warmup** (`a_ctc=0`), stage 2 = **KD+CTC**. The `frozen`/`unfrozen` stage *names* are kept only for checkpoint bookkeeping; nothing is actually frozen.

## Architecture (the student: `ConMambaStudent`, cell 26)

- **Front-end:** log-mel `[B,80,T]` → `cnn` (2× stride-2 Conv1d, 4× downsample, ~100→~25 fps) → `in_norm`.
- **Encoder:** `ConmambaEncoder` (cell 25): `cm_layers=18`, `cm_d_model=256`, `cm_d_ffn=1024`, causal `mamba_ssm.Mamba` (d_state=16, expand=2, d_conv=4) + causal Conformer conv module.
- **Heads:** `ctc_head` (d_model→vocab=5001), `kl_head` (d_model→1280, regresses teacher features).
- **Single forward:** `forward(mel, mel_lens) -> (ctc_logits, kl_feat, enc_lens)`; `enc_lens` derived from `mel_lens` through the conv subsample.
- **Tokenizer:** SentencePiece unigram, vocab 5000 (+1 CTC blank).

## Critical correctness constraints (don't regress)

- **ConMamba encoder runs in fp32, always.** Selective-scan overflows fp16 over long seqs. `ConMambaStudent.forward` wraps the encoder in `autocast(enabled=False)` + `.float()`; the front-end CNN may stay fp16. Keep this island.
- **Feature distillation is cosine, not KL.** `kd_feature_loss = 1 - cosine_similarity`. Teacher tensors are raw encoder hidden states, not logits (softmax → KL≈0). Student ~25 fps vs teacher ~50 fps, so the loss **interpolates the teacher to the student length** before the frame-wise cosine, masked by `enc_lens`.
- **Grad checkpointing patches the layer CLASS, not instances.** `enable_grad_ckpt` patches `ConmambaEncoderLayer.forward` once (DataParallel-safe, eval-bypass). Idempotent.
- `/kaggle/input/...` are **read-only mounts** — never `makedirs` there; only `/kaggle/working/...` is writable (`_is_writable_path`).

## Pipeline (notebook cell order)

1. Env/GPU check, install mamba-ssm + causal-conv1d + **speechbrain**, HF login (cells 0–6)
2. **Config** dataclass — single source of truth for hyperparameters + Kaggle paths (cell 8)
3. Pseudo-labeling (10–11) · 4. Filtering (12–16) · 5. SentencePiece (17–18) · 6. Cache teacher encoder → int8 npz (19–20) — *all pre-baked as Kaggle inputs, mostly commented out*
7. **Cache mel features** (cells 21–22) — companion int8 log-mel cache, 1:1 aligned to the teacher cache
8. **ConMamba modules** (cells 24–25) — vendored SpeechBrain encoder + `mamba_ssm` import + `BiMamba` sentinel
9. **Distillation** (cell 26) — `ConMambaStudent`, `KDDataset` (+mel), `collate_kd` (+mel, `max_mel_frames`), losses, two-stage `train_stage` lives in the full-training cell
10. **Smoke test** (cell 28) — *gates* training, sets `_SMOKE_PASSED`
11. **Full training** (cell 30) — stage1 KD-only → stage2 KD+CTC, resumable
12. **Eval** (cell 32) — WER/CER + latency/RTF
13. **CTC diagnostic** (cell 34) — run if WER pins at 100%

Effective batch 32 via grad accumulation; checkpoints to `C.ckpt_dir/<stage>_latest.pt`; wandb (one continuous run via `wandb_run.json`) mirrored to append-only `training_log.jsonl`.

## Next steps (next session)

1. **Generate the mel cache** (cell 22): uncomment, run once where the source audio streams, upload as a Kaggle dataset, point `C.mel_dir` at it. **Training is blocked until this exists.**
2. **Run the smoke test** (cell 28); retune `_T4_OVERRIDES` micro-batch if it OOMs on a T4.
3. **Full training** (cell 30) → **eval** (cell 32); run **CTC diag** (cell 34) if WER pins at 100%.
4. **Watch for SpeechBrain ↔ torch 2.10 import friction** when the env cell runs — may need to pin a `speechbrain` version.
5. **Deferred:** ONNX export (mamba_ssm uses custom CUDA ops, not exportable); the front-end CNN still has a ~k//2 lookahead — make those convs causal for strict streaming.

## Working in this repo

- Windows/PowerShell, but the notebook targets Kaggle Linux — `!apt-get`/`!pip`/`/kaggle/...` are Kaggle-isms. Don't run it locally (no GPU; mamba_ssm/causal_conv1d/speechbrain are Kaggle-side).
- The notebook has **no cell `id`s** and is too large for the notebook editor — edit by manipulating the JSON with Python (`json.load`/`dump`, `ensure_ascii=False, indent=1`), locating cells by content markers, and `ast.parse`-checking each changed cell. Force UTF-8 stdout (box-drawing/em-dash chars crash cp1252).
- Not a git repo. Only `.kdedit/conmamba-src/` has its own `.git` — a **read-only** reference clone; don't edit it, and don't `import` it (it's SpeechBrain-coupled and isn't present on Kaggle — the needed modules are vendored into cell 25 instead).
- Preserve the smoke-test gate and the fp32 / cosine-loss / class-level-grad-ckpt invariants. Match the heavy-comment style — document *why* (T4 memory, fp16 overflow, frame alignment, DataParallel device bugs), not just *what*.
