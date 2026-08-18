# RAVEN: Real-time Autoregressive Video Extrapolation with Consistency-model GRPO

[Yanzuo Lu](https://yanzuo.lu/) · [Ronglai Zuo](https://2000zrl.github.io/) · [Jiankang Deng](https://jiankangdeng.github.io/) — Imperial College London

Project page: <https://yanzuo.lu/raven>

[![arXiv](https://img.shields.io/badge/arXiv-2605.15190-b31b1b.svg)](https://arxiv.org/abs/2605.15190) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-mvp--lab%2FRAVEN-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/collections/mvp-lab/raven) [![Papers with Code: #2 on VBench](https://paperswithcode.co/api/v1/papers/2605.15190/leaderboard-badge.svg?eval=23202&live=1)](https://paperswithcode.co/api/v1/papers/2605.15190/leaderboard-badge-link?eval=23202)

## TL;DR

https://github.com/user-attachments/assets/c1aa3b08-4a6e-431f-8b63-d7266774de3b

Causal autoregressive video diffusion models support real-time streaming generation by extrapolating future chunks from previously generated content. Distilling such generators from high-fidelity bidirectional teachers yields competitive few-step models, yet a persistent gap between the history distributions encountered during training and those arising at inference constrains generation quality over long horizons. We introduce the **Real-time Autoregressive Video Extrapolation Network (RAVEN)**, a training-time test framework that repacks each self rollout into an interleaved sequence of clean historical endpoints and noisy denoising states. This formulation aligns training attention with inference-time extrapolation and allows downstream chunk losses to supervise the history representations on which future predictions depend. We further propose **Consistency-model Group Relative Policy Optimization (CM-GRPO)**, which reformulates a consistency sampling step as a conditional Gaussian transition and applies online Reinforcement Learning (RL) directly to this kernel, avoiding the Euler–Maruyama auxiliary process adopted in prior flow-model RL formulations. Experiments demonstrate that RAVEN surpasses recent causal video distillation baselines across quality, semantic, and dynamic degree evaluations, and that CM-GRPO provides further gains when combined with RAVEN.

## Repository scope

This refactored repository contains the general training/launch framework and the `projects/wan_t2v` implementation required for RAVEN DMD training, CM-GRPO training, and validation-only generation. It does **not** include baseline inference implementations or a VBench evaluation harness.

The four supported trial entrypoints are:

- `projects/wan_t2v/trials/dmd/wan2_1_1_3B/causal_wan_t2v_dmd/raven.yaml`
- `projects/wan_t2v/trials/dmd/wan2_1_1_3B/causal_wan_t2v_dmd/val_only/raven_sample100.yaml`
- `projects/wan_t2v/trials/dmd/wan2_1_1_3B/causal_wan_t2v_dmd/val_only/cmgrpo_raven_sample100.yaml`
- `projects/wan_t2v/trials/grpo/wan2_1_1_3B/causal_wan_t2v_grpo/cmgrpo_raven.yaml`

The trial YAML files, including their inherited project and reward-model YAML files, are the source of truth for model construction, optimization, sampling, resources, and output naming.

## Platform and setup

The current build targets:

- Linux
- NVIDIA Hopper GPUs (SM90)
- Python 3.10
- CUDA 12.8

From the repository root:

```sh
conda env create -f tools/environment.yaml
CONDA_ENV=raven bash tools/prepare_venv.sh
source venv/bin/activate
```

`tools/prepare_venv.sh` currently defaults `CONDA_ENV` to `base`, so the documented command passes `CONDA_ENV=raven` explicitly. The build architecture variables in that script target SM90/SM90a. For non-Hopper GPUs, adjust the CUDA, FlashAttention, and MagiAttention build architecture settings before building.

## Resource paths

The shipped trial YAML files intentionally retain site-specific absolute paths. Before running a trial, edit or override every path that does not exist at your site. Do not assume that model checkpoints, datasets, released RAVEN weights, or reward weights are included in this repository.

Check the selected YAML for all required resources. The current trials reference these categories:

- **Wan2.1-T2V-1.3B resources:** diffusion backbone, VAE, UMT5 tokenizer directory, and T5 text-encoder weights.
- **Wan2.1-T2V-14B teacher:** the sharded diffusion checkpoint index used by DMD.
- **RAVEN/CM-GRPO weights:** the released `raven_model.pt`, merged `cmgrpo_raven_merge.pt`, or adapter-only `cmgrpo_raven_lora.safetensors`, depending on the chosen loading schema.
- **Training prompts:** the text prompt file under `data.args.paths`.
- **GRPO reward resources:** RAFT optical-flow weights; CLIP plus the aesthetic linear head; MUSIQ image-quality weights; AMT motion-smoothness weights; and the VideoAlign reward checkpoint plus its Qwen2-VL base model.

Review at least `models.*.weight.path`, model-directory arguments under `models.*.args`, `data.args.paths`, and all `reward_model_*` nodes. The YAML configuration—not this README—is authoritative for the exact paths and fields used by a run.

## Launcher

Launch any supported trial with:

```sh
CONDA_ENV=raven bash tools/multi_run.sh <trial.yaml> [KEY VALUE ...]
```

The script activates the requested conda environment and `venv`, then executes the equivalent of:

```sh
torchrun ... -m common.launch --config <trial.yaml> [KEY VALUE ...]
```

Launcher environment variables:

- `N`: processes per node; defaults to `SLURM_GPUS_ON_NODE` or the locally detected GPU count.
- `NNODES`: node count; defaults to `SLURM_NNODES` or `1`.
- `NODE_RANK`: node rank; defaults to `SLURM_NODEID` or `0`.
- `MASTER_ADDR`: rendezvous host; derived from the first Slurm host or defaults to `localhost`.
- `MASTER_PORT`: rendezvous port; derived from `SLURM_JOB_ID` or defaults to `7890`.
- `LOCAL_ADDR`: local address passed to `torchrun`; defaults to the current hostname's resolved address.
- `D`: debug mode. `D=0` uses the normal distributed path; `D=<n>` starts a single-node debug run with `<n>` processes.
- `DEBUGPY_PORT`: debugpy listen port when `D>0`; defaults to `5678`.

## CLI overrides

Overrides must be supplied as `KEY VALUE` pairs after the trial path. Keys use dotted paths; a numeric path segment indexes a list. Values are parsed with `yaml.safe_load`, so booleans, nulls, numbers, lists, and mappings receive YAML types rather than remaining strings.

Examples using current fields:

```sh
# Disable the first logging plugin and make a short DMD run.
CONDA_ENV=raven bash tools/multi_run.sh \
  projects/wan_t2v/trials/dmd/wan2_1_1_3B/causal_wan_t2v_dmd/raven.yaml \
  logging.plugins.0.mode disabled \
  engine.training_steps 10

# Change GRPO rollout reuse and sampling steps.
CONDA_ENV=raven bash tools/multi_run.sh \
  projects/wan_t2v/trials/grpo/wan2_1_1_3B/causal_wan_t2v_grpo/cmgrpo_raven.yaml \
  engine.offline 1 \
  diffusion.sampling_timesteps.args.num_sampling_steps 4

# Override a list element and a weight path.
CONDA_ENV=raven bash tools/multi_run.sh \
  projects/wan_t2v/trials/dmd/wan2_1_1_3B/causal_wan_t2v_dmd/raven.yaml \
  data.args.paths.0 /path/to/train_prompts.txt \
  models.tea_model.weight.path /path/to/diffusion_pytorch_model.safetensors.index.json
```

Keep every key and value as a separate shell argument. Quote values containing spaces or shell metacharacters.

## Training

### RAVEN DMD

```sh
CONDA_ENV=raven bash tools/multi_run.sh \
  projects/wan_t2v/trials/dmd/wan2_1_1_3B/causal_wan_t2v_dmd/raven.yaml
```

The DMD trial trains `models.backbone` and also requires:

- `models.fake_model`: a trainable Wan2.1-T2V-1.3B fake model.
- `models.tea_model`: the frozen Wan2.1-T2V-14B teacher.
- Wan 1.3B VAE, tokenizer, and text encoder resources.
- A training prompt file in `data.args.paths`.

Update all site-specific paths before launching.

### CM-GRPO

```sh
CONDA_ENV=raven bash tools/multi_run.sh \
  projects/wan_t2v/trials/grpo/wan2_1_1_3B/causal_wan_t2v_grpo/cmgrpo_raven.yaml
```

The GRPO trial starts from `raven_model.pt` via `models.backbone.weight.path`. It attaches and trains the LoRA defined by `models.backbone.adapter`; the current schema uses `r`, `lora_alpha`, and `target_modules`.

CM-GRPO also requires all five configured reward models:

1. RAFT motion/dynamic reward (`reward_model_raft`).
2. Aesthetic reward (`reward_model_aq`).
3. Imaging-quality reward (`reward_model_iq`).
4. Motion-smoothness reward (`reward_model_ms`).
5. VideoAlign text-alignment reward (`reward_model_videoalign`).

Set every reward checkpoint and base-model path in the YAML before running.

## Validation-only generation

Generate the 100-prompt RAVEN validation set:

```sh
CONDA_ENV=raven bash tools/multi_run.sh \
  projects/wan_t2v/trials/dmd/wan2_1_1_3B/causal_wan_t2v_dmd/val_only/raven_sample100.yaml
```

Generate the 100-prompt merged CM-GRPO validation set:

```sh
CONDA_ENV=raven bash tools/multi_run.sh \
  projects/wan_t2v/trials/dmd/wan2_1_1_3B/causal_wan_t2v_dmd/val_only/cmgrpo_raven_sample100.yaml
```

Both trials set:

```yaml
engine:
  training_steps: 0
  val_before_train: true
  resume: never
```

They perform validation generation only. Their run directories use the YAML stems:

- `runs/distribution_matching_distillation/raven_sample100/`
- `runs/distribution_matching_distillation/cmgrpo_raven_sample100/`

Validation videos are written below `media/0000000/validation/backbone/`. Re-running the same YAML stem uses the same run directory and skips prompt indices that already have MP4 files, providing file-level continuation. This skip logic is based on existing prompt MP4s even though checkpoint resume is disabled.

Inspect `logs/log_rank0.txt` before trusting outputs. For a full or merged backbone, confirm a message ending in `clean load`; for an adapter-only load, also confirm that adapter loading reports no unexpected keys.

## Current weights and LoRA schema

### RAVEN or merged CM-GRPO backbone

Load `raven_model.pt` or a merged `cmgrpo_raven_merge.pt` as the backbone weight and do not add an adapter:

```yaml
models:
  backbone:
    weight:
      path: /path/to/raven_model.pt
```

For the merged checkpoint, replace the path with `/path/to/cmgrpo_raven_merge.pt`.

### Adapter-only CM-GRPO

Use `raven_model.pt` as the base backbone, retain the adapter structure from the GRPO YAML, and add the adapter weight inside `models.backbone.adapter`:

```yaml
models:
  backbone:
    weight:
      path: /path/to/raven_model.pt
    adapter:
      r: 256
      lora_alpha: 256
      target_modules:
        - text_embedding.0
        - text_embedding.2
        - time_embedding.0
        - time_embedding.2
        - time_projection.1
        - self_attn.q
        - self_attn.k
        - self_attn.v
        - self_attn.o
        - cross_attn.q
        - cross_attn.k
        - cross_attn.v
        - cross_attn.o
        - ffn.0
        - ffn.2
        - head.head
      weight: /path/to/cmgrpo_raven_lora.safetensors
```

The old JSONC-style `lora: {enabled, weight}` block is not supported by the current schema.

`cmgrpo_raven_full.pt` is a full PEFT bundle, not a directly interchangeable backbone file for the loading patterns above. Convert or merge it offline into a current-schema adapter-only or merged checkpoint before using it with these trials.

## Outputs and resume

A run directory is derived as:

```text
runs/<persistence.proj_name>/<yaml-stem>/
```

Important subdirectories include:

- `configs/`: original and resolved configuration snapshots, CLI overrides, and git-state metadata.
- `checkpoints/`: numbered distributed checkpoints for training runs.
- `media/<step>/`: validation videos and optional grids.

The DMD and GRPO training YAML files use `engine.resume: auto`, which loads the latest complete checkpoint in the same run directory when present. Renaming a YAML file changes its stem, therefore changes the run directory and its automatic-resume location. Validation-only YAML files instead use `resume: never`, while retaining file-level MP4 continuation as described above.

## Minimal config check

The following command resolves inheritance and parses all four trial YAML files without constructing models or loading weights:

```sh
python - <<'PY'
from common.config import CfgNode

paths = [
    "projects/wan_t2v/trials/dmd/wan2_1_1_3B/causal_wan_t2v_dmd/raven.yaml",
    "projects/wan_t2v/trials/dmd/wan2_1_1_3B/causal_wan_t2v_dmd/val_only/raven_sample100.yaml",
    "projects/wan_t2v/trials/dmd/wan2_1_1_3B/causal_wan_t2v_dmd/val_only/cmgrpo_raven_sample100.yaml",
    "projects/wan_t2v/trials/grpo/wan2_1_1_3B/causal_wan_t2v_grpo/cmgrpo_raven.yaml",
]

for path in paths:
    cfg = CfgNode.from_file(path)
    print(path, "->", cfg.entry.module, cfg.entry.class_name)
PY
```

Run it from the repository root after activating `venv`.

## Citation

If you find this work useful, please cite RAVEN. A BibTeX entry will be added when available.

```bibtex
@article{lu2026raven,
  title = {RAVEN: Real-time Autoregressive Video Extrapolation with Consistency-model GRPO},
  author = {Lu, Yanzuo and Zuo, Ronglai and Deng, Jiankang},
  year = 2026,
  journal = {arXiv preprint arXiv:2605.15190}
}
```
