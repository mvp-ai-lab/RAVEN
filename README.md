# RAVEN: Real-time Autoregressive Video Extrapolation with Consistency-model GRPO

[Yanzuo Lu](https://yanzuo.lu/) · [Ronglai Zuo](https://2000zrl.github.io/) · [Jiankang Deng](https://jiankangdeng.github.io/) — Imperial College London

Project page: <https://yanzuo.lu/raven>

[![arXiv](https://img.shields.io/badge/arXiv-2605.15190-b31b1b.svg)](https://arxiv.org/abs/2605.15190) [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-mvp--lab%2FRAVEN-FFD21E)](https://huggingface.co/collections/mvp-lab/raven) [![Papers with Code: #2 on VBench](https://paperswithcode.co/api/v1/papers/2605.15190/leaderboard-badge.svg?eval=23202&live=1)](https://paperswithcode.co/api/v1/papers/2605.15190/leaderboard-badge-link?eval=23202)

## News

- **August 19, 2026.** We released the MiniMax-H3 training code and initial [4-NFE preview LoRA weights](https://huggingface.co/mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA). This first preview is still undertrained and its texture details remain limited, but it establishes the complete end-to-end training pipeline for streaming generation on MiniMax-H3. More to come.
- **May 14, 2026.** We made the initial commit and released the [paper](https://arxiv.org/abs/2605.15190), code, and [weights](https://huggingface.co/mvp-lab/RAVEN). The initial implementation and released weights were validated on Wan2.1-T2V-1.3B.

## Releases

| Models | Checkpoints | Description |
| --- | --- | --- |
| MiniMax-H3-RAVEN-Streaming-LoRA-4NFE-Preview | 🤗 [HuggingFace](https://huggingface.co/mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA/blob/main/minimax_h3_raven_streaming_lora_4nfe_preview.safetensors) | Initial 4-NFE streaming LoRA preview. It remains undertrained and has limited texture detail. |
| Wan2.1-T2V-1.3B-RAVEN | 🤗 [HuggingFace](https://huggingface.co/mvp-lab/RAVEN/blob/main/raven_model.pt) | RAVEN backbone after distillation and before CM-GRPO. |
| Wan2.1-T2V-1.3B-CMGRPO-LoRA | 🤗 [HuggingFace](https://huggingface.co/mvp-lab/RAVEN/blob/main/cmgrpo_raven_lora.safetensors) | CM-GRPO adapter only. Load it on top of Wan2.1-T2V-1.3B-RAVEN. |
| Wan2.1-T2V-1.3B-CMGRPO-Merge | 🤗 [HuggingFace](https://huggingface.co/mvp-lab/RAVEN/blob/main/cmgrpo_raven_merge.pt) | Full CM-GRPO backbone with the adapter merged. |
| Wan2.1-T2V-1.3B-CMGRPO-Full | 🤗 [HuggingFace](https://huggingface.co/mvp-lab/RAVEN/blob/main/cmgrpo_raven_full.pt) | PEFT-wrapped base-and-adapter bundle for archival interchange. Convert or merge it before use with the current code. |

## Demo

https://github.com/user-attachments/assets/96047839-e7cb-4416-a82b-4acb59f8bab3

https://github.com/user-attachments/assets/628a070c-fee5-4b77-97af-5875e79fca1d

## Repository layout

This repository is a general, project-agnostic training and launch framework plus a set of concrete projects built on top of it.

| Path | Contents |
| --- | --- |
| `common/` | Core framework for launching, configuration, model construction, engines, diffusion, distributed execution, optimization, persistence, logging, and plugins. |
| `engines/` | Reusable training engines shared across projects (diffusion finetuning, DMD, GRPO, TSCD). |
| `modeling/` | Shared, project-independent models, currently the reward models. |
| `configs/` | Shared config fragments for those shared models. |
| `projects/<project>/` | Self-contained project packages with `configs/`, `data/`, `meta_models/`, `modeling/`, `trials/`, and optional `tools/`. |
| `utils/` | Low-level kernels and helpers (attention backends, caches). |
| `tools/` | Environment and launch scripts. |

The repository currently includes two projects. They are `projects/wan_t2v` for the paper's experiments and `projects/minimax_h3`.

Nothing in `common/` imports from `projects/`. Projects are reached only through the dynamic `{module, class_name}` references written in a trial config.

## Setup

The current build targets Linux, NVIDIA Hopper GPUs (SM90), Python 3.10, and CUDA 12.8.

Run these commands from the repository root.

```sh
conda env create -f tools/environment.yaml
bash tools/prepare_venv.sh
source venv/bin/activate
```

The repository scripts activate the `raven` conda environment by default. The build-architecture variables in `tools/prepare_venv.sh` target SM90/SM90a. On non-Hopper GPUs, adjust the CUDA, FlashAttention, and MagiAttention architecture settings before building.

Dependencies are pinned in `tools/requirements.lock`. If you change `tools/requirements.txt`, regenerate the lock with the command below.

```sh
bash tools/compile_dependencies.sh
```

## Running a trial

A *trial* is a single YAML file describing one complete run. Use the command below to launch any trial.

```sh
bash tools/multi_run.sh <trial.yaml> [KEY VALUE ...]
```

The script activates the conda environment and `venv`, then runs the equivalent of the following command.

```sh
torchrun ... -m common.launch --config <trial.yaml> [KEY VALUE ...]
```

Always launch from the repository root. Relative `__inherit__` paths and the default `runs/` output directory are resolved against the current working directory.

The launcher supports these environment variables.

| Variable | Meaning | Default |
| --- | --- | --- |
| `N` | Processes per node | `SLURM_GPUS_ON_NODE`, else detected GPU count |
| `NNODES` | Node count | `SLURM_NNODES`, else `1` |
| `NODE_RANK` | This node's rank | `SLURM_NODEID`, else `0` |
| `MASTER_ADDR` | Rendezvous host | First Slurm host, else `localhost` |
| `MASTER_PORT` | Rendezvous port | Derived from `SLURM_JOB_ID`, else `7890` |
| `LOCAL_ADDR` | Local address passed to `torchrun` | Resolved hostname address |
| `D` | Debug mode. `0` selects the normal distributed path, while `<n>` starts a single-node debug run with `<n>` processes | `0` |
| `DEBUGPY_PORT` | debugpy listen port when `D>0` | `5678` |

## Trial configuration

A trial YAML is a plain tree of sections. The table below lists the common ones.

| Section | Purpose |
| --- | --- |
| `entry` | The engine class to run. Instantiated as `cls(config, meta_model)` and must expose `run()`. |
| `meta_model` | The project's algorithm object. Instantiated as `cls(config)` and handed to the engine. |
| `models` | One node per model. Each node drives the build pipeline below. |
| `data` | Dataset class and its `args`, plus loader settings. |
| `diffusion` | Schedule, sampler, and sampling-timestep objects. |
| `engine` | Training and validation loop knobs such as steps, resume policy, and gradient accumulation. |
| `validation` | What to generate for validation and how to write it. |
| `distributed` | Parallelism sizes for the unified-parallel setup. |
| `persistence` | `proj_name`, `output_dir`, checkpoint behaviour. |
| `logging` | Text/metrics logging and logging plugins (e.g. Weights & Biases). |

Every pluggable object is named by a `{module, class_name}` pair and imported dynamically at run time. There is **no registry**. Adding a class requires no registration step, and the config is the only place a class name appears.

`common.engine.BaseEngine` is the default validation-only engine. Its `run()` performs one `validate()` and returns. Training engines (in `engines/` or a project) build their own optimizers, dataloaders, and checkpoint trees and override `run()`.

## Inheritance and CLI overrides

Any mapping node may carry `__inherit__: <file>`. The node is replaced by that file's recursively resolved content, and the node's remaining keys are deep-merged on top. A trial can therefore inherit a whole model-args file and override single fields in place.

```yaml
models:
  backbone:
    args:
      __inherit__: projects/minimax_h3/configs/dit.yaml
      # local keys here override the inherited ones
```

Relative paths resolve against the containing file's directory first, then against the current working directory (the repository root).

Command-line overrides come after the trial path as `KEY VALUE` pairs. Keys are dotted paths, and a numeric segment indexes a list. Values are parsed with `yaml.safe_load`, so booleans, nulls, numbers, lists, and mappings get real YAML types instead of strings.

```sh
bash tools/multi_run.sh <trial.yaml> \
  engine.training_steps 10 \
  logging.plugins.0.mode disabled \
  models.backbone.weight.path /path/to/weights.safetensors \
  data.args.paths.0 /path/to/prompts.txt
```

Keep every key and value as a separate shell argument. The current `multi_run.sh` wrapper forwards overrides without preserving embedded whitespace, so avoid spaces and shell metacharacters in override values.

## Model build

Each entry under `models` is built by one fixed pipeline (`common/model/`).

1. **instantiate**. Constructs the module from `module` / `class_name` / `args`, optionally under `meta_init` or an alternative `instantiate` strategy such as `FromPretrainedInstantiate`.
2. **weight**. Loads the initial weights described by `weight` (a `path`, and optionally a project-specific loader class).
3. **adapter**. Attaches the LoRA/PEFT adapter described by `adapter`, if present.
4. **runtime**. Sets `training`, `requires_grad`, `param_dtype`, and wraps the module with runtime plugins.
5. **placement**. Applies FSDP wrapping, shard sizes, dtypes, and CPU offload.

Two reminders apply to every project.

- `weight.path` names *initial* weights, not a resume point. Resuming a run reads the run's own checkpoints instead (see below).
- An adapter is configured under the `adapter` key, which currently accepts `r`, `lora_alpha`, `target_modules`, an optional `weight` file, and optional custom module mappings. Older `lora: {...}` blocks are not part of this schema.

## Resources

Trial YAML files retain site-specific **absolute paths**. Before running anything, edit the YAML or override the paths on the command line. This repository ships no base model checkpoints, datasets, or reward weights. Check the selected trial for what it needs, starting with `models.*.weight.path`, model-directory arguments under `models.*.args`, and `data.args`.

The resolved configuration is the single source of truth, not this README. Each run writes its original and resolved config, CLI overrides, and git state under `runs/<proj_name>/<exp_name>/configs/`.

After launching, inspect `logs/log_rank0.txt` before trusting outputs. A full or merged backbone load should report a message ending in `clean load`. An adapter load should report no unexpected keys.

## Outputs and resume

A run directory takes the following form.

```text
<persistence.output_dir>/<persistence.proj_name>/<persistence.exp_name>/
```

The value of `output_dir` is usually `runs/`. `persistence.proj_name` is **required** and never inferred. `persistence.exp_name` defaults to the trial YAML's file stem. Renaming a trial file therefore changes the run directory and its automatic-resume location.

A run directory holds the following subdirectories.

| Directory | Contents |
| --- | --- |
| `configs/` | Original and resolved config snapshots, CLI overrides, git-state metadata. |
| `logs/` | Per-rank text logs. |
| `metrics/` | `metrics.jsonl`. |
| `checkpoints/<NNNNNNN>/` | Distributed (DCP) checkpoints. |
| `media/<step>/` | Generated videos and optional grids. |

`engine.resume` accepts `auto` (latest complete checkpoint in this run directory, if any), `never`, or an explicit integer step. A checkpoint counts as complete only when its directory contains the DCP `.metadata` marker. Incomplete directories are ignored by `auto` and rejected on explicit load. `engine.resume_dir` can point resume at another run's `checkpoints` directory.

Validation-only trials typically set `resume: never`. Some projects additionally implement file-level continuation for generation runs, skipping prompts that already have output files in the run's media directory. This is project-specific behaviour implemented in the project's meta model, not a framework guarantee. Check the project you are running.

## Projects

### MiniMax-H3

Training implementations live in `projects/minimax_h3/meta_models/`, including causal/streaming teacher-forcing, DMD, DMD2, and TSCD paths, together with base and causal validation implementations.

Two runnable trials are bundled. Both use `common.engine.BaseEngine`, so they are validation-only generation runs rather than training trials.

- `projects/minimax_h3/trials/base/minimax_h3_base/minimax_h3_base_50nfe.yaml` runs the bidirectional base model with 50-NFE DDIM sampling.
- `projects/minimax_h3/trials/base/causal_minimax_h3_base/minimax_h3_raven_streaming_lora_4nfe_preview.yaml` runs the causal streaming model with the 4-NFE preview LoRA and a consistency sampler.

```sh
N=8 bash tools/multi_run.sh \
  projects/minimax_h3/trials/base/causal_minimax_h3_base/minimax_h3_raven_streaming_lora_4nfe_preview.yaml
```

The bundled preview trial is configured for 4-way unified parallelism and an FSDP shard size of 8, so the command above uses one 8-GPU node. Adjust the distributed and placement settings together for another topology.

The preview LoRA is at [mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA](https://huggingface.co/mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA). Set its local path in `models.backbone.adapter.weight`. Both trials additionally require the MiniMax-H3 base components (DiT, text encoder, video VAE, audio VAE, tokenizer), which are **not** included here and are governed by the [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). Obtain them separately and point the corresponding paths at your local copies.

### Wan T2V

`projects/wan_t2v` is the project used for the paper's experiments (RAVEN DMD distillation, CM-GRPO, and validation-only generation on Wan2.1-T2V-1.3B). It bundles the following trials.

- `projects/wan_t2v/trials/dmd/wan2_1_1_3B/causal_wan_t2v_dmd/raven.yaml`
- `projects/wan_t2v/trials/dmd/wan2_1_1_3B/causal_wan_t2v_dmd/val_only/raven_sample100.yaml`
- `projects/wan_t2v/trials/dmd/wan2_1_1_3B/causal_wan_t2v_dmd/val_only/cmgrpo_raven_sample100.yaml`
- `projects/wan_t2v/trials/grpo/wan2_1_1_3B/causal_wan_t2v_grpo/cmgrpo_raven.yaml`

```sh
bash tools/multi_run.sh \
  projects/wan_t2v/trials/dmd/wan2_1_1_3B/causal_wan_t2v_dmd/raven.yaml
```

The trial YAML files and the project/reward-model YAML files they inherit are the source of truth for the models, resources, and weights each run needs.

## Export and tools

The export tool extracts a tensor subtree from a DCP checkpoint into a safetensors or torch file, without instantiating any model.

```sh
python -m common.plugin.export \
  --run-dir runs/<proj_name>/<exp_name> \
  --key models.backbone \
  --out /path/to/out.safetensors \
  --content full
```

`--content` selects `full` (the subtree verbatim), `peft` (only PEFT/LoRA tensors, remapped to the single-adapter layout), or `merged` (LoRA deltas folded into the base weights, which requires `--lora-alpha`). Use `--checkpoint-dir` instead of `--run-dir` to read one checkpoint directly, or `--step` to pick a specific step. Run `python -m common.plugin.export --help` for the full argument list.

For large MiniMax-H3 weight files, convert them to a sharded DCP checkpoint once so that many ranks load their own slices instead of faulting through one file.

```sh
torchrun --nproc_per_node=16 projects/minimax_h3/tools/convert_to_dcp.py \
  --src /path/to/weights.safetensors \
  --out /path/to/weights.dcp
```

See the module docstring in `projects/minimax_h3/tools/convert_to_dcp.py` for the released-DiT layout flag and the other options.

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
