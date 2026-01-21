# Batch Triage Experiments

This repo helps you generate OpenAI-style batch JSONL payloads and run them
either via OpenAI Batch or a local vLLM endpoint. Run commands from the repo
root so relative paths resolve correctly.

## Quickstart (uv + requirements.txt)

Create and activate a virtual environment, then install dependencies:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Windows PowerShell:

```powershell
uv venv
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

## Configure Your Experiment

Edit these YAMLs under `config/`:

- `config/config.yaml`: models to test, prompt template, variables, paths,
  batch endpoint, polling, W&B project.
- `config/logging.yaml`: which input/output fields to store in
  `results_compact`.
- `config/vllm.yaml`: vLLM endpoint and request settings (only for vLLM runs).
- `config/logging.yaml`: results_compact field - which input / output features to upload to wandb

## Run an Experiment

1) Generate per-model JSONL/CSV inputs:

```bash
python scripts/generate_batch_payloads.py
```

2) Submit and poll OpenAI Batch (use `--dry-run` to validate and estimate cost):

```bash
python scripts/run_openai_batch.py --dry-run
python scripts/run_openai_batch.py
```

Optional: replay JSONL against a local vLLM endpoint:

```bash
python scripts/run_vllm_http.py
```

## Folder Structure (What Goes Where)

```
batch_triage/
├─ config/              # YAML configuration you edit
├─ scripts/             # CLI entrypoints (generate + run)
├─ src/                 # core Python modules (batch client, logging, parsing)
├─ batch/               # generated JSONL/CSV inputs per model
├─ outputs/             # batch outputs and compact CSVs
├─ state/               # runtime state (resume/polling)
└─ requirements.txt     # dependencies
```

## Important Notes

- Set `OPENAI_API_KEY` for OpenAI Batch runs. Set `WANDB_API_KEY` if you want
  W&B logging; otherwise pass `--no-wandb`.
- `scripts/run_openai_batch.py --dry-run` is the safest first step to validate
  config and estimate costs before submitting.
- Generated files appear under `batch/` and `outputs/`; delete them if you want
  a clean slate between experiments.
