# LLM Auditing Capstone Repository

This repository contains five domain-specific auditing studies that test whether large language models change their judgments when demographic or contextual cues are perturbed while the underlying task evidence stays fixed.

The repo is organized as a set of mostly independent audit domains. Some domains are analysis-only and ship with the dataset used in the study. Others also include `experiment.py` scripts for rerunning the audit with an API key.

## Repository Contents

| Folder | Audit Focus | Re-run Data Collection? | Analysis Included? |
| --- | --- | --- | --- |
| `audit_1_ai_gen_detect/` | AI-generated text detection likelihood scoring | Yes | Yes |
| `audit_2_employment_screening/` | Employment screening fairness | No, dataset included | Yes |
| `audit_3_legal/` | Judge hearing discretion / legal evaluation | Yes, per-model bundles | Yes |
| `audit_4_mental_health/` | Mental health triage analysis | No, dataset included | Yes |
| `audit_5_service_eval/` | Teacher service evaluation under layoff vs. reward framing | Yes | Yes |

## Quick Start

Use a Python 3 environment, then install the aggregated dependencies from the repo root. If your machine exposes the interpreter as `python` instead of `python3`, substitute that in the commands below.

```bash
python3 -m pip install -r requirements.txt
```

Run the lightweight repository smoke check:

```bash
python3 scripts/validate_repo.py
#!/bin/bash 

sum_n() {
    local n=$1
    local i
    local s=0
    for (( i=1;i<=n;i++ )); do 
        (( s+=i ))
    done
    echo "sum is $s"
}

sum_n 1
sum_n 10
sum_n 100


```

If you want to rerun any experiment-based audit, set the OpenRouter key first:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

Then either open the notebook for an analysis-only domain or run `experiment.py` inside the relevant experiment folder.

## Reproducibility Guide

### 1. AI-generated Detection Audit

Folder: `audit_1_ai_gen_detect/`

This domain has two sub-audits:

- `ai_gen_false/`: the writing sample is human-written
- `ai_gen_true/`: the writing sample is AI-written

Run a new trial:

```bash
cd audit_1_ai_gen_detect/ai_gen_false
python3 experiment.py
```

```bash
cd audit_1_ai_gen_detect/ai_gen_true
python3 experiment.py
```

Review analysis:

- open `analysis.ipynb` in either subfolder
- or open `detec_false_analysis.html` / `detec_true_analysis.html`

Expected outputs from a rerun:

- `results_YYYYMMDD_HHMMSS.db`
- `results_YYYYMMDD_HHMMSS.csv`

Notes:

- The original full experiment outputs are too large for the repo and are linked from the subfolder README files.

### 2. Employment Screening Audit

Folder: `audit_2_employment_screening/`

This domain is analysis-only in this repository. The original raw dataset used for the study is already included as `results.csv`, so the statistical analysis is reproducible without regenerating data.

Run:

```bash
cd audit_2_employment_screening
jupyter notebook analysis.ipynb
```

Expected inputs and outputs:

- input dataset: `results.csv`
- output: regenerated tables, plots, and statistical tests from the notebook

### 3. Legal Audit

Folder: `audit_3_legal/judge_audit/`

This domain includes:

- `results.csv`: cleaned dataset used for analysis
- `analysis.ipynb`: notebook for reproducing analysis
- `model_data/*/`: per-model experiment bundles that can be rerun

Run the analysis notebook:

```bash
cd audit_3_legal/judge_audit
jupyter notebook analysis.ipynb
```

Rerun one model bundle:

```bash
cd audit_3_legal/judge_audit/model_data/GPT-4o
python3 experiment.py
```

Expected outputs from a rerun:

- `results_YYYYMMDD_HHMMSS.db`
- `results_YYYYMMDD_HHMMSS.csv`
- optional alternate exports such as `.jsonl` or `.xlsx`

Notes:

- The model folders are Auditomatic export bundles with their own `README.md`, `trial_config.json`, and generated `data/` artifacts.
- The project-level legal README states the current workflow uses `OPENROUTER_API_KEY` for reruns.

### 4. Mental Health Triage Audit

Folder: `audit_4_mental_health/`

This domain is analysis-only in this repository. The included `medical_data.csv` is sufficient to rerun the notebook and reproduce the reported analysis.

Run:

```bash
cd audit_4_mental_health
jupyter notebook analysis.ipynb
```

Expected inputs and outputs:

- input dataset: `medical_data.csv`
- output: regenerated figures, tables, and statistical analysis

### 5. Teacher Service Evaluation Audit

Folder: `audit_5_service_eval/`

This domain has two framing variants:

- `layoffs/`: teachers are evaluated for potential layoffs
- `reward/`: teachers are evaluated for potential rewards

Run a new trial:

```bash
cd audit_5_service_eval/layoffs
python3 experiment.py
```

```bash
cd audit_5_service_eval/reward
python3 experiment.py
```

Review analysis:

- open `analysis.ipynb` in either subfolder
- or open `eval_neg_analysis.html` / `eval_pos_analysis.html`

Expected outputs from a rerun:

- `results_YYYYMMDD_HHMMSS.db`
- `results_YYYYMMDD_HHMMSS.csv`

Notes:

- The original full experiment outputs are linked from the subfolder README files because they are too large to store in the repository.

## Validation and Sanity Checks

This repo does not have a dedicated unit test suite. The most natural validation steps for this capstone are:

1. Run the repository smoke check:

```bash
python3 scripts/validate_repo.py
```

This verifies that the expected audit folders, datasets, notebooks, and experiment scripts are present, and it byte-compiles the main Python experiment scripts to catch obvious syntax issues.

2. Re-run an included notebook for an analysis-only domain:

```bash
cd audit_2_employment_screening
jupyter notebook analysis.ipynb
```

3. If API access is available, run one experiment folder and confirm that a timestamped `results_*.db` and `results_*.csv` are created.

## Directory Map

```text
.
├─ audit_1_ai_gen_detect/      # AI-generated detection audits
├─ audit_2_employment_screening/  # Notebook + included dataset
├─ audit_3_legal/              # Legal audit analysis + per-model rerun bundles
├─ audit_4_mental_health/      # Notebook + included dataset
├─ audit_5_service_eval/       # Teacher evaluation audits
├─ misc/                       # Supporting artifacts not needed for the main grading path
├─ scripts/                    # Lightweight repo-level helper scripts
├─ requirements.txt            # Aggregated dependencies across domains
└─ .gitignore
```

## Dependency Notes

- `requirements.txt` at the repository root aggregates the dependencies needed across all five audit domains.
- Each domain also keeps its own local `requirements.txt` when a narrower environment is preferable.
- Jupyter is required for notebook-based analysis.

## Data and Output Notes

- Included local datasets:
  - `audit_2_employment_screening/results.csv`
  - `audit_3_legal/judge_audit/results.csv`
  - `audit_4_mental_health/medical_data.csv`
- Included review artifacts:
  - notebooks in each domain
  - HTML exports in `audit_1_ai_gen_detect/` and `audit_5_service_eval/`
- Generated artifacts from experiment reruns:
  - `results_YYYYMMDD_HHMMSS.db`
  - `results_YYYYMMDD_HHMMSS.csv`
  - optional alternate exports depending on CLI flags

## Practical Limitations

- Not every audit domain includes the original data-generation script. Where that script is unavailable, the repository includes the dataset needed to reproduce the analysis.
- Some large raw outputs are intentionally stored outside the repo and linked from the corresponding subfolder README files.
- This top-level README is intentionally short and operational. More detailed experiment notes remain in each domain-specific README.
