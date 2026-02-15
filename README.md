# LLM Fairness Audits

This repository contains domain-specific LLM fairness audits and supporting
extraction workflows. The audits use controlled perturbation designs where
task-relevant evidence is held constant and demographic or context variables are
varied to evaluate consistency in model behavior.

`data_extraction/` provides a shared extraction and batch-running pipeline.
Domains can also use additional extraction workflows when needed.

## Repository Structure

```text
.
├─ audit_1_ai_gen_detect/
│  ├─ ai_gen_false/
│  └─ ai_gen_true/
├─ audit_2_employment_screening/
├─ audit_3_legal/
│  ├─ judge_audit/
│  └─ legal_service_audit/
├─ audit_4_mental_health/
├─ audit_5_service_eval/
│  ├─ layoffs/
│  └─ reward/
└─ data_extraction/
   ├─ batch/
   ├─ config/
   ├─ scripts/
   └─ src/
      ├─ openai_batch/
      └─ wandb_logging/
```

## Folder Overview

- `audit_1_ai_gen_detect/`: AI-generated text detection likelihood audit with
  paired false and true writing conditions.
- `audit_2_employment_screening/`: Hiring-screening fairness audit under
  demographic perturbations.
- `audit_3_legal/judge_audit/`: Judge-evaluation fairness audit with controlled
  demographic/context variables.
- `audit_3_legal/legal_service_audit/`: Legal-service triage fairness audit for
  housing-related scenarios.
- `audit_4_mental_health/`: Mental-health triage fairness audit on urgency and
  triage-level outputs.
- `audit_5_service_eval/`: Teacher service evaluation audit with two framings:
  `layoffs` and `reward`.
- `data_extraction/`: Shared tooling for payload generation and model batch
  execution.

Each experiment folder includes its own README with run and analysis details.
