# LLM Fairness Audits

This repository contains a set of domain-specific LLM fairness audits plus a
shared extraction/runtime pipeline in `data_extraction/`. The audit folders
focus on controlled perturbation experiments: task-relevant evidence is held
constant while demographic/context variables are varied to test consistency and
potential bias in model outputs. Domains may also use additional extraction
pipelines as needed; they are not required to use only `data_extraction/`.

## Repository Structure (Non-Ignored)

```text
.
├─ audit_1_ai_gen_detect/
│  ├─ ai_gen_false/
│  │  ├─ README.md
│  │  ├─ experiment.py
│  │  ├─ analysis.ipynb
│  │  └─ detec_false_analysis.html
│  └─ ai_gen_true/
│     ├─ README.md
│     ├─ experiment.py
│     ├─ analysis.ipynb
│     └─ detec_true_analysis.html
├─ audit_2_employment_screening/
│  ├─ README.md
│  ├─ results.csv
│  └─ analysis.ipynb
├─ audit_3_legal/
│  └─ legal_service_audit/
│     ├─ legal_services_audit_handoff.csv
│     └─ analysis.ipynb
├─ audit_4_mental_health/
│  ├─ README.md
│  ├─ medical_data.csv
│  └─ analysis.ipynb
├─ audit_5_service_eval/
│  ├─ layoffs/
│  │  ├─ README.md
│  │  ├─ experiment.py
│  │  ├─ statistical_utils.py
│  │  ├─ analysis.ipynb
│  │  └─ eval_neg_analysis.html
│  └─ reward/
│     ├─ README.md
│     ├─ experiment.py
│     ├─ statistical_utils.py
│     ├─ analysis.ipynb
│     └─ eval_pos_analysis.html
└─ data_extraction/
   ├─ README.MD
   ├─ requirements.txt
   ├─ config/                          # model/pricing/logging/limits/vLLM YAMLs
   ├─ scripts/                         # payload generation and batch run scripts
   ├─ src/                             # config, cost, OpenAI batch, state, W&B modules
   └─ batch/                           # generated request payloads used by some runs
```

## What Each Audit Covers

- `audit_1_ai_gen_detect/`: AI-generated text detection scoring audit with two
  conditions (`ai_gen_false` and `ai_gen_true`) using controlled name/level
  perturbations.
- `audit_2_employment_screening/`: Employment recommendation fairness audit
  where demographic attributes are perturbed while applicant qualifications stay
  fixed.
- `audit_3_legal/legal_service_audit/`: Legal-services triage audit on housing
  scenarios, measuring recommendation and risk outputs across perturbed inputs.
- `audit_4_mental_health/`: Mental-health triage audit measuring urgency score
  and triage level under controlled demographic/context perturbations.
- `audit_5_service_eval/`: Teacher service evaluation audit with two policy
  framings: `layoffs` (negative framing) and `reward` (positive framing).
- `data_extraction/`: Shared extraction/runtime utilities for batch payload
  creation and model execution; optional for domains that need custom pipelines.

For replication details, start with the README inside each experiment folder.
