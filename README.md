## Folder Structure 
```
ai_gen_detect/
├─ ai_gen_false/
|├─ README.md                        # Experiment Description & Replication Instructions
|├─ analysis.ipynb                   # Jupyter Notebook with analysis, visualizations & statistical testing
|├─ experiment.py                    # Python Script to generate dataset, exports .csv by default
|├─ requirements.txt                 # dependencies
├─ ai_gen_true/
|├─ README.md                        # Experiment Description & Replication Instructions
|├─ analysis.ipynb                   # Jupyter Notebook with analysis, visualizations & statistical testing
|├─ experiment.py                    # Python Script to generate dataset, exports .csv by default
|├─ requirements.txt                 # dependencies
audito-jsonl/
batch_triage/
├─ config/                           # YAML configuration you edit
├─ scripts/                          # CLI entrypoints (generate + run)
├─ src/                              # core Python modules (batch client, logging, parsing)
├─ batch/                            # generated JSONL/CSV inputs per model
├─ outputs/                          # batch outputs and compact CSVs
├─ state/                            # runtime state (resume/polling)
└─ requirements.txt                  # dependencies
docker/
legal/legal_service_audit/
├─ analysis.ipynb                    # Jupyter Notebook with analysis, visualizations & statistical testing
├─ legal_services_audit_handoff.csv  # Dataset obtained from legal services audit
mental_health/
├─ analysis.ipynb                    # Jupyter Notebook with analysis, visualizations & statistical testing
├─ medical_data.csv                  # Dataset obtained from mental health audit
├─ requirements.txt                  # dependencies 
employment_screening/
├─ analysis.ipynb                    # Jupyter Notebook with analysis, visualizations & statistical testing
├─ results.csv                       # Dataset obtained from employment screening audit
├─ requirements.txt                  # dependencies 
```
