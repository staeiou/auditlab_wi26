## Folder Structure 
```
audit_1_ai_gen_detect/
├─ ai_gen_false/
|├─ README.md                        # Experiment Description & Replication Instructions
|├─ analysis.ipynb                   # Jupyter Notebook with analysis, visualizations & statistical testing
|├─ detec_false_analysis.html        # HTML analysis export for ease of reading via web browser
|├─ experiment.py                    # Python Script to generate dataset, exports .csv by default
|└─ requirements.txt                 # dependencies
├─ ai_gen_true/
|├─ README.md                        # Experiment Description & Replication Instructions
|├─ analysis.ipynb                   # Jupyter Notebook with analysis, visualizations & statistical testing
|├─ detec_true_analysis.html         # HTML analysis export for ease of reading via web browser
|├─ experiment.py                    # Python Script to generate dataset, exports .csv by default
|└─ requirements.txt                 # dependencies
data_extraction/
├─ config/                           # YAML configuration you edit
├─ scripts/                          # CLI entrypoints (generate + run)
├─ src/                              # core Python modules (batch client, logging, parsing)
├─ batch/                            # generated JSONL/CSV inputs per model
├─ outputs/                          # batch outputs and compact CSVs
├─ state/                            # runtime state (resume/polling)
└─ requirements.txt                  # dependencies
audit_2_employment_screening/
├─ analysis.ipynb                    # Jupyter Notebook with analysis, visualizations & statistical testing
├─ results.csv                       # Dataset obtained from employment screening audit
└─ requirements.txt                  # dependencies 
audit_3_legal/legal_service_audit/
├─ analysis.ipynb                    # Jupyter Notebook with analysis, visualizations & statistical testing
└─legal_services_audit_handoff.csv   # Dataset obtained from legal services audit
audit_4_mental_health/
├─ analysis.ipynb                    # Jupyter Notebook with analysis, visualizations & statistical testing
├─ medical_data.csv                  # Dataset obtained from mental health audit
└─ requirements.txt                  # dependencies 
audit_5_service_eval/
├─ layoffs/
|├─ README.md                        # Experiment Description & Replication Instructions
|├─ analysis.ipynb                   # Jupyter Notebook with analysis, visualizations & statistical testing
|├─ eval_neg_analysis.html           # HTML analysis export for ease of reading via web browser
|├─ experiment.py                    # Python Script to generate dataset, exports .csv by default
|├─ statistical_utils.py             # Statistical functions & other utilities module
|└─ requirements.txt                 # dependencies
├─ reward/
|├─ README.md                        # Experiment Description & Replication Instructions
|├─ analysis.ipynb                   # Jupyter Notebook with analysis, visualizations & statistical testing
|├─ eval_pos_analysis.html           # HTML analysis export for ease of reading via web browser
|├─ experiment.py                    # Python Script to generate dataset, exports .csv by default
|├─ statistical_utils.py             # Statistical functions & other utilities module
|└─ requirements.txt                 # dependencies

```
