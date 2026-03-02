## Mental Health Triage Audit
This folder contains documents relating to the data analysis and statistical testing of the mental health audit. AI is largely being used to remove the administrative burden from mental health triaging. Through stakeholders interviews, it is clear there is concern regarding how a biased system can directly shape how individuals are advised to respond to psychological distress and their access to care. For individuals from marginalized communities who may already experience barriers to care and historical mistrust of medical systems, differential AI assessments risk compounding existing inequities. This page documents our mental health audit analysis.

### Structure
```
mental_health/
├─ analysis.ipynb           # Jupyter Notebook with analysis, visualizations & statistical testing
├─ medical_data.csv         # Dataset obtained from mental health audit
├─ requirements.txt         # dependencies 
```

### Data Generation

The original dataset (`medical_data.csv`) was generated through an automated mental health audit experiment that queried LLMs using systematically varied demographic cues while holding qualifications constant. The prompt and variables used can be found in the Introduction section of analysis.ipynb. 

The script used to generate the dataset (`experiment.py`) is no longer available. However, this repository includes:

- The complete raw dataset used in the study (`medical_data.csv`)
- The full statistical analysis and visualizations (`analysis.ipynb`)
- The prompt template and experimental variables (documented in `analysis.ipynb` Introduction section)

Because the raw dataset is included, all statistical results in the paper are fully reproducible from the provided files.

### Reproducing the Analysis

1. Clone the repository
2. Install dependencies:
```
pip install -r requirements.txt
```
4. Launch Jupyter Notebook:
```
jupyter notebook
```
5. Open `analysis.ipynb`
6. Run all cells from top to bottom

Running all cells will regenerate all statistical tests, tables, and visualizations reported in this project.
