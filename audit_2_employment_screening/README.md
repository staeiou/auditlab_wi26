## Employment Screening Audit
This folder contains documents relating to the data analysis and statistical testing of the employment screening audit. Hiring teams increasingly use AI tools for resume screening and candidate ranking. While these tools promise efficiency, stakeholders (job seekers and recruiters) worry that demographic cues - such as perceived race/ethnicity, gender, or disability status - could influence screening decisions even when qualifications are identical. This page documents our employment screening-domain analysis and highlights when demographic effects appear negligible in aggregate yet substantial at the individual-model level.

### Structure
```
employment_screening/
├─ analysis.ipynb           # Jupyter Notebook with analysis, visualizations & statistical testing
├─ results.csv              # Dataset obtained from employment screening audit
├─ requirements.txt         # dependencies 
```

### Data Generation

The original dataset (`results.csv`) was generated through an automated employment screening audit experiment that queried LLMs using systematically varied demographic cues while holding qualifications constant. The prompt and variables used can be found in the Introduction section of analysis.ipynb. 

The script used to generate the dataset (`experiment.py`) is no longer available. However, this repository includes:

- The complete raw dataset used in the study (`results.csv`)
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


