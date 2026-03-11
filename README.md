# LLM Auditing Repository

This repository contains five domain-specific LLM auditing projects. Each domain
uses controlled perturbation designs: the task evidence stays fixed while
demographic or contextual variables change so model behavior can be compared
consistently across prompts, models, and repetitions.

This root README consolidates the domain documentation into one place. It
includes:

- a repository-level overview
- the folder structure for each audit domain
- aggregated setup guidance
- verbatim copies of the domain README files

`data_extraction/` is intentionally excluded from this document and from the
repository contents.

## Repository Overview

### Domains

| Domain Folder | Focus | Main Artifacts | Typical Workflow |
| --- | --- | --- | --- |
| `audit_1_ai_gen_detect/` | AI-generated text detection likelihood scoring | `experiment.py`, `analysis.ipynb`, HTML exports, per-condition requirements | Install deps, set `OPENROUTER_API_KEY`, run experiment, analyze notebook |
| `audit_2_employment_screening/` | Employment screening fairness analysis | `analysis.ipynb`, `results.csv`, requirements | Install deps, launch Jupyter, rerun notebook |
| `audit_3_legal/` | Judge hearing discretion / evaluation audit | `judge_audit/` notebook, cleaned data, per-model experiment folders, requirements | Install deps, set `OPENROUTER_API_KEY`, rerun per-model experiments or analyze notebook |
| `audit_4_mental_health/` | Mental health triage audit analysis | `analysis.ipynb`, `medical_data.csv`, requirements | Install deps, launch Jupyter, rerun notebook |
| `audit_5_service_eval/` | Teacher service evaluation under negative and positive framings | `experiment.py`, `analysis.ipynb`, HTML exports, per-framing requirements | Install deps, set `OPENROUTER_API_KEY`, run experiment, analyze notebook |

### Top-Level Layout

```text
.
├─ audit_1_ai_gen_detect/
│  ├─ ai_gen_false/
│  └─ ai_gen_true/
├─ audit_2_employment_screening/
├─ audit_3_legal/
│  └─ judge_audit/
├─ audit_4_mental_health/
├─ audit_5_service_eval/
│  ├─ layoffs/
│  └─ reward/
└─ misc/
```

## Aggregated Setup

### Install Once For All Domains

Use the root dependency file to install the combined environment for all five
audit domains:

```bash
pip install -r requirements.txt
```

### Runtime Requirements By Domain

- `audit_1_ai_gen_detect/`: `OPENROUTER_API_KEY` is required to rerun
  `ai_gen_false/experiment.py` or `ai_gen_true/experiment.py`.
- `audit_2_employment_screening/`: notebook-only analysis stack; open
  `analysis.ipynb` after installing dependencies.
- `audit_3_legal/`: `OPENROUTER_API_KEY` is required to rerun experiment folders
  under `judge_audit/model_data/`; the cleaned dataset and analysis notebook are
  already included in `judge_audit/`.
- `audit_4_mental_health/`: notebook-only analysis stack; open
  `analysis.ipynb` after installing dependencies.
- `audit_5_service_eval/`: `OPENROUTER_API_KEY` is required to rerun
  `layoffs/experiment.py` or `reward/experiment.py`.

### Shared Commands

```bash
# install the aggregated environment
pip install -r requirements.txt

# set the API key for experiment-based domains
export OPENROUTER_API_KEY="sk-or-..."

# launch notebooks for analysis-oriented domains
jupyter notebook
```

## Domain Details

## `audit_1_ai_gen_detect`

High-level summary: this domain audits AI-generated detection likelihood scoring
under two conditions: text that is actually not AI-generated and text that is
AI-generated. Both subprojects use identical task content while varying student
names and grade levels to test whether demographic markers shift model scoring.

### Folder Structure

```text
audit_1_ai_gen_detect/
├─ README.md
├─ ai_gen_false/
│  ├─ README.md
│  ├─ analysis.ipynb
│  ├─ detec_false_analysis.html
│  ├─ experiment.py
│  └─ requirements.txt
└─ ai_gen_true/
   ├─ README.md
   ├─ analysis.ipynb
   ├─ detec_true_analysis.html
   ├─ experiment.py
   └─ requirements.txt
```

### Setup Notes

- Install the root `requirements.txt`, or the subfolder-specific requirements
  file if you only want one experiment.
- Set `OPENROUTER_API_KEY`.
- Run `python experiment.py` inside `ai_gen_false/` or `ai_gen_true/`.
- Open the matching notebook or HTML export for analysis review.

### Verbatim README Copies

<details>
<summary><code>audit_1_ai_gen_detect/README.md</code></summary>

<pre>
# Education Audits: AI Generated Detection Likelihood Scoring (True & False)

This folder contains the AI Generated Detection Likelihood audits, two controlled perturbation audits where a pre-GPT era piece of writing, and a GPT-written blurb are scrutinized by the models and assigned a "likelihood" score that indicates how likely it is that the pieces were AI generated. 

The short answer response is identical for every call in each audit, and the only difference is the student's name and the grade level (7th grade, 10th grade, college level, graduate level). Gender and ethnicity markers are embedded in each name, and a neutral control variable was also included. We prompted each model ten times, in order to assess the consistency of scoring across the models.  

Each folder has detailed instructions for that particular experiment.

</pre>

</details>

<details>
<summary><code>audit_1_ai_gen_detect/ai_gen_false/README.md</code></summary>

<pre>
# Experiment: AI Generated Detection Likelihood Scoring (False: The writing is NOT AI-generated)

This repository contains the full workflow for Experiment I, a controlled perturbation audit where a pre-GPT era piece of writing is scrutinized by the models and assigned a "likelihood" score that indicates how likely it is that the piece was AI generated. 

The short answer response is identical for every call, and the only difference is the student's name and the grade level (7th grade, 10th grade, college level, graduate level). Gender and ethnicity markers are embedded in each name, and a neutral control variable was also included. We prompted each model ten times, in order to assess the consistency of scoring across the models.  

## Contents

- `requirements.txt` - Necessary python package dependencies
- `experiment.py` - Python script to reproduce the experiment data
- `analysis.ipynb` - Original Experiment Data Analysis
- Output files (generated from `experiment.py`):
    - `results_YYYYMMDD_HHMMSS.db` - Sqlite3 database with full prompt/response logs
    - `results_YYYYMMDD_HHMMSS.csv` - Default output.

### Data: 

The original experiment data is too large for the repository, it can be found [here](https://drive.google.com/file/d/1-LWynxhK8YJwlf8aCTGo4wfYimXHMWMx/view?usp=sharing). 

#### `experiment.py` will generate:
- `results_YYYYMMDD_HHMMSS.db` - sqlite3 db for storing results and managing execution
- `results_YYYYMMDD_HHMMSS.csv` - generated by default after trial is complete

`analysis.ipynb` has outputs for the original experiment run, a new set of data can be generated using the `experiment.py` script. The analysis notebook can be run with the new data by changing the `data/results.csv` filepath in the beginning of the notebook to the resulting `results_YYYYMMDD_HHMMSS.csv` file

To successfully get a new set of data, you need an OpenRouter API key, and enough credits (this experiment was roughly $6 USD, but costs may vary).

---

## Running a new Trial (create new data)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages plus optional dependencies for Excel and Parquet export.

### 2. Set API Keys (Required)

API keys **must** be set as environment variables:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

**Or on Windows Powershell:**
```powershell
$env:OPENROUTER_API_KEY="sk-or-..."
```

The script checks all required keys upfront and will exit if any are missing.

### 3. Run the Experiment

**Basic usage:**
```bash
python experiment.py
```

This will:

- Create a timestamped SQLite database
- Run all prompt variations
- Parse numeric scores
- Export results_*.csv upon completion

---

## Command-Line Options

| Option                 | Default| Description |
|------------------------|--------|------------------------------------------------------------------|
| `--output` / `-o`      | `csv`  | Output format: `csv`, `tsv`, `json`, `jsonl`, `excel`, `parquet` |
| `--concurrent` / `-c`  | `10`   | Number of concurrent API requests |
| `--rate-limit` / `-r`  | `5.0`  | Maximum requests per second |
| `--timeout` / `-t`     | `90`   | Request timeout in seconds |
| `--output-file` / `-f` | Auto   | Custom output filename |
| `--resume`             | Off    | Resume from existing database |
| `--db-file`            | Auto   | Database file to use/resume from |

**Examples:**

```bash
# Fast batch processing
python experiment.py --concurrent 20 --rate-limit 10.0

# Conservative (avoid rate limits)
python experiment.py --concurrent 1 --rate-limit 1.0

# Long-running API calls
python experiment.py --timeout 180

# Resume from previous run
python experiment.py --resume --db-file results_20250128_143022.db

# Export to specific format
python experiment.py --output excel --output-file my_results.xlsx
```

**With concurrency and rate limiting (enabled by default):**
```bash
python experiment.py --concurrent 10 --rate-limit 5.0
```
Concurrency is how many simultaneous/parallel API requests are made at a time.
Rate limits are how many completed requests are made per second, on average.
The script tracks both and lets you set limits independently. The default is 10
concurrent requests, max of 5 requests per second.

This command runs 30 API calls concurrently (in parallel), with an additional
maximum of 10 completed requests per second:

```bash
python experiment.py --concurrent 30 --rate-limit 10.0
```

Or if you are using ollama for a local model on a GPU that cannot handle many
parallel requests, you can set the concurrency to 1 for serial/one-at-a-time mode:

```bash
python experiment.py --concurrent 1 --rate-limit 10.0
```

**With custom output format:**
By default, it saves results to the sqlite3 db file and exports to CSV on completion.
You can do:
```bash
python experiment.py --output excel
python experiment.py --output parquet
```

---

## Features

### Reliability

**Retry Logic:**
- Up to 10 automatic retry attempts with exponential backoff (1s → 30s)
- Smart handling of rate limits (429) and server errors (500/502/503/504)
- Immediate failure on auth errors (401/403) - no wasted retries
- Detailed logging of retry attempts

**SQLite Persistence:**
- All results saved to SQLite database in real-time
- Atomic transactions ensure no data loss
- Query results with standard SQL tools
- Database survives crashes and interruptions

### Resume from Interruptions

Press Ctrl+C at any time. To resume you must specify the db generated:

```bash
python experiment.py --resume --db-file results_20250128_143022.db
```

The script will:
- Skip all previously successful calls
- Retry any previously failed calls
- Continue from where you left off
- Export with the updated results

### Smart Output Schema

**Dynamic columns based on your configuration:**
- Each parameter gets its own column: `param_temperature`, `param_max_tokens`, etc.
- Each variable gets its own column
- Full request/response payloads saved as JSON
- Extracted answers in dedicated column

**Example CSV structure:**
```
config_index | model_display_name | param_temperature | param_max_tokens | variable1 | variable2 | extracted | success | error
```

---

## Troubleshooting

### Rate Limit Errors (429)
```
Retry 3 for GPT-4 after HTTPStatusError: 429 Rate Limit
```
**Fix:** Reduce `--concurrent` or `--rate-limit`, or wait and use `--resume`

### Extraction Failures
```
[WARN] GPT-4: Extraction failed. Tried: choices[0].message.content, data.content
```
**Fix:** Check the API response format in the database (`response` column) and update `extract_paths` in the script

### Network Timeouts
```
Network error GPT-4: TimeoutException
```
**Fix:** Increase `--timeout` or check your network connection

</pre>

</details>

<details>
<summary><code>audit_1_ai_gen_detect/ai_gen_true/README.md</code></summary>

<pre>
# Experiment: AI Generated Detection Likelihood Scoring (True: The writing is AI-generated)

This repository contains the full workflow for a controlled perturbation audit where an AI-generated piece of writing is scrutinized by the models and assigned a "likelihood" score that indicates how likely it is that the piece was AI generated. 

The short answer response is identical for every call, and the only difference is the student's name and the grade level (7th grade, 10th grade, college level, graduate level). Gender and ethnicity markers are embedded in each name, and a neutral control variable was also included. We prompted each model ten times, in order to assess the consistency of scoring across the models.  

## Contents

- `requirements.txt` - Necessary python package dependencies
- `experiment.py` - Python script to reproduce the experiment data
- `analysis.ipynb` - Original Experiment Data Analysis
- Output files (generated from `experiment.py`):
    - `results_YYYYMMDD_HHMMSS.db` - Sqlite3 database with full prompt/response logs
    - `results_YYYYMMDD_HHMMSS.csv` - Default output.

### Data: 

The original experiment data is too large for the repository, it can be found [here](https://drive.google.com/file/d/1nv_DQkXHTj5dk33a_Obajbavas0Get-5/view?usp=sharing). 

#### `experiment.py` will generate:
- `results_YYYYMMDD_HHMMSS.db` - sqlite3 db for storing results and managing execution
- `results_YYYYMMDD_HHMMSS.csv` - generated by default after trial is complete

`analysis.ipynb` has outputs for the original experiment run, a new set of data can be generated using the `experiment.py` script. The analysis notebook can be run with the new data by changing the `data/results.csv` filepath in the beginning of the notebook to the resulting `results_YYYYMMDD_HHMMSS.csv` file

To successfully get a new set of data, you need an OpenRouter API key, and enough credits (this experiment was roughly $6 USD, but costs may vary).

---

## Running a new Trial (create new data)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages plus optional dependencies for Excel and Parquet export.

### 2. Set API Keys (Required)

API keys **must** be set as environment variables:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

**Or on Windows Powershell:**
```powershell
$env:OPENROUTER_API_KEY="sk-or-..."
```

The script checks all required keys upfront and will exit if any are missing.

### 3. Run the Experiment

**Basic usage:**
```bash
python experiment.py
```

This will:

- Create a timestamped SQLite database
- Run all prompt variations
- Parse numeric scores
- Export results_*.csv upon completion

---

## Command-Line Options

| Option                 | Default| Description |
|------------------------|--------|------------------------------------------------------------------|
| `--output` / `-o`      | `csv`  | Output format: `csv`, `tsv`, `json`, `jsonl`, `excel`, `parquet` |
| `--concurrent` / `-c`  | `10`   | Number of concurrent API requests |
| `--rate-limit` / `-r`  | `5.0`  | Maximum requests per second |
| `--timeout` / `-t`     | `90`   | Request timeout in seconds |
| `--output-file` / `-f` | Auto   | Custom output filename |
| `--resume`             | Off    | Resume from existing database |
| `--db-file`            | Auto   | Database file to use/resume from |

**Examples:**

```bash
# Fast batch processing
python experiment.py --concurrent 20 --rate-limit 10.0

# Conservative (avoid rate limits)
python experiment.py --concurrent 1 --rate-limit 1.0

# Long-running API calls
python experiment.py --timeout 180

# Resume from previous run
python experiment.py --resume --db-file results_20250128_143022.db

# Export to specific format
python experiment.py --output excel --output-file my_results.xlsx
```

**With concurrency and rate limiting (enabled by default):**
```bash
python experiment.py --concurrent 10 --rate-limit 5.0
```
Concurrency is how many simultaneous/parallel API requests are made at a time.
Rate limits are how many completed requests are made per second, on average.
The script tracks both and lets you set limits independently. The default is 10
concurrent requests, max of 5 requests per second.

This command runs 30 API calls concurrently (in parallel), with an additional
maximum of 10 completed requests per second:

```bash
python experiment.py --concurrent 30 --rate-limit 10.0
```

Or if you are using ollama for a local model on a GPU that cannot handle many
parallel requests, you can set the concurrency to 1 for serial/one-at-a-time mode:

```bash
python experiment.py --concurrent 1 --rate-limit 10.0
```

**With custom output format:**
By default, it saves results to the sqlite3 db file and exports to CSV on completion.
You can do:
```bash
python experiment.py --output excel
python experiment.py --output parquet
```

---

## Features

### Reliability

**Retry Logic:**
- Up to 10 automatic retry attempts with exponential backoff (1s → 30s)
- Smart handling of rate limits (429) and server errors (500/502/503/504)
- Immediate failure on auth errors (401/403) - no wasted retries
- Detailed logging of retry attempts

**SQLite Persistence:**
- All results saved to SQLite database in real-time
- Atomic transactions ensure no data loss
- Query results with standard SQL tools
- Database survives crashes and interruptions

### Resume from Interruptions

Press Ctrl+C at any time. To resume you must specify the db generated:

```bash
python experiment.py --resume --db-file results_20250128_143022.db
```

The script will:
- Skip all previously successful calls
- Retry any previously failed calls
- Continue from where you left off
- Export with the updated results

### Smart Output Schema

**Dynamic columns based on your configuration:**
- Each parameter gets its own column: `param_temperature`, `param_max_tokens`, etc.
- Each variable gets its own column
- Full request/response payloads saved as JSON
- Extracted answers in dedicated column

**Example CSV structure:**
```
config_index | model_display_name | param_temperature | param_max_tokens | variable1 | variable2 | extracted | success | error
```

---

## Troubleshooting

### Rate Limit Errors (429)
```
Retry 3 for GPT-4 after HTTPStatusError: 429 Rate Limit
```
**Fix:** Reduce `--concurrent` or `--rate-limit`, or wait and use `--resume`

### Extraction Failures
```
[WARN] GPT-4: Extraction failed. Tried: choices[0].message.content, data.content
```
**Fix:** Check the API response format in the database (`response` column) and update `extract_paths` in the script

### Network Timeouts
```
Network error GPT-4: TimeoutException
```
**Fix:** Increase `--timeout` or check your network connection

</pre>

</details>

## `audit_2_employment_screening`

High-level summary: this domain is an analysis-first employment screening audit
focused on how demographic cues may affect resume screening and ranking even
when candidate qualifications are held constant.

### Folder Structure

```text
audit_2_employment_screening/
├─ README.md
├─ analysis.ipynb
├─ requirements.txt
└─ results.csv
```

### Setup Notes

- Install the root `requirements.txt`, or just
  `audit_2_employment_screening/requirements.txt` for this domain.
- Launch `jupyter notebook`.
- Open `analysis.ipynb` and run all cells to reproduce the analysis.

### Verbatim README Copies

<details>
<summary><code>audit_2_employment_screening/README.md</code></summary>

<pre>
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

</pre>

</details>

## `audit_3_legal`

High-level summary: this domain audits judge hearing evaluation and discretion
scoring. The main `judge_audit/` folder contains the cleaned results and
analysis notebook, while `judge_audit/model_data/` contains per-model rerun
folders with their own `experiment.py`, `trial_config.json`, and experiment
requirements.

### Folder Structure

```text
audit_3_legal/
├─ README.md
└─ judge_audit/
   ├─ README.md
   ├─ analysis.ipynb
   ├─ requirements.txt
   ├─ results.csv
   └─ model_data/
      ├─ GPT-4o/
      ├─ claude-3.5/
      ├─ deepseek-chat/
      ├─ gemini-3/
      ├─ gemma-2-27b/
      ├─ gpt-5/
      ├─ gpt-oss-120b/
      ├─ grok-3-mini/
      ├─ llama-4-maverick/
      ├─ nova-micro-v1/
      └─ qwen-max/
```

### Setup Notes

- Install the root `requirements.txt` for the combined environment.
- Set `OPENROUTER_API_KEY` before rerunning experiments.
- Use `judge_audit/results.csv` and `judge_audit/analysis.ipynb` for the cleaned
  analysis workflow.
- Use folders under `judge_audit/model_data/` when you want to rerun a specific
  model configuration.

### Verbatim README Copies

<details>
<summary><code>audit_3_legal/README.md</code></summary>

<pre>
# Legal Audits: Judge Hearing Evaluation 

---

This folder contains: Judge Evaluation 

### 1. Judge Hearing Evaluation
This repository contains the full workflow for a controlled perturbation audit examing whether large language models (LLMs) assign different ordinal scores to the same judge when only the judge's name (a proxy for gender and ethnicity), age, and immigration status is changed.

The LLM is provided an identical prompt for every call, and the only difference is the person's name, age, and immigration status. Gender and ethnicity markers are embedded in each name, and a neutral control variable was also included.

---

</pre>

</details>

<details>
<summary><code>audit_3_legal/judge_audit/README.md</code></summary>

<pre>
# Experiment: Judge Hearing Discretion Evaluation  

This repository contains the full workflow for Experiemnt III, a controlled perturbation audit examing whether large language models (LLMs) assign different ordinal scores to the same judge when only the judge's name (a proxy for gender and ethnicity), age, and immigration status is changed.

---

## Contents

```
judge_audit/
├─ analysis.ipynb           # Jupyter Notebook with analysis, visualizations & statistical testing
├─ model_data/
   ├─ 11 model perturbation files/
       ├─ data/ (generated from experiemnt.py)
           ├─ results.excel
           ├─ results.jsonl
           ├─ results.csv
       ├─ experiment.py
       ├─ README.md
       ├─ requirements.txt
       ├─ trial_config.json
├─ results.csv              # Dataset obtained and cleaned from judge evaluation audit
├─ requirements.txt         # Necessary python package dependencies
```

---

## Running a New Trial

Only one key is required.

### 1. Set OpenRouter API Key

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

Windows PowerShell:

```powershell
$env:OPENROUTER_API_KEY="sk-or-..."
```

This is the only key needed. The script exclusively uses OpenRouter to access models such as `gpt-oss:20b`, `qwen3:14b`, and others.

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run the Experiment

```bash
python experiment.py
```

This will:

- Create a timestamped SQLite database  
- Run all prompt variations  
- Parse numeric scores  
- Export `results_*.csv` upon completion  

---

## Command-Line Options

| Option                 | Default| Description |
|------------------------|--------|------------------------------------------------------------------|
| `--output` / `-o`      | `csv`  | Output format: `csv`, `tsv`, `json`, `jsonl`, `excel`, `parquet` |
| `--concurrent` / `-c`  | `10`   | Number of concurrent API requests |
| `--rate-limit` / `-r`  | `5.0`  | Maximum requests per second |
| `--timeout` / `-t`     | `90`   | Request timeout in seconds |
| `--output-file` / `-f` | Auto   | Custom output filename |
| `--resume`             | Off    | Resume from existing database |
| `--db-file`            | Auto   | Database file to use/resume from |

**Examples:**

```bash
# Fast batch processing
python experiment.py --concurrent 20 --rate-limit 10.0

# Conservative (avoid rate limits)
python experiment.py --concurrent 1 --rate-limit 1.0

# Long-running API calls
python experiment.py --timeout 180

# Resume from previous run
python experiment.py --resume --db-file results_20250128_143022.db

# Export to specific format
python experiment.py --output excel --output-file my_results.xlsx
```

**With concurrency and rate limiting (enabled by default):**
```bash
python experiment.py --concurrent 10 --rate-limit 5.0
```
Concurrency is how many simultaneous/parallel API requests are made at a time.
Rate limits are how many completed requests are made per second, on average.
The script tracks both and lets you set limits independently. The default is 10
concurrent requests, max of 5 requests per second.

This command runs 30 API calls concurrently (in parallel), with an additional
maximum of 10 completed requests per second:

```bash
python experiment.py --concurrent 30 --rate-limit 10.0
```

Or if you are using ollama for a local model on a GPU that cannot handle many
parallel requests, you can set the concurrency to 1 for serial/one-at-a-time mode:

```bash
python experiment.py --concurrent 1 --rate-limit 10.0
```

**With custom output format:**
By default, it saves results to the sqlite3 db file and exports to CSV on completion.
You can do:
```bash
python experiment.py --output excel
python experiment.py --output parquet
```

---

## Features

### Reliability

**Retry Logic:**
- Up to 10 automatic retry attempts with exponential backoff (1s → 30s)
- Smart handling of rate limits (429) and server errors (500/502/503/504)
- Immediate failure on auth errors (401/403) - no wasted retries
- Detailed logging of retry attempts

**SQLite Persistence:**
- All results saved to SQLite database in real-time
- Atomic transactions ensure no data loss
- Query results with standard SQL tools
- Database survives crashes and interruptions

### Resume from Interruptions

Press Ctrl+C at any time. To resume you must specify the db generated:

```bash
python experiment.py --resume --db-file results_20250128_143022.db
```

The script will:
- Skip all previously successful calls
- Retry any previously failed calls
- Continue from where you left off
- Export with the updated results

### Smart Output Schema

**Dynamic columns based on your configuration:**
- Each parameter gets its own column: `param_temperature`, `param_max_tokens`, etc.
- Each variable gets its own column
- Full request/response payloads saved as JSON
- Extracted answers in dedicated column

**Example CSV structure:**
```
config_index | model_display_name | param_temperature | param_max_tokens | variable1 | variable2 | extracted | success | error
```

---

## Troubleshooting

### Rate Limit Errors (429)
```
Retry 3 for GPT-4 after HTTPStatusError: 429 Rate Limit
```
**Fix:** Reduce `--concurrent` or `--rate-limit`, or wait and use `--resume`

### Extraction Failures
```
[WARN] GPT-4: Extraction failed. Tried: choices[0].message.content, data.content
```
**Fix:** Check the API response format in the database (`response` column) and update `extract_paths` in the script

### Network Timeouts
```
Network error GPT-4: TimeoutException
```
**Fix:** Increase `--timeout` or check your network connection

</pre>

</details>

## `audit_4_mental_health`

High-level summary: this domain analyzes mental health triage outputs under
controlled demographic perturbations, focusing on whether triage and urgency
recommendations change across groups.

### Folder Structure

```text
audit_4_mental_health/
├─ README.md
├─ analysis.ipynb
├─ medical_data.csv
└─ requirements.txt
```

### Setup Notes

- Install the root `requirements.txt`, or just
  `audit_4_mental_health/requirements.txt` for this domain.
- Launch `jupyter notebook`.
- Open `analysis.ipynb` and run all cells to reproduce the statistical results.

### Verbatim README Copies

<details>
<summary><code>audit_4_mental_health/README.md</code></summary>

<pre>
## Mental Health Triage Audit
This folder contains documents relating to the data analysis and statistical testing of the mental health audit. AI is largely being used to remove the administrative burden from mental health triaging. Through stakeholders interviews, it is clear there is concern regarding how a biased system can directly shape how individuals are advised to respond to psychological distress and their access to care. For individuals from marginalized communities who may already experience barriers to care and historical mistrust of medical systems, differential AI assessments risk compounding existing inequities. This page documents our mental health audit analysis.

### Structure

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

</pre>

</details>

## `audit_5_service_eval`

High-level summary: this domain audits teacher service evaluation in two
decision framings: negative framing (`layoffs/`) and positive framing
(`reward/`). The quasi-resumes are held constant while names, majors, and
language attributes vary.

### Folder Structure

```text
audit_5_service_eval/
├─ README.md
├─ layoffs/
│  ├─ README.md
│  ├─ analysis.ipynb
│  ├─ eval_neg_analysis.html
│  ├─ experiment.py
│  ├─ requirements.txt
│  └─ statistical_utils.py
└─ reward/
   ├─ README.md
   ├─ analysis.ipynb
   ├─ eval_pos_analysis.html
   ├─ experiment.py
   ├─ requirements.txt
   └─ statistical_utils.py
```

### Setup Notes

- Install the root `requirements.txt`, or the subfolder-specific requirements
  file if you only want one framing.
- Set `OPENROUTER_API_KEY`.
- Run `python experiment.py` inside `layoffs/` or `reward/`.
- Open the matching notebook or HTML export for analysis review.

### Verbatim README Copies

<details>
<summary><code>audit_5_service_eval/README.md</code></summary>

<pre>
# Education Audits: How AI Evaluates Teacher Service (Positive & Negative Framings)

This folder contains two different contexts for controlled perturbation audits where we have different LLMs act as administrative assistant tools for a school district. In these audits, we ask the AI tool to evaluate teachers based on their duties and activities with the purpose of either laying off those with lower scores, or rewarding those with higher scores. 

The quasi-resumes are identical for every call, with the only difference being the teacher's name, their degree major, and an additional foreign language they use to facilitate communication with parents.

Each folder has detailed instructions for that particular experiment.

</pre>


</details>

<details>
<summary><code>audit_5_service_eval/layoffs/README.md</code></summary>


<pre>
# Experiment: How AI Evaluates Teacher Service (Negative)

Trial Name: Teacher Service Evaluations (layoffs)

This folder contains the full workflow for a controlled perturbation audit where we have different LLMs act as administrative assistant tools for a school district. In this variant, we ask the AI tool to evaluate teachers based on their duties and activities with the purpose of laying off those with lower scores. 

The quasi-resumes are identical for every call, with the only difference being the teacher's name, their degree major, and an additional foreign language they use to facilitate communication with parents.

## Contents

- `requirements.txt`                - Necessary python package dependencies
- `experiment.py`                   - Python script to reproduce the experiment data
- `statistical_utils.py`            - Statistical functions & other utilities module
- `analysis.ipynb`                  - Original Experiment Data Analysis
- `eval_neg_analysis.html`          - HTML export for analysis ease of viewing via browser
- Output files (generated from `experiment.py`):
    - `results_YYYYMMDD_HHMMSS.db`  - Sqlite3 database with full prompt/response logs
    - `results_YYYYMMDD_HHMMSS.csv` - Default output.

### Data: 

The original experiment data is too large for the repository, it can be found [here](https://drive.google.com/file/d/1I0gz8pcd52Sy01CbYwX58oQDFLNbyqbT/view?usp=sharing). 

#### `experiment.py` will generate:
- `results_YYYYMMDD_HHMMSS.db` - sqlite3 db for storing results and managing execution
- `results_YYYYMMDD_HHMMSS.csv` - generated by default after trial is complete

`analysis.ipynb` has outputs for the original experiment run, a new set of data can be generated using the `experiment.py` script. The analysis notebook can be run with the new data by changing the `data/results.csv` filepath in the beginning of the notebook to the resulting `results_YYYYMMDD_HHMMSS.csv` file

To successfully get a new set of data, you need an OpenRouter API key, and enough credits (this experiment was roughly $16 USD, but costs may vary).

---

## Running a new Trial (create new data)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages plus optional dependencies for Excel and Parquet export.

### 2. Set API Keys (Required)

API keys **must** be set as environment variables:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

**Or on Windows Powershell:**
```powershell
$env:OPENROUTER_API_KEY="sk-or-..."
```

The script checks all required keys upfront and will exit if any are missing.

### 3. Run the Experiment

**Basic usage:**
```bash
python experiment.py
```

This will:

- Create a timestamped SQLite database
- Run all prompt variations
- Parse numeric scores
- Export results_*.csv upon completion

---

## Command-Line Options

| Option                 | Default| Description |
|------------------------|--------|------------------------------------------------------------------|
| `--output` / `-o`      | `csv`  | Output format: `csv`, `tsv`, `json`, `jsonl`, `excel`, `parquet` |
| `--concurrent` / `-c`  | `10`   | Number of concurrent API requests |
| `--rate-limit` / `-r`  | `5.0`  | Maximum requests per second |
| `--timeout` / `-t`     | `90`   | Request timeout in seconds |
| `--output-file` / `-f` | Auto   | Custom output filename |
| `--resume`             | Off    | Resume from existing database |
| `--db-file`            | Auto   | Database file to use/resume from |

**Examples:**

```bash
# Fast batch processing
python experiment.py --concurrent 20 --rate-limit 10.0

# Conservative (avoid rate limits)
python experiment.py --concurrent 1 --rate-limit 1.0

# Long-running API calls
python experiment.py --timeout 180

# Resume from previous run
python experiment.py --resume --db-file results_20250128_143022.db

# Export to specific format
python experiment.py --output excel --output-file my_results.xlsx
```

**With concurrency and rate limiting (enabled by default):**
```bash
python experiment.py --concurrent 10 --rate-limit 5.0
```
Concurrency is how many simultaneous/parallel API requests are made at a time.
Rate limits are how many completed requests are made per second, on average.
The script tracks both and lets you set limits independently. The default is 10
concurrent requests, max of 5 requests per second.

This command runs 30 API calls concurrently (in parallel), with an additional
maximum of 10 completed requests per second:

```bash
python experiment.py --concurrent 30 --rate-limit 10.0
```

Or if you are using ollama for a local model on a GPU that cannot handle many
parallel requests, you can set the concurrency to 1 for serial/one-at-a-time mode:

```bash
python experiment.py --concurrent 1 --rate-limit 10.0
```

**With custom output format:**
By default, it saves results to the sqlite3 db file and exports to CSV on completion.
You can do:
```bash
python experiment.py --output excel
python experiment.py --output parquet
```

---

## Features

### Reliability

**Retry Logic:**
- Up to 10 automatic retry attempts with exponential backoff (1s → 30s)
- Smart handling of rate limits (429) and server errors (500/502/503/504)
- Immediate failure on auth errors (401/403) - no wasted retries
- Detailed logging of retry attempts

**SQLite Persistence:**
- All results saved to SQLite database in real-time
- Atomic transactions ensure no data loss
- Query results with standard SQL tools
- Database survives crashes and interruptions

### Resume from Interruptions

Press Ctrl+C at any time. To resume you must specify the db generated:

```bash
python experiment.py --resume --db-file results_20250128_143022.db
```

The script will:
- Skip all previously successful calls
- Retry any previously failed calls
- Continue from where you left off
- Export with the updated results

### Smart Output Schema

**Dynamic columns based on your configuration:**
- Each parameter gets its own column: `param_temperature`, `param_max_tokens`, etc.
- Each variable gets its own column
- Full request/response payloads saved as JSON
- Extracted answers in dedicated column

**Example CSV structure:**
```
config_index | model_display_name | param_temperature | param_max_tokens | variable1 | variable2 | extracted | success | error
```

---

## Troubleshooting

### Rate Limit Errors (429)
```
Retry 3 for GPT-4 after HTTPStatusError: 429 Rate Limit
```
**Fix:** Reduce `--concurrent` or `--rate-limit`, or wait and use `--resume`

### Extraction Failures
```
[WARN] GPT-4: Extraction failed. Tried: choices[0].message.content, data.content
```
**Fix:** Check the API response format in the database (`response` column) and update `extract_paths` in the script

### Network Timeouts
```
Network error GPT-4: TimeoutException
```
**Fix:** Increase `--timeout` or check your network connection

</pre>

</details>

<details>
<summary><code>audit_5_service_eval/reward/README.md</code></summary>


<pre>
# Experiment: How AI Evaluates Teacher Service (Positive)

Trial Name: Teacher Service Evaluations (reward)

This folder contains the full workflow for a controlled perturbation audit where we have different LLMs act as administrative assistant tools for a school district. In this variant, we ask the AI tool to evaluate teachers based on their duties and activities with the purpose of rewarding those with high scores. 

The quasi-resumes are identical for every call, with the only difference being the teacher's name, their degree major, and an additional foreign language they use to facilitate communication with parents.

## Contents

- `requirements.txt`                - Necessary python package dependencies
- `experiment.py`                   - Python script to reproduce the experiment data
- `statistical_utils.py`            - Statistical functions & other utilities module
- `analysis.ipynb`                  - Original Experiment Data Analysis
- `eval_pos_analysis.html`          - HTML export for analysis ease of viewing via browser
- Output files (generated from `experiment.py`):
    - `results_YYYYMMDD_HHMMSS.db`  - Sqlite3 database with full prompt/response logs
    - `results_YYYYMMDD_HHMMSS.csv` - Default output.

### Data: 

The original experiment data is too large for the repository, it can be found [here](https://drive.google.com/file/d/1LFkcwMf2Droz21QlLFtRXu5Auqmv51VA/view?usp=sharing). 

#### `experiment.py` will generate:
- `results_YYYYMMDD_HHMMSS.db` - sqlite3 db for storing results and managing execution
- `results_YYYYMMDD_HHMMSS.csv` - generated by default after trial is complete

`analysis.ipynb` has outputs for the original experiment run, a new set of data can be generated using the `experiment.py` script. The analysis notebook can be run with the new data by changing the `data/results.csv` filepath in the beginning of the notebook to the resulting `results_YYYYMMDD_HHMMSS.csv` file

To successfully get a new set of data, you need an OpenRouter API key, and enough credits (this experiment was roughly $16 USD, but costs may vary).

---

## Running a new Trial (create new data)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages plus optional dependencies for Excel and Parquet export.

### 2. Set API Keys (Required)

API keys **must** be set as environment variables:

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

**Or on Windows Powershell:**
```powershell
$env:OPENROUTER_API_KEY="sk-or-..."
```

The script checks all required keys upfront and will exit if any are missing.

### 3. Run the Experiment

**Basic usage:**
```bash
python experiment.py
```

This will:

- Create a timestamped SQLite database
- Run all prompt variations
- Parse numeric scores
- Export results_*.csv upon completion

---

## Command-Line Options

| Option                 | Default| Description |
|------------------------|--------|------------------------------------------------------------------|
| `--output` / `-o`      | `csv`  | Output format: `csv`, `tsv`, `json`, `jsonl`, `excel`, `parquet` |
| `--concurrent` / `-c`  | `10`   | Number of concurrent API requests |
| `--rate-limit` / `-r`  | `5.0`  | Maximum requests per second |
| `--timeout` / `-t`     | `90`   | Request timeout in seconds |
| `--output-file` / `-f` | Auto   | Custom output filename |
| `--resume`             | Off    | Resume from existing database |
| `--db-file`            | Auto   | Database file to use/resume from |

**Examples:**

```bash
# Fast batch processing
python experiment.py --concurrent 20 --rate-limit 10.0

# Conservative (avoid rate limits)
python experiment.py --concurrent 1 --rate-limit 1.0

# Long-running API calls
python experiment.py --timeout 180

# Resume from previous run
python experiment.py --resume --db-file results_20250128_143022.db

# Export to specific format
python experiment.py --output excel --output-file my_results.xlsx
```

**With concurrency and rate limiting (enabled by default):**
```bash
python experiment.py --concurrent 10 --rate-limit 5.0
```
Concurrency is how many simultaneous/parallel API requests are made at a time.
Rate limits are how many completed requests are made per second, on average.
The script tracks both and lets you set limits independently. The default is 10
concurrent requests, max of 5 requests per second.

This command runs 30 API calls concurrently (in parallel), with an additional
maximum of 10 completed requests per second:

```bash
python experiment.py --concurrent 30 --rate-limit 10.0
```

Or if you are using ollama for a local model on a GPU that cannot handle many
parallel requests, you can set the concurrency to 1 for serial/one-at-a-time mode:

```bash
python experiment.py --concurrent 1 --rate-limit 10.0
```

**With custom output format:**
By default, it saves results to the sqlite3 db file and exports to CSV on completion.
You can do:
```bash
python experiment.py --output excel
python experiment.py --output parquet
```

---

## Features

### Reliability

**Retry Logic:**
- Up to 10 automatic retry attempts with exponential backoff (1s → 30s)
- Smart handling of rate limits (429) and server errors (500/502/503/504)
- Immediate failure on auth errors (401/403) - no wasted retries
- Detailed logging of retry attempts

**SQLite Persistence:**
- All results saved to SQLite database in real-time
- Atomic transactions ensure no data loss
- Query results with standard SQL tools
- Database survives crashes and interruptions

### Resume from Interruptions

Press Ctrl+C at any time. To resume you must specify the db generated:

```bash
python experiment.py --resume --db-file results_20250128_143022.db
```

The script will:
- Skip all previously successful calls
- Retry any previously failed calls
- Continue from where you left off
- Export with the updated results

### Smart Output Schema

**Dynamic columns based on your configuration:**
- Each parameter gets its own column: `param_temperature`, `param_max_tokens`, etc.
- Each variable gets its own column
- Full request/response payloads saved as JSON
- Extracted answers in dedicated column

**Example CSV structure:**
```
config_index | model_display_name | param_temperature | param_max_tokens | variable1 | variable2 | extracted | success | error
```

---

## Troubleshooting

### Rate Limit Errors (429)
```
Retry 3 for GPT-4 after HTTPStatusError: 429 Rate Limit
```
**Fix:** Reduce `--concurrent` or `--rate-limit`, or wait and use `--resume`

### Extraction Failures
```
[WARN] GPT-4: Extraction failed. Tried: choices[0].message.content, data.content
```
**Fix:** Check the API response format in the database (`response` column) and update `extract_paths` in the script

### Network Timeouts
```
Network error GPT-4: TimeoutException
```
**Fix:** Increase `--timeout` or check your network connection

</pre>

</details>
