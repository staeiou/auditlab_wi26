# Experiment: Judge Hearing Discretion Evaluation  

This repository contains the full workflow for Experiemnt III, a controlled perturbation audit examing whether large language models (LLMs) assign different ordinal scores to the same judge when only the judge's name (a proxy for gender and ethnicity), age, and immigration status is changed.

---

## Contents

```
judge_audit/
├─ analysis.ipynb           # Jupyter Notebook with analysis, visualizations & statistical testing
├─ data
   ├─ 11 model perturbation files
       ├─ data (generated from experiemnt.py)
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
