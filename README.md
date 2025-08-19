
# MLE-STAR-Open (Unofficial)

A Google-free, local-friendly reimplementation of the MLE-STAR multi‑agent ML engineering pipeline.  
Uses OpenAI‑compatible LLM APIs (OpenRouter; planned local Ollama) and DuckDuckGo search.  
Not affiliated with the original authors or Google.

---

## Features

- Multi-agent stages: initialization → refinement → ensembling → submission
- OpenRouter LLM backend (free-tier; rate limits apply)
- Planned local LLM (Ollama) integration
- DuckDuckGo search (no API key)
- Automated basic leakage / data usage checks
- Minimal runner for low token usage
- Kaggle submission formatting helper

---

## Architecture

![Machine-Learning-Engineering](machine-learning-engineering-architecture.svg)

---

<!-- markdownlint-disable MD046 -->
## Installation

1. Prerequisites

        - Python 3.12+
        - Git
        - (Optional) Poetry

1. Clone & env

```powershell
git clone https://github.com/<yourname>/mle-star-open.git
cd mle-star-open
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

1. Configure `.env`

```env
OPENROUTER_API_KEY=sk-...
LLM_MODEL=meta-llama/llama-3.1-8b-instruct:free
# Optional overrides:
# MAX_AGENT_STEPS=4
# TEMPERATURE=0.2
```

See `machine_learning_engineering/shared_libraries/config.py` (`DefaultConfig`) for all keys.

---

## Task Layout

Create a task directory (default path expected under `machine_learning_engineering/tasks/`):

```text
machine_learning_engineering/tasks/
        california-housing-prices/
                task_description.txt
                train.csv
                test.csv            # if producing submission
```

Minimal `task_description.txt` example:

```text
task_name: california-housing-prices
target: median_house_value
id_column: id
metric: rmse
```

Ensure `train.csv` includes the target column; `test.csv` omits it (plus an `id` column if required for submission).

---

## Running

Full pipeline (all agents):

```powershell
python .\scripts\run_pipeline.py --task-name california-housing-prices
```

Minimal (fewer LLM calls):

```powershell
python .\scripts\run_task.py --task-dir .\machine_learning_engineering\tasks\california-housing-prices
```

Create Kaggle-style submission (expects prediction file in workspace run folder):

```powershell
python .\scripts\make_submission.py --output-dir .\machine_learning_engineering\workspace\california-housing-prices\1\output
```

Example outputs:

```text
machine_learning_engineering/workspace/<task>/<run_id>/
        init/
        refine/
        ensemble/
        output/predictions.csv
        logs/
```

---

## Configuration Tweaks

Edit `config.py` or set environment variables (env vars override defaults). Examples:

```powershell
$env:MAX_AGENT_STEPS=3
$env:LLM_MODEL="meta-llama/llama-3.1-8b-instruct:free"
```

---

## Testing

```powershell
pytest
```
Scenario and pipeline tests live under `eval/` and `tests/`.

---

## Limitations

- Not sized for official MLE-Bench specs (36 vCPUs / 440GB RAM / A10 24GB); local experimentation focus.
- OpenRouter free-tier ~50 requests/day (subject to change).
- Early-stage (planned) Ollama integration.
- Test suite may lag new features.

---

## FAQ

**Kaggle error: missing id column?** Ensure `id_column` in `task_description.txt` and that CSVs include it, or run `make_submission.py` to add one.  
**Hit rate limits?** Wait for reset, reduce steps (`MAX_AGENT_STEPS`), use minimal runner, or upgrade plan.  
**Force CPU?** Leave `CUDA_VISIBLE_DEVICES` empty when invoking Python.  
**Change model?** Set `LLM_MODEL` in `.env` or env var before running.

---

## Citation

Original paper: MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement  
[https://arxiv.org/abs/2506.15692](https://arxiv.org/abs/2506.15692)

---

## License

MIT (see LICENSE). Project is independent; no affiliation with the original authors or Google.
