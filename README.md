
# MLE-STAR-Open (Unofficial)

**Unofficial, Google-free reimplementation of MLE-STAR:**  
A lightweight, local-friendly multi-agent ML engineering pipeline inspired by the MLE-STAR paper.  
Uses OpenAI-compatible LLMs (OpenRouter, with Ollama support in progress) and DuckDuckGo search.  
Not affiliated with the original authors or Google.

---

## What is this?

This project is a reimagined, open-source version of MLE-STAR, designed for:
- **Tabular ML tasks** (classification/regression)
- **Automated code generation, debugging, refinement, ensembling, and submission**
- **No Google SDKs or paid cloud required**: all LLM calls go through OpenAI-compatible APIs (OpenRouter, local Ollama coming soon)
- **Free web search** via DuckDuckGo

**Why?**  
The original MLE-STAR requires Google Vertex AI and GenAI, which are paid and require significant cloud resources.  
This repo is for researchers, students, and hobbyists who want to experiment with MLE-STAR-style agents on local or free-tier compute.

---

## Key Features

- Multi-agent pipeline: initialization, refinement, ensembling, submission
- LLM backend: OpenRouter (free-tier, rate-limited); Ollama (local, in progress)
- Web search: DuckDuckGo (no API key needed)
- Debugging, data leakage, and data usage checkers
- Minimal runner for low-token, quick experiments
- Helper scripts for Kaggle submission formatting

---

## Agent Architecture

![Machine-Learning-Engineering](machine-learning-engineering-architecture.svg)

---

## Setup and Installation

1.  **Prerequisites**
                - Python 3.12+
                - Git
                - (Optional) Poetry (for advanced dependency management)
2.  **Install**
                ```powershell
                git clone https://github.com/<yourname>/mle-star-free.git
                cd mle-star-free
                python -m venv .venv
                . .\.venv\Scripts\Activate.ps1
                python -m pip install --upgrade pip
                pip install -r requirements.txt
                ```
3.  **Configure**
                - Copy `.env.example` to `.env` and add your OpenRouter API key.
                - Set `LLM_MODEL` to a free-tier model (e.g., `meta-llama/llama-3.1-8b-instruct:free`).

---

## Running the Agent

**Prepare your task**
- Create a folder under `tasks` with the name of your task.
- Add a `task_description.txt` and your data files.

**Run the pipeline**
```powershell
python .\scripts\run_pipeline.py --task-name california-housing-prices
```
Or, for a minimal run (fewer LLM calls):
```powershell
python .\scripts\run_task.py --task-dir .\machine_learning_engineering\tasks\california-housing-prices
```

**Format submission for Kaggle**
```powershell
python .\scripts\make_submission.py --output-dir .\machine_learning_engineering\workspace\california-housing-prices\1\output
```

---

## Web Interface (Experimental/Future)

The original MLE-STAR supports a web UI via `adk web`.  
This repo does not yet have a web interface, but the architecture is compatible and a web UI is planned for a future release.

---

## Running Tests

```powershell
pytest
```
- Scenario and pipeline tests are in `eval` and `tests`.

---

## Limitations

- **Compute**: The official MLE-Bench requires 36 vCPUs, 440GB RAM, and a 24GB A10 GPU.  
        This repo is tested on much smaller hardware (laptop/desktop).  
        *You can run the pipeline, but not at MLE-Bench scale or speed.*
- **LLM backend**: OpenRouter free-tier is rate-limited (50 requests/day).  
        Ollama/local LLM support is in progress.
- **Test coverage**: Pytest and scenario tests are included, but may need updates as the codebase evolves.
- **Submission format**: Ensure your `task_description.txt` matches the required Kaggle format (see FAQ).

---

## FAQ

**Q: Why does my submission fail on Kaggle for missing `id`?**  
A: Update your `task_description.txt` to specify the correct submission format (e.g., `id,median_house_value`).  
Or use `make_submission.py` to add an `id` column to your predictions.

**Q: What if I hit OpenRouter rate limits?**  
A: Wait for the daily reset, add credits, or use the minimal runner to reduce LLM calls.  
Ollama/local LLM support is coming.

**Q: Can I run the full MLE-Bench?**  
A: Not on typical local hardware. This repo is for experimentation and research, not leaderboard competition.

---

## Appendix: Required Config Parameters

See the `DefaultConfig` dataclass in `machine_learning_engineering/shared_libraries/config.py` for all options.

---

## Citation

If you use this repo, please cite the original MLE-STAR paper:

> [MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement](https://arxiv.org/abs/2506.15692)

---

## License

MIT License (see LICENSE).  
This project is independent and not affiliated with the MLE-STAR authors or Google.

# MLE-STAR-Free (Unofficial)

**Unofficial, Google-free reimplementation of MLE-STAR:**  
A lightweight, local-friendly multi-agent ML engineering pipeline inspired by the MLE-STAR paper.  
Uses OpenAI-compatible LLMs (OpenRouter, with Ollama support in progress) and DuckDuckGo search.  
Not affiliated with the original authors or Google.

---

## What is this?

This project is a reimagined, open-source version of MLE-STAR, designed for:
- **Tabular ML tasks** (classification/regression)
- **Automated code generation, debugging, refinement, ensembling, and submission**
- **No Google SDKs or paid cloud required**: all LLM calls go through OpenAI-compatible APIs (OpenRouter, local Ollama coming soon)
- **Free web search** via DuckDuckGo

**Why?**  
The original MLE-STAR requires Google Vertex AI and GenAI, which are paid and require significant cloud resources.  
This repo is for researchers, students, and hobbyists who want to experiment with MLE-STAR-style agents on local or free-tier compute.

---

## Key Features

- Multi-agent pipeline: initialization, refinement, ensembling, submission
- LLM backend: OpenRouter (free-tier, rate-limited); Ollama (local, in progress)
- Web search: DuckDuckGo (no API key needed)
- Debugging, data leakage, and data usage checkers
- Minimal runner for low-token, quick experiments
- Helper scripts for Kaggle submission formatting

---

## Agent Architecture

<img src="machine-learning-engineering-architecture.svg" alt="Machine-Learning-Engineering" width="800"/>

---

## Setup and Installation

1.  **Prerequisites**
        - Python 3.12+
        - Git
        - (Optional) Poetry (for advanced dependency management)
2.  **Install**
        ```powershell
        git clone https://github.com/<yourname>/mle-star-free.git
        cd mle-star-free
        python -m venv .venv
        . .\.venv\Scripts\Activate.ps1
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        ```
3.  **Configure**
        - Copy `.env.example` to `.env` and add your OpenRouter API key.
        - Set `LLM_MODEL` to a free-tier model (e.g., `meta-llama/llama-3.1-8b-instruct:free`).

---

## Running the Agent

**Prepare your task**
- Create a folder under `tasks` with the name of your task.
- Add a `task_description.txt` and your data files.

**Run the pipeline**
```powershell
python .\scripts\run_pipeline.py --task-name california-housing-prices
```
Or, for a minimal run (fewer LLM calls):
```powershell
python .\scripts\run_task.py --task-dir .\machine_learning_engineering\tasks\california-housing-prices
```

**Format submission for Kaggle**
```powershell
python .\scripts\make_submission.py --output-dir .\machine_learning_engineering\workspace\california-housing-prices\1\output
```

---

## Web Interface (Experimental/Future)

The original MLE-STAR supports a web UI via `adk web`.  
This repo does not yet have a web interface, but the architecture is compatible and a web UI is planned for a future release.

---

## Running Tests

```powershell
pytest
```
- Scenario and pipeline tests are in `eval` and `tests`.

---

## Limitations

- **Compute**: The official MLE-Bench requires 36 vCPUs, 440GB RAM, and a 24GB A10 GPU.  
    This repo is tested on much smaller hardware (laptop/desktop).  
    *You can run the pipeline, but not at MLE-Bench scale or speed.*
- **LLM backend**: OpenRouter free-tier is rate-limited (50 requests/day).  
    Ollama/local LLM support is in progress.
- **Test coverage**: Pytest and scenario tests are included, but may need updates as the codebase evolves.
- **Submission format**: Ensure your `task_description.txt` matches the required Kaggle format (see FAQ).

---

## FAQ

**Q: Why does my submission fail on Kaggle for missing `id`?**  
A: Update your `task_description.txt` to specify the correct submission format (e.g., `id,median_house_value`).  
Or use `make_submission.py` to add an `id` column to your predictions.

**Q: What if I hit OpenRouter rate limits?**  
A: Wait for the daily reset, add credits, or use the minimal runner to reduce LLM calls.  
Ollama/local LLM support is coming.

**Q: Can I run the full MLE-Bench?**  
A: Not on typical local hardware. This repo is for experimentation and research, not leaderboard competition.

---

## Appendix: Required Config Parameters

See the `DefaultConfig` dataclass in `machine_learning_engineering/shared_libraries/config.py` for all options.

---

## Citation

If you use this repo, please cite the original MLE-STAR paper:

> [MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement](https://arxiv.org/abs/2506.15692)

---

## License

MIT License (see LICENSE).  
This project is independent and not affiliated with the MLE-STAR authors or Google.
