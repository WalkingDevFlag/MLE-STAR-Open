# Machine Learning Engineering with Multiple Agents (MLE-STAR)

## Overview

The Machine Learning Engineering Agent is an approach to building Machine Learning Engineering (MLE) agents that can train state-of-the-art machine learning models on various tasks (including classification and regression tasks), through a novel approach of leveraging web search and targeted code block refinement. Using the example of predicting California housing prices, we show how MLE-STAR can create a regression model based on factors like population, income, etc. that outperforms traditional approaches to training ML models. The experimental results show that MLE-STAR achieves medals in 63.6% of the Kaggle competitions on the MLE-bench-Lite, significantly outperforming the best alternative. The implementation is based on the Google Cloud AI Research paper "MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement" (https://www.arxiv.org/abs/2506.15692).

#### Performance of MLE agents on [MLE-Bench-Lite](https://github.com/openai/mle-bench/tree/main) datasets.

| MLE Agents | Base LLM | Any Medals| Gold Medals | Silver Medals | Bronze Medals |
| --- | --- | --- | --- | --- | --- |
| [ **MLE-STAR** ](https://www.arxiv.org/pdf/2506.15692) | **Gemini-2.5-Pro** | **63.6%** | **36.4%** | **21.2%** | 6.1% |
| [ **MLE-STAR** ](https://www.arxiv.org/pdf/2506.15692) | **Gemini-2.5-Flash** | 43.9% | 30.3% | 4.5% | **9.1%** |
---

<br>

## Agent Details

The key features of the Machine Learning Agent include:

| Feature | Description |
| --- | --- |
| **Interaction Type** | Conversational |
| **Complexity**  | Advanced |
| **Agent Type**  | Multi Agent |
| **Components**  | Tools: Code execution, Retrieval |
| **Vertical**  | All |

### Agent architecture

This diagram shows the detailed architecture of the agents and tools used
to implement this workflow.
<img src="machine-learning-engineering-architecture.svg" alt="Machine-Learning-Engineering" width="800"/>

### Key Features

1. **Initial Solution Generation:** Uses a search engine to retrieve state-of-the-art models and their example codes, then merges the best-performing candidates into a consolidated initial solution.

2. **Code Block Refinement:** Iteratively improves the solution by identifying and targeting specific code blocks (ML pipeline components) that have the most significant impact on performance, determined through ablation studies. An inner loop refines the targeted block with various strategies.

3. **Ensemble Strategies:** Introduces a novel ensembling method where the Agent proposes and refines ensemble strategies to combine multiple solutions, aiming for superior performance than individual best solutions.

4. **Robustness Modules:** Includes a debugging agent for error correction, a data leakage checker to prevent improper data access during preprocessing, and a data usage checker to ensure all provided data sources are utilized.

## Setup and Installation

1.  **Prerequisites**

    *   Python 3.12+
    *   Poetry
        *   For dependency management and packaging. Please follow the
            instructions on the official
            [Poetry website](https://python-poetry.org/docs/) for installation.

        ```bash
        pip install poetry
        ```
    *  Git
        *   Git can be downloaded from https://git-scm.com/. Then follow the [installation guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).


    * Google Cloud Account
        *   You need a Google Cloud account
    * A project on Google Cloud Platform
    * Google Cloud CLI
        *   For installation, please follow the instruction on the official
            [Google Cloud website](https://cloud.google.com/sdk/docs/install).

2.  **Installation and Setup**

    *   Clone repository
        ```bash
        # Clone this repository.
        git clone https://github.com/google/adk-samples.git
        cd adk-samples/python/agents/machine-learning-engineering
        ```

    *   Install Poetry
        ```bash
        # Install the Poetry package and dependencies.
        poetry install
        ```

        This command reads the `pyproject.toml` file and installs all the necessary dependencies into a virtual environment managed by Poetry.

        If the above command returns with a `command not found` error, then use:

        ```bash
        python -m poetry install
        ```

    *   Activate the shell

        ```bash
        poetry env activate
        ```

        This activates the virtual environment, allowing you to run commands within the project's environment. To make sure the environment is active, use for example

        ```bash
        $> poetry env list
        machine-learning-engineering-Gb54hHID-py3.12 (Activated)
        ```

        If the above command did not activate the environment for you, you can also activate it through

        ```bash
        source $(poetry env info --path)/bin/activate
        ```

<a name="configuration"></a>

3.  **Configuration**

    *   Set up Google Cloud credentials.

        *   You may set the following environment variables in your shell, or in
            a `.env` file instead.

        ```bash
        export GOOGLE_GENAI_USE_VERTEXAI=true
        export GOOGLE_CLOUD_PROJECT=<your-project-id>
        export GOOGLE_CLOUD_LOCATION=<your-project-location>
        export ROOT_AGENT_MODEL=<Google LLM to use>
        export GOOGLE_CLOUD_STORAGE_BUCKET=<your-storage-bucket>  # Only required for deployment on Agent Engine
        ```

    *   Authenticate your GCloud account.

        ```bash
        gcloud auth application-default login
        gcloud auth application-default set-quota-project $GOOGLE_CLOUD_PROJECT
        ```

## Running the Agent

**Prepare your task**

You should prepare the inputs for your task in the following way:

1. Create a folder under `tasks` with the name of your task.
2. In that folder, create a file containing the description of the task.
3. Place the data files in this folder.

**Using `adk`**

ADK provides convenient ways to bring up agents locally and interact with them.
You may talk to the agent using the CLI:

```bash
adk run machine_learning_engineering
```

Or via the Poetry shell:
```bash
poetry run adk run machine_learning_engineering
```

Or on a web interface:

```bash
 adk web
```

The command `adk web` will start a web server on your machine and print the URL.

### Example Interaction

You may open the URL, select "machine_learning_engineering" in the top-left drop-down menu, and
a chatbot interface will appear on the right. The conversation is initially
blank. Here are some example requests you may ask the Machine Learning Agent to identity itself:

> **[user]:** who are you?

> **[mle_frontdoor_agent]:** I am a machine learning engineer agent.

> **[user]:** what can you do?

> **[mle_frontdoor_agent]:** I am a machine learning engineer. My primary role is to engineer solutions for machine learning tasks, such as the California Housing Task. I can also describe the task if you'd like. I work by executing a sequence of sub-agents to solve the machine learning engineering task.

> **[user]:** describe the task that you have

> **[mle_frontdoor_agent]:** The task I have is the California Housing Task. This task involves predicting the median house value in California districts, given various features about those districts. It's a regression problem where the goal is to build a model that can accurately estimate house prices based on factors like population, median income, and housing age within a district.

> **[user]:** execute the task

> **[mle_frontdoor_agent]:** *\<intermediate output snipped\>*.
<br>
\# Save the submission file to CSV without the index
print(f"Submission file saved successfully to {submission_file_path}")

## Running Tests

For running tests and evaluation, install the extra dependencies:

```bash
poetry install --with dev
```

Then the tests and evaluation can be run from the `machine-learning-engineering` directory using
the `pytest` module:

```bash
python3 -m pytest tests
python3 -m pytest eval
```

`tests` runs the agent on a sample request, and makes sure that every component
is functional. `eval` is a demonstration of how to evaluate the agent, using the
`AgentEvaluator` in ADK. It sends a couple requests to the agent and expects
that the agent's responses match a pre-defined response reasonablly well.


## Deployment

You will need to have specified a GCS bucket in the environment variable `GOOGLE_CLOUD_BUCKET` as detailed in the [Configuration](#configuration) section.

If the bucket does not exist, ADK will create one for you. This is the easiest option. If the bucket does exist, then you must provide permissions to the service account as described in [this](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/troubleshooting/deploy#permission_errors) Troubleshooting article.

The Machine Learning Engineering Agent can be deployed to Vertex AI Agent Engine using the following
commands:

```bash
poetry install --with deployment
python3 deployment/deploy.py --create
```

When the deployment finishes, it will print a line like this:

```
Created remote agent: projects/<PROJECT_NUMBER>/locations/<PROJECT_LOCATION>/reasoningEngines/<AGENT_ENGINE_ID>
```

If you forget the AGENT_ENGINE_ID, you can list the existing agents using:

```bash
python3 deployment/deploy.py --list
```

The output will be like:

```
All remote agents:

123456789 ("machine_learning_engineering")
- Create time: 2025-07-11 09:46:07+00:00
- Update time: 2025-05-10 09:46:09+00:00
```

You may interact with the deployed agent using the `test_deployment.py` script
```bash
$ export USER_ID=<any string>
$ python3 deployment/test_deployment.py --resource_id=${AGENT_ENGINE_ID} --user_id=${USER_ID}
Found agent with resource ID: ...
Created session for user ID: ...
Type 'quit' to exit.
Input: Hello. What can you do for me?
Response: Hello! I'm a Machine Learning Engineer Assistant. I can help you achieve competition-level quality in solving machine learning tasks.

To get started, please provide the task description of the competition.
```

To delete the deployed agent, you may run the following command:

```bash
python3 deployment/deploy.py --delete --resource_id=${AGENT_ENGINE_ID}
```


## Appendix

### Required Config Parameters

This document describes the required configuration parameters in the `DefaultConfig` dataclass.

---

#### `data_dir`
-   **Description:** Specifies the directory path where the machine learning tasks and their data are stored.
-   **Type:** `str`
-   **Default:** `"./machine_learning_engineering/tasks/"`

---

#### `task_name`
-   **Description:** The name of the specific task to be loaded and processed.
-   **Type:** `str`
-   **Default:** `"california-housing-prices"`

---

#### `task_type`
-   **Description:** Defines the type of machine learning problem.
-   **Type:** `str`
-   **Default:** `"Tabular Regression"`

---

#### `lower`
-   **Description:** A boolean flag, indicating whether a lower value of the metric is better.
-   **Type:** `bool`
-   **Default:** `True`

---

#### `workspace_dir`
-   **Description:** The directory path used for saving intermediate outputs, results, logs, or any other artifacts generated during the task execution.
-   **Type:** `str`
-   **Default:** `"./machine_learning_engineering/workspace/"`

---

#### `agent_model`
-   **Description:** Specifies the identifier for the LLM model to be used by the agent. It defaults to the value of the environment variable `ROOT_AGENT_MODEL` or `"gemini-2.0-flash-001"` if the variable is not set.
-   **Type:** `str`
-   **Default:** `os.environ.get("ROOT_AGENT_MODEL", "gemini-2.0-flash-001")`
