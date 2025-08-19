# Gnosis - Prompt Research Bench

This project provides a flexible and robust pipeline for evaluating the performance of Large Language Models (LLMs) on a given set of questions using different prompt configurations. It supports both API-based models (e.g., Google's Gemini) and locally-hosted models (e.g., via Ollama).

The pipeline is designed in two main phases:
1.  **Generate**: Models generate answers to a series of questions based on provided source texts.
2.  **Judge**: An automated LLM judge evaluates the generated answers against a configurable set of criteria.
3.  A third **Analyze** phase is coming with a full refactor and a more coherent solution design. (Planned Aug '25)

## Features

-   **Centralized JSON Configuration**: Control all aspects of an experiment from a single, easy-to-read configuration file.
-   **Decoupled Generation and Judging**: Run model generation and evaluation as separate steps.
-   **Efficient Re-Judging**: Re-evaluate results with a new judge model or criteria without having to re-run the generation phase. Supports both full re-runs and efficient "patch" runs to fill in missing evaluations.
-   **Multi-Criteria Evaluation**: Judge model outputs against multiple, distinct criteria (e.g., correctness, clarity, faithfulness), each with its own prompt template.
-   **Local & API Model Support**: Seamlessly use both local Ollama models and remote Google API models for generation and judging.

## Prerequisites

-   Python 3.10+
-   An active virtual environment (recommended)
-   [Ollama](https://ollama.com/) installed and running (if using local models)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd sop-research
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a file named `.env` in the project root and add your API keys.
    ```
    # .env
    GOOGLE_API_KEY="your_google_api_key_here"
    ```

5.  **Ensure Ollama is running (for local models):**
    Make sure the Ollama service is active and you have pulled the models you intend to use (e.g., `ollama pull llama3:8b`).

## Configuration

All experiment parameters are controlled by the `config/configs.json` file. This allows for easy management and high reproducibility.

### Key Configuration Sections

#### `run_settings`
This object controls the script's primary execution mode.

-   `"mode"`: Determines what the script will do.
    -   `"generate"`: Runs a full experiment, generating new answers and judging them.
    -   `"judge_only"`: Skips generation and only runs the judging pipeline on an existing results file.
-   `"judge_input_file"`: (Required for `judge_only` mode) The path to the CSV file containing the model outputs you want to evaluate.
-   `"judge_mode"`: (Used in `judge_only` mode)
    -   `"full"`: Performs a complete re-evaluation of every row in the input file.
    -   `"patch"`: Intelligently finds and re-judges only the rows that were previously skipped or resulted in an error.

#### Model and Judge Selection

-   `"models_to_test"`: A dictionary defining the models that will generate answers. The key is a friendly name, and the value is an object specifying the `model_name`.
-   `"auto_judge"`: Set to `true` to enable the automated evaluation phase.
-   `"judge_model_name"`: The model identifier for the judge (e.g., `"gemini-1.5-pro-preview-0409"` or `"llama3:8b"`).
-   `"ollama_models"`: A list of model identifiers that should be treated as local Ollama models. This is crucial for routing requests correctly.

#### Criteria and Data

-   `"judge_criteria"`: A dictionary where each key is a criterion name (e.g., `"correctness"`) and the value is the path to a JSON file containing the prompt template for that criterion.
-   `"questions_file"`: The path to the JSON file containing the list of questions for the experiment.
-   `"source_text_dir"`: The directory where source text files (referenced in `questions.json`) are located.

## Usage

All runs are initiated by executing the main script. The behavior is determined by your `config/configs.json` file.

```bash
python main.py
```

### Example 1: Running a Full Generation Experiment

To run a new experiment from scratch, configure `config/configs.json` as follows:

```json
// config/configs.json
{
  "run_settings": {
    "mode": "generate"
  },
  "auto_judge": true,
  "judge_model_name": "llama3:8b",
  "models_to_test": {
    "gemini-pro": { "model_name": "gemini-1.0-pro" }
  },
  ...
}
```

### Example 2: Re-judging an Entire Results File

To re-evaluate every row in an existing CSV file with a new judge model, configure it for a `full` judge-only run:

```json
// config/configs.json
{
  "run_settings": {
    "mode": "judge_only",
    "judge_input_file": "results/my_previous_run.csv",
    "judge_mode": "full"
  },
  "auto_judge": true,
  "judge_model_name": "gemini-1.5-pro-preview-0409",
  ...
}
```

### Example 3: Patching a Partially Judged File

If a previous run was interrupted, you can efficiently complete it using `patch` mode:

```json
// config/configs.json
{
  "run_settings": {
    "mode": "judge_only",
    "judge_input_file": "results/interrupted_run.csv",
    "judge_mode": "patch"
  },
  "auto_judge": true,
  "judge_model_name": "llama3:8b",
  ...
}
```

## Project Structure

```
.
├── config/              # Experiment and prompt template configurations
├── results/             # Output CSV files from experiment runs (ignored by Git)
├── sources/             # Source text documents for experiments
├── src/                 # Core Python source code modules
├── .env                 # Environment variables (e.g., API keys)
├── .gitignore           # Specifies files and directories for Git to ignore
├── main.py              # The main entry point for running the pipeline
└──
