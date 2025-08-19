# Changelog

All notable changes to this project will be documented in this file.

## [2.0.1] - 2025-08-18

### Added
- **Project README**: Added a comprehensive `README.md` with project overview, installation instructions, and detailed usage examples for all run modes.

### Changed
- **Improved Code Hygiene**: Updated `.gitignore` to properly exclude generated cache files and directories from the repository.
- **Standardized Entry Point**: Renamed the primary execution script from `run_study_v2.py` to `main.py` to align with Python community conventions.

## [2.0.0] - 2025-08-18

### Added
- **"Judge Only" Execution Mode**: The script can now be run in a dedicated `judge_only` mode. This decouples the evaluation phase from the generation phase, allowing for re-judging of existing model outputs without re-running costly generation.
- **Efficient "Patch" and "Full" Re-judging**: Within the new Judge Only mode, you can now specify a `judge_mode` of `full` (re-judges all rows) or `patch` (intelligently re-judges only failed or skipped rows).
- **Local LLM Judge Support**: Added support for using local models (e.g., via Ollama) as an automated judge, reducing reliance on API-based services.
- **Expanded Evaluation Criteria**: Bifurcated the judging process into distinct factors and added new criteria (`Reasoning Quality`, `Depth of Insight`, `Conciseness`) based on best practices in LLM evaluation.

### Changed
- **Centralized Run Configuration**: All runtime settings, including execution modes, have been moved from command-line arguments into a `run_settings` object within `config/configs.json` for improved reproducibility.
- **Modular Project Structure**: The core Python modules have been refactored into a dedicated `src/` package to improve code maintainability and prevent conflicts with other libraries.

## v1.2.0 - August 17, 2025

### Enhancements
- **Simplified Concurrency Model**: Refactored the trial execution logic from a fully thread-pooled system to a more stable small-batch asynchronous process. This prioritizes data integrity for current experiment scales.

### Bug Fixes
- **Resolved Data Pollution in Results**: Fixed a critical bug where concurrent threads could cause data corruption in the output logs. The new batching model ensures each trial is processed in isolation, preventing cross-contamination of results.

## v1.1.0 - August 17, 2025

### Enhancements
- Added support for displaying **two progress bars**:
  - **Generator Tasks**: Tracks the progress of all generator tasks.
  - **Judge Tasks**: Tracks the progress of all judge tasks.
- Improved progress bar handling using `tqdm` with proper positioning for simultaneous display.

### Bug Fixes
- Fixed an issue where **only one iteration per question** was executed, regardless of the `iters` value specified in `configs.json`. The script now correctly processes all iterations for each question.
- Resolved a **variable scoping error** caused by asynchronous thread pooling:
  - Explicitly passed variables (`problem_id`, `test_name`, `question`, `raw_generation_for_judge`, and `judge_prompt_for_task`) to the `_run_judge_and_update` function to ensure proper scoping.
- Fixed a bug where **only one progress bar** was displayed, even when both generator and judge tasks were active.
