# Changelog

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
