import os
import sys
import re
import json
import csv
import time
import hashlib
import threading
import concurrent.futures
import logging
import platform
import argparse

import requests
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

# --- Local Imports from new modules ---
from src.data_structures import Trial, ModelResponse, JudgeEvaluation
from src.models import LanguageModel, get_model_instance, CONTEXT_CACHE

# --- Load environment variables ---
load_dotenv()

# --- Global Configuration & Utilities ---
CONFIG = {}
HTTP_SESSION = requests.Session()
GLOBAL_FILE_LOCK = threading.Lock()

class Judge:
    """Encapsulates all logic for LLM-based evaluation."""
    def __init__(self, model: LanguageModel, judge_criteria_config: dict):
        self.model = model
        # Stores the file paths for each criterion's JSON template
        self.criteria_templates = judge_criteria_config
        # This cache is now only used if the judge is NOT a local Ollama model
        self.source_text_cache = {}

    def _cache_source_text(self, source_text: str) -> str:
        """Hashes and caches the source text, returning a key."""
        if not source_text:
            return ""
        
        key = f"sha256:{hashlib.sha256(source_text.encode('utf-8')).hexdigest()}"

        # For local models, use the global context cache to avoid resending large texts
        if "OllamaModel" in str(type(self.model)):
            # Use standard dictionary assignment for the global cache
            CONTEXT_CACHE[key] = source_text
        else:
            # For remote models (like Google), use the instance-level cache
            self.source_text_cache[key] = source_text
        
        return key

    def _evaluate_criterion(self, criterion_name: str, question: str, answer: str, source_text_or_key: str) -> tuple[str, dict]:
        """Evaluates a single criterion for a given trial."""
        prompt = self._build_prompt_for_criterion(criterion_name, question, answer, source_text_or_key)
        response = self.model.generate(prompt, timeout=90)

        if response.error or not response.raw_text.strip():
            return criterion_name, {"error": response.error or "Empty response", "raw_response": ""}

        parsed = self._parse_criterion_response(response.raw_text)
        # Always include the raw response for debugging
        parsed["raw_response"] = response.raw_text
        return criterion_name, parsed

    def evaluate(self, trial: Trial) -> JudgeEvaluation:
        """
        Evaluates a model's response against all configured criteria.
        This can be run in parallel for multiple trials.
        """
        # Robust Guard Clause: Explicitly check if the raw_text is a non-empty string.
        # This handles None, NaN, and empty/whitespace-only strings correctly.
        if not isinstance(trial.model_response.raw_text, str) or not trial.model_response.raw_text.strip():
            return JudgeEvaluation(judge_error="Skipped: Model generation was empty or had an error.")

        # --- Setup for Parallel Judging ---
        all_results = {}
        all_errors = []
        
        # Cache the source text once before starting parallel evaluations
        source_key = self._cache_source_text(trial.source_text)

        judge_workers = int(CONFIG.get("judge_workers", 4))
        with concurrent.futures.ThreadPoolExecutor(max_workers=judge_workers, thread_name_prefix="JudgeCriterion") as executor:
            future_to_criterion = {
                executor.submit(self._evaluate_criterion, name, trial.question, trial.model_response.raw_text, source_key): name
                for name in self.criteria_templates.keys()
            }
            for future in concurrent.futures.as_completed(future_to_criterion):
                criterion_name = future_to_criterion[future]
                try:
                    name, result = future.result()
                    all_results[name] = {
                        "rating": result.get("rating"),
                        "justification": result.get("justification", "")
                    }
                    # Store the raw judge response for this criterion
                    if result.get("raw_response"):
                        all_results[name]["raw_response"] = result["raw_response"]
                    if result.get("error"):
                        all_errors.append(f"{name}: {result['error']}")
                except Exception as e:
                    all_errors.append(f"Future for {criterion_name} failed: {e}")

        return JudgeEvaluation(
            raw_judge_response=json.dumps(all_results, indent=2), # Save all raw responses
            judge_error=" | ".join(all_errors) if all_errors else None,
            scores=all_results
        )

    def _build_prompt_for_criterion(self, criterion_name: str, question: str, answer: str, source_text_key: str) -> str:
        """Builds a focused prompt by injecting the full JSON for a single evaluation criterion."""
        source_text = ""
        # Look up the source text from the appropriate cache
        if source_text_key:
            if "OllamaModel" in str(type(self.model)):
                source_text = CONTEXT_CACHE.get(source_text_key, "[Source Text Not Found in Global Cache]")
            else:
                source_text = self.source_text_cache.get(source_text_key, "[Source Text Not Found in Instance Cache]")

        # Load the specific JSON template for the criterion
        template_path = self.criteria_templates[criterion_name]
        judge_template_obj = load_json_file(template_path)
        judge_instruction_json = json.dumps(judge_template_obj, indent=2)
        
        # Extract the output format template from the JSON
        output_instructions = judge_template_obj.get("outputFormat", {}).get("template", "Critique: [Critique]\nScore: [Score]")

        return (
            f"**YOUR INSTRUCTIONS ARE PROVIDED IN THIS JSON OBJECT:**\n"
            f"```json\n{judge_instruction_json}\n```\n\n"
            f"**SOURCE TEXT(S) FOR EVALUATION:**\n{source_text}\n\n"
            f"**MODEL'S ANSWER TO EVALUATE:**\n{answer}\n\n---\n\n"
            f"**YOUR RESPONSE (use this exact format):**\n{output_instructions}"
        )

    def _parse_criterion_response(self, text: str) -> dict:
        """Parses the 'Critique: ... Score: ...' format with fallbacks."""
        try:
            # Make regex robust to optional markdown asterisks around keywords
            critique_match = re.search(r"\*?\*?Critique:\*?\*?\s*(.*)", text, re.IGNORECASE | re.DOTALL)
            score_match = re.search(r"\*?\*?Score:\*?\*?\s*(\d+)", text, re.IGNORECASE)

            critique = critique_match.group(1).strip() if critique_match else "N/A"
            score = int(score_match.group(1).strip()) if score_match else None

            # Fallback: If 'Score: X' is not found, look for a number on the last non-empty line.
            if score is None:
                lines = [line.strip() for line in text.strip().split('\n')]
                last_line = lines[-1] if lines else ""
                # Check if the last line contains only a number
                if last_line.isdigit():
                    score = int(last_line)
                    # If critique was not found, use the rest of the text as critique
                    if critique == "N/A" and len(lines) > 1:
                        critique = "\n".join(lines[:-1]).strip()


            if score is None:
                return {"error": "Could not parse score from response."}

            return {"rating": score, "justification": critique}
        except Exception as e:
            return {"error": f"Parsing failed: {e}"}

# --- Configuration and Setup (largely unchanged) ---
def setup_google_api_client():
    """Initializes the Google Generative AI client with the API key."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            sys.exit("--- FATAL ERROR --- \n'GOOGLE_API_KEY' not found.")
        genai.configure(api_key=api_key)
    except Exception as e:
        sys.exit(f"--- FATAL ERROR --- \nFailed to configure Google AI client: {e}")

def load_config(filepath="config/configs.json"):
    """Loads the main configuration file."""
    global CONFIG
    script_dir = Path(__file__).parent
    config_path = script_dir / filepath
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            CONFIG = json.load(f)
        print(f"✅ Successfully loaded configuration from '{config_path}'")
    except Exception as e:
        sys.exit(f"--- FATAL ERROR --- \nCould not load or parse '{config_path}': {e}")

def load_json_file(filepath):
    """Generic JSON file loader."""
    script_dir = Path(__file__).parent
    full_path = script_dir / filepath
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise IOError(f"Failed to load JSON file '{full_path}': {e}")

def load_questions_from_json(filepath="questions.json"):
    """Parses the multi-document questions.json file."""
    problems = []
    data = load_json_file(filepath)
    for i, paper_block in enumerate(data):
        source_papers = paper_block.get("source_papers")
        if not source_papers: continue
        for question_text in paper_block.get("questions", []):
            problem_id = f"synthesis-{len(problems)+1}"
            problems.append({"problem_id": problem_id, "source_papers": source_papers, "question": question_text})
    print(f"✅ Successfully loaded {len(problems)} multi-document problems.")
    return problems

def load_source_text_for_papers(source_papers_list):
    """Loads full text files for the provided paper titles."""
    if not isinstance(source_papers_list, list): return ""
    base = Path(CONFIG.get("source_text_dir", "sources"))
    parts = []
    for title in source_papers_list:
        safe = re.sub(r'[^A-Za-z0-9_\-]+', '_', title).strip('_')
        for ext in ['.txt', '.md', '']:
            path = base / f"{safe}{ext}"
            if path.exists():
                try:
                    parts.append(f"<!-- SOURCE: {title} -->\n{path.read_text(encoding='utf-8')}\n\n")
                    break
                except Exception:
                    continue
    return "\n".join(parts)

def write_trials_to_csv(log_filename, trials, columns):
    """Appends a list of Trial objects to a CSV file."""
    with GLOBAL_FILE_LOCK:
        file_exists = os.path.exists(log_filename)
        with open(log_filename, 'a', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            if not file_exists:
                writer.writeheader()
            for trial in trials:
                writer.writerow(trial.to_csv_row(columns))

# --- Main Experiment Logic ---

def run_trial(model: LanguageModel, trial_data: dict) -> Trial:
    """Runs a single generation trial for one model."""
    prompt = trial_data['prompt']
    
    # Verbose logging
    if CONFIG.get("verbose_mode"):
        print(f"\n--- API CALL for {trial_data['test_name']} ---")
        print(f"MODEL: {model.model_name}")
        print(f"PROMPT (first 500 chars):\n{prompt[:500]}...")

    response = model.generate(prompt, timeout=120)
    
    if CONFIG.get("verbose_mode"):
        print(f"RAW RESPONSE (first 500 chars):\n{response.raw_text[:500]}...")
        if response.error:
            print(f"ERROR: {response.error}")
        print("-----------------------\n")

    return Trial(
        problem_id=trial_data['problem_id'],
        test_name=trial_data['test_name'],
        question=trial_data['question'],
        source_text=trial_data['source_text'],
        ablation_condition=trial_data['ablation_condition'],
        model_response=response
    )

def run_judging_for_trials(judge: Judge, trials: list[Trial]):
    """Runs judging in parallel for a list of completed trials."""
    if not trials:
        return

    if CONFIG.get("verbose_mode"):
        print(f"\n--- Starting Judging for {len(trials)} items ---")

    judge_workers = int(CONFIG.get("judge_workers", 4))
    with concurrent.futures.ThreadPoolExecutor(max_workers=judge_workers, thread_name_prefix="Judge") as executor:
        future_to_trial = {executor.submit(judge.evaluate, trial): trial for trial in trials}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_trial), total=len(trials), desc="  Judging Batch", unit="q", leave=False):
            trial = future_to_trial[future]
            try:
                trial.judge_evaluation = future.result()
            except Exception as e:
                trial.judge_evaluation = JudgeEvaluation(judge_error=f"Judge future failed: {e}")

def run_judge_only_mode(input_filepath: str, mode: str):
    """Loads a CSV, runs judging on its contents, and saves to a new file."""
    script_start_time = time.time()
    print(f"--- Running in Judge-Only Mode ('{mode}') on file: {input_filepath} ---")
    
    # Config is already loaded by the main execution block
    setup_google_api_client()

    # 1. Setup Judge
    if not CONFIG.get("auto_judge", False):
        print("Auto-judging is disabled in the config. Exiting.")
        return
    
    judge_criteria_config = CONFIG.get("judge_criteria")
    if not judge_criteria_config:
        print("❌ Error: 'judge_criteria' mapping not found in config. Cannot proceed with judging.")
        return

    api_params = CONFIG.get("api_parameters", {})
    
    # Use the factory to get the correct judge model instance
    judge_model_identifier = CONFIG.get('judge_model_name')
    if not judge_model_identifier:
        print("❌ Error: 'judge_model_name' not found in config. Cannot create judge.")
        return
    # Add CONFIG as the second argument
    judge_model = get_model_instance(judge_model_identifier, CONFIG, api_params)
    judge = Judge(judge_model, judge_criteria_config)

    # 2. Load existing results from CSV
    try:
        df = pd.read_csv(input_filepath)
        # Fill NaNs in object columns to prevent errors
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('')
        
        # Explicitly handle source_text to ensure it's a string and NaNs are empty strings.
        if 'source_text' in df.columns:
            df['source_text'] = df['source_text'].fillna('').astype(str)

    except FileNotFoundError:
        sys.exit(f"--- FATAL ERROR --- \nInput file not found: {input_filepath}")

    # 3. Prepare output file and get rows to judge
    p = Path(input_filepath)
    suffix = "patched" if mode == 'patch' else "rejudged"
    output_filename = p.parent / f"{p.stem}_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_columns = list(df.columns)
    
    # Write header to the new file immediately
    with open(output_filename, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=results_columns)
        writer.writeheader()
    
    print(f"✅ Output will be saved to '{output_filename}'")

    # 4. Process and judge the selected rows
    with tqdm(total=df.shape[0], desc=f"Judging ({mode})") as pbar:
        for index, row in df.iterrows():
            # Determine if this row needs judging based on the mode
            needs_judging = False
            if mode == 'full':
                needs_judging = True
            elif mode == 'patch':
                is_response_empty = str(row.get('raw_judge_response', '')).strip() == ''
                has_error = str(row.get('judge_error', '')).strip() != ''
                if is_response_empty or has_error:
                    needs_judging = True

            trial = Trial(
                problem_id=row.get('problem_id', ''),
                test_name=row.get('test_name', ''),
                question=row.get('question', ''),
                source_text=row.get('source_text', ''),
                ablation_condition=row.get('ablation_condition', ''),
                model_response=ModelResponse(
                    raw_text=row.get('raw_model_output', ''),
                    error=row.get('error') if row.get('error') else None,
                    api_latency_ms=int(row.get('api_latency_ms', 0))
                )
            )

            if needs_judging:
                # Run the evaluation and assign the result back to the trial object
                trial.judge_evaluation = judge.evaluate(trial)
            else:
                # If not judging, reconstruct the old evaluation to preserve it
                trial.judge_evaluation = JudgeEvaluation(
                    raw_judge_response=row.get('raw_judge_response', ''),
                    judge_error=row.get('judge_error', ''),
                    scores={crit: {"rating": row.get(f'score_{crit}_rating'), "justification": row.get(f'score_{crit}_justification')} for crit in judge.criteria_templates}
                )

            # Atomically append the result for this single trial to the new CSV
            write_trials_to_csv(output_filename, [trial], results_columns)
            pbar.update(1)

    print(f"\n✅ {mode.capitalize()} run complete. All results saved to '{output_filename}'")
    print(f"--- Judge-Only Mode Complete ---")
    print(f"Total time: {time.time() - script_start_time:.2f} seconds")


def main():
    script_start_time = time.time()
    # The argument parsing is now handled in the `if __name__ == "__main__"` block.
    # This function will only be called if we are NOT in judge-only mode.
    
    load_config()
    setup_google_api_client()

    # --- Setup Models & Judge ---
    models_to_test = {}
    api_params = CONFIG.get("api_parameters", {})
    for name, conf in CONFIG.get("models_to_test", {}).items():
        # Add CONFIG as the second argument
        models_to_test[name] = get_model_instance(conf['model_name'], CONFIG, api_params)

    judge = None
    if CONFIG.get("auto_judge", False):
        judge_model_id = CONFIG.get('judge_model_name')
        judge_criteria_cfg = CONFIG.get('judge_criteria')
        if judge_model_id and judge_criteria_cfg:
            # Add CONFIG as the second argument
            judge_model = get_model_instance(judge_model_id, CONFIG, api_params)
            judge = Judge(judge_model, judge_criteria_cfg)
        else:
            print("⚠️ Warning: Auto-judging enabled but 'judge_model_name' or 'judge_criteria' is missing in config.")

    # --- Setup Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = Path("results") / f"full_log_{CONFIG.get('experiment_name', 'exp')}_{timestamp}.csv"
    log_filename.parent.mkdir(parents=True, exist_ok=True)
    print(f"✅ Results will be saved to '{log_filename}'")

    # Define CSV columns
    results_columns = [
        'problem_id', 'test_name', 'ablation_condition', 'raw_model_output', 'question',
        'error', 'api_latency_ms', 'source_text', 'raw_judge_response', 'judge_error',
        'score_faithfulness_rating', 'score_faithfulness_justification',
        'score_correctness_rating', 'score_correctness_justification',
        'score_completeness_rating', 'score_completeness_justification',
        'score_synthesis_rating', 'score_synthesis_justification',
        'score_clarity_rating', 'score_clarity_justification',
        'score_reasoning_quality_rating', 'score_reasoning_quality_justification',
        'score_depth_of_insight_rating', 'score_depth_of_insight_justification',
        'score_conciseness_rating', 'score_conciseness_justification'
    ]
    # Initialize CSV with header
    with open(log_filename, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=results_columns)
        writer.writeheader()

    # Load resources
    dataset = load_questions_from_json(CONFIG.get("questions_file", "questions.json"))
    if CONFIG.get("obs", 0) > 0:
        dataset = dataset[:CONFIG.get("obs")]

    # Main processing loop
    for item in tqdm(dataset, desc="Processing Questions"):
        source_text = load_source_text_for_papers(item.get("source_papers", []))

        for iter_num in range(CONFIG.get("iters", 1)):
            for ablation_name, ablation_config in CONFIG.get("models_to_test", {}).items():
                model = models_to_test[ablation_name]
                
                # Build prompt from SOP/prefix
                sop_content = ""
                if sop_file := ablation_config.get("sop_template_file"):
                    sop_content = json.dumps(load_json_file(sop_file), indent=2)
                
                prompt_parts = [
                    ablation_config.get("prompt_prefix", ""),
                    sop_content,
                    f"---\n**USER'S QUESTION:**\n{item['question']}\n\n---\n\nBased on the instructions, synthesize an answer."
                ]
                prompt_str = "\n\n".join(filter(None, prompt_parts))

                trial_data = {
                    'problem_id': f"{item['problem_id']}-iter{iter_num + 1}",
                    'test_name': f"{ablation_name}::{Path(sop_file).stem if sop_file else 'no_sop'}",
                    'question': item['question'],
                    'source_text': source_text,
                    'ablation_condition': ablation_config.get("description", ""),
                    'prompt': prompt_str
                }
                
                # Run the generation
                completed_trial = run_trial(model, trial_data)
                
                # Judge this single trial immediately
                if judge:
                    try:
                        completed_trial.judge_evaluation = judge.evaluate(completed_trial)
                    except Exception as e:
                        completed_trial.judge_evaluation = JudgeEvaluation(judge_error=f"Judge evaluation failed: {e}")

                # Save result for this single trial to CSV
                write_trials_to_csv(log_filename, [completed_trial], results_columns)
                if CONFIG.get("verbose_mode"):
                    print(f"  -> Saved result for {trial_data['test_name']} on question {item['problem_id']}")

    print("\n--- Experiment Complete ---")
    print(f"Total time: {time.time() - script_start_time:.2f} seconds")

if __name__ == "__main__":
    # Load config to determine run mode, replacing argparse
    load_config()
    
    run_settings = CONFIG.get("run_settings", {})
    mode = run_settings.get("mode", "generate")

    if mode == "judge_only":
        input_file = run_settings.get("judge_input_file")
        judge_mode = run_settings.get("judge_mode", "full")

        if not input_file:
            sys.exit("--- FATAL ERROR --- \n'judge_input_file' not specified in config for judge_only mode.")
        
        if judge_mode not in ["full", "patch"]:
            sys.exit(f"--- FATAL ERROR --- \nInvalid 'judge_mode': '{judge_mode}'. Must be 'full' or 'patch'.")

        run_judge_only_mode(input_file, judge_mode)
    
    elif mode == "generate":
        main()
    
    else:
        sys.exit(f"--- FATAL ERROR --- \nUnknown run mode '{mode}' specified in config. Must be 'generate' or 'judge_only'.")