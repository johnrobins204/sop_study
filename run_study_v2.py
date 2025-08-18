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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict

import requests
import pandas as pd
import google.generativeai as genai
import google.api_core.exceptions

from dotenv import load_dotenv
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from queue import Queue

# --- Load environment variables ---
load_dotenv()

# --- Global Configuration & Utilities ---
CONFIG = {}
HTTP_SESSION = requests.Session()
GLOBAL_FILE_LOCK = threading.Lock()

# --- START: New Object-Oriented Structure ---

@dataclass
class ModelResponse:
    """A structured container for a model's response."""
    raw_text: str = ""
    error: str | None = None
    api_latency_ms: int = 0
    finish_reason: str | None = None
    safety_ratings: dict = field(default_factory=dict)

@dataclass
class JudgeEvaluation:
    """A structured container for a judge's evaluation."""
    raw_judge_response: str = ""
    judge_error: str | None = None
    scores: dict = field(default_factory=dict)

@dataclass
class Trial:
    """Represents a single experimental trial."""
    problem_id: str
    test_name: str
    question: str
    source_text: str
    ablation_condition: str
    model_response: ModelResponse | None = None
    judge_evaluation: JudgeEvaluation | None = None

    def to_csv_row(self, columns):
        """Flattens the Trial object into a dictionary for CSV writing."""
        row = {
            'problem_id': self.problem_id,
            'test_name': self.test_name,
            'question': self.question,
            'source_text': self.source_text,
            'ablation_condition': self.ablation_condition,
        }
        if self.model_response:
            row['raw_model_output'] = self.model_response.raw_text
            row['error'] = self.model_response.error
            row['api_latency_ms'] = self.model_response.api_latency_ms
        
        if self.judge_evaluation:
            row['raw_judge_response'] = self.judge_evaluation.raw_judge_response
            row['judge_error'] = self.judge_evaluation.judge_error
            # Flatten scores
            for k, v in self.judge_evaluation.scores.items():
                row[f"score_{k}_rating"] = v.get('rating')
                row[f"score_{k}_justification"] = v.get('justification')

        # Ensure all columns are present
        return {c: row.get(c, "") for c in columns}

class LanguageModel(ABC):
    """Abstract base class for language models."""
    def __init__(self, model_name: str, api_parameters: dict):
        self.model_name = model_name
        self.api_parameters = api_parameters

    @abstractmethod
    def generate(self, prompt: str, timeout: int) -> ModelResponse:
        """Generates a response from the model."""
        pass

class GoogleModel(LanguageModel):
    """LanguageModel implementation for the Google Gemini API."""
    def generate(self, prompt: str, timeout: int = 90) -> ModelResponse:
        start_time = time.time()
        try:
            model = genai.GenerativeModel(self.model_name)
            generation_config = genai.types.GenerationConfig(
                temperature=self.api_parameters.get("temperature"),
                max_output_tokens=self.api_parameters.get("max_tokens")
            )
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={"timeout": timeout}
            )
            latency = (time.time() - start_time) * 1000

            # Use the robust response.text property
            raw_text = response.text
            
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason.name if hasattr(candidate, 'finish_reason') else "UNKNOWN"
            safety_ratings = {r.category.name: r.probability.name for r in candidate.safety_ratings}

            return ModelResponse(
                raw_text=raw_text,
                api_latency_ms=int(latency),
                finish_reason=finish_reason,
                safety_ratings=safety_ratings
            )
        except ValueError as e:
            # This catches cases where response.text fails (e.g., blocked response)
            finish_reason = "BLOCKED"
            safety_ratings = {}
            if response and response.candidates:
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason.name if hasattr(candidate, 'finish_reason') else 'BLOCKED'
                safety_ratings = {r.category.name: r.probability.name for r in candidate.safety_ratings}
            
            error_message = f"Response blocked or empty. Finish Reason: {finish_reason}. Safety: {safety_ratings}"
            logging.warning(error_message)
            return ModelResponse(error=error_message, api_latency_ms=int((time.time() - start_time) * 1000))
        except Exception as e:
            logging.error(f"Error in GoogleModel for {self.model_name}: {e}")
            return ModelResponse(error=str(e), api_latency_ms=int((time.time() - start_time) * 1000))

class OllamaModel(LanguageModel):
    """LanguageModel implementation for a local Ollama model."""
    def generate(self, prompt: str, timeout: int = 120) -> ModelResponse:
        start_time = time.time()
        url = f"{CONFIG.get('ollama_base_url', 'http://localhost:11434')}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.api_parameters.get("temperature"),
                "num_predict": self.api_parameters.get("max_tokens")
            }
        }
        try:
            response = HTTP_SESSION.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            response_json = response.json()
            return ModelResponse(
                raw_text=response_json.get("response", ""),
                api_latency_ms=int((time.time() - start_time) * 1000)
            )
        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama API request failed for {self.model_name}: {e}")
            return ModelResponse(error=str(e), api_latency_ms=int((time.time() - start_time) * 1000))

class Judge:
    """Encapsulates all logic for LLM-based evaluation."""
    def __init__(self, model: LanguageModel, judge_prompt_template: dict):
        self.model = model
        self.prompt_template = judge_prompt_template

    def evaluate(self, trial: Trial) -> JudgeEvaluation:
        if not trial.model_response or trial.model_response.error or not trial.model_response.raw_text.strip():
            return JudgeEvaluation(judge_error="Skipped: Model generation was empty or had an error.")

        prompt = self._build_prompt(
            trial.question,
            trial.model_response.raw_text,
            trial.source_text
        )
        
        response = self.model.generate(prompt, timeout=180)
        
        if response.error or not response.raw_text.strip():
            return JudgeEvaluation(
                raw_judge_response=response.raw_text,
                judge_error=response.error or "Empty raw response from judge model."
            )

        parsed_scores = self._parse_response(response.raw_text)
        
        return JudgeEvaluation(
            raw_judge_response=response.raw_text,
            judge_error=parsed_scores.get("error"),
            scores=parsed_scores.get("scores", {})
        )

    def _build_prompt(self, question: str, answer: str, source_text: str) -> str:
        sop_str = json.dumps(self.prompt_template, indent=2)
        source_block = f"**SOURCE TEXTS FOR EVALUATION:**\n{source_text}\n\n---\n\n" if source_text.strip() else ""
        return (
            f"Please act as an impartial evaluator. Your instructions are provided in the following JSON object:\n\n"
            f"{sop_str}\n\n---\n\n"
            f"{source_block}"
            f"**QUESTION TO EVALUATE:**\n{question}\n\n"
            f"**MODEL'S ANSWER TO EVALUATE:**\n{answer}\n\n---\n\n"
            "Based on your instructions, provide your evaluation in the specified JSON format, enclosed between <<JUDGE_JSON>> and <</JUDGE_JSON>> markers."
        )

    def _parse_response(self, text: str) -> dict:
        try:
            match = re.search(r'<<JUDGE_JSON>>\s*(.*?)\s*<</JUDGE_JSON>>', text, re.DOTALL)
            if not match:
                return {"error": "Failed to find <<JUDGE_JSON>> markers."}
            
            parsed = json.loads(match.group(1))
            scores = {}
            score_keys = ["faithfulness", "correctness", "completeness", "clarity", "citation_following"]
            
            for key in score_keys:
                # Flexible key matching
                rating_val = parsed.get(key) or parsed.get(f"score_{key}") or parsed.get(f"score_{key}_rating")
                just_val = parsed.get(f"justification_{key}") or parsed.get(f"{key}_justification") or parsed.get(f"score_{key}_justification")
                
                scores[key] = {
                    "rating": int(rating_val) if rating_val is not None else None,
                    "justification": str(just_val) if just_val is not None else ""
                }
            return {"scores": scores}
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            return {"error": f"JSON parsing failed: {e}"}

# --- END: New Object-Oriented Structure ---


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

def main():
    script_start_time = time.time()
    load_config()
    setup_google_api_client()

    # Setup logging and results file
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
        'score_clarity_rating', 'score_clarity_justification',
        'score_citation_following_rating', 'score_citation_following_justification'
    ]
    # Initialize CSV with header
    with open(log_filename, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=results_columns)
        writer.writeheader()

    # Load resources
    dataset = load_questions_from_json(CONFIG.get("questions_file", "questions.json"))
    if CONFIG.get("obs", 0) > 0:
        dataset = dataset[:CONFIG.get("obs")]

    # Instantiate models and judge
    api_params = CONFIG.get("api_parameters", {})
    models_to_test = {
        name: OllamaModel(config['model_name'], api_params)
        for name, config in CONFIG.get("models_to_test", {}).items()
    }
    
    judge = None
    if CONFIG.get("auto_judge", False):
        judge_prompt_template = load_json_file(CONFIG.get("judge_prompt_file", "config/judge_sop.json"))
        judge_model = GoogleModel(CONFIG.get('judge_model_name'), api_params)
        judge = Judge(model=judge_model, judge_prompt_template=judge_prompt_template)
        print(f"Auto-judging enabled with model: {judge.model.model_name}")

    # Main processing loop
    for item in tqdm(dataset, desc="Processing Questions"):
        source_text = load_source_text_for_papers(item.get("source_papers", []))
        completed_trials_for_question = []

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
                completed_trials_for_question.append(completed_trial)

        # Judge the batch of results for this question
        if judge and completed_trials_for_question:
            run_judging_for_trials(judge, completed_trials_for_question)

        # Save results for this question to CSV
        if completed_trials_for_question:
            write_trials_to_csv(log_filename, completed_trials_for_question, results_columns)
            if CONFIG.get("verbose_mode"):
                print(f"  -> Saved {len(completed_trials_for_question)} results for question {item['problem_id']}")

    print("\n--- Experiment Complete ---")
    print(f"Total time: {time.time() - script_start_time:.2f} seconds")

if __name__ == "__main__":
    main()