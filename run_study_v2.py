import requests
from dotenv import load_dotenv
import google.generativeai as genai
import os
import re
import sys
import json
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import google.api_core.exceptions
import concurrent.futures
import threading
import platform
import gc  # <-- added for explicit collection

# added imports for document caching
import hashlib
from pathlib import Path
import csv

# --- Global Variables ---
CONFIG = {}
API_PARAMETERS = {}

# Document cache index (in-memory)
DOC_CACHE_INDEX = None
MASTER_DOC_CACHE = None

# add a single global requests session to reuse connections
HTTP_SESSION = requests.Session()

# NEW: single shared short-lived executor for call_with_timeout to avoid creating many executors
SHORT_CALL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# NEW: global file lock used for CSV appends (kept at module level so helpers can use it)
GLOBAL_FILE_LOCK = threading.Lock()

# --- Core Functions ---

def setup_google_api_client():
    """Initializes the Google AI client for the judge."""
    print("Configuring Google AI client for the judge...")
    try:
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not found for the judge.")
        genai.configure(api_key=api_key)
        print("✅ Google AI client configured.")
    except Exception as e:
        print(f"\n--- FATAL ERROR --- \n{e}")
        sys.exit(1)

def load_config(filepath="config/configs.json"):
    """Loads the experiment configuration."""
    global CONFIG, API_PARAMETERS
    try:
        with open(filepath, 'r') as f:
            CONFIG = json.load(f)
        API_PARAMETERS = CONFIG.get("api_parameters", {})
        # ensure explicit default for auto_judge (can be set in config to True/False)
        CONFIG.setdefault("auto_judge", True)
        # ensure default for judge model name if desired
        CONFIG.setdefault("judge_model_name", CONFIG.get("judge_model_name", "gemini-pro"))
        # Document cache defaults
        CONFIG.setdefault("use_doc_cache", True)                 # enable local caching of source texts
        CONFIG.setdefault("doc_cache_dir", "doc_cache")          # directory to store cached docs
        CONFIG.setdefault("use_api_document_cache", False)       # if True, attempt to push to provider-side cache (best-effort)
        CONFIG.setdefault("source_text_dir", "sources")          # where to look up per-paper full texts (optional)
        print(f"✅ Successfully loaded configuration from '{filepath}'")
    except FileNotFoundError:
        print(f"--- FATAL ERROR --- \nConfiguration file not found at '{filepath}'")
        sys.exit(1)

def load_sop_template(filepath):
    """Loads a JSON SOP template file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading SOP template '{filepath}': {e}")
        return None

def _normalize(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_questions_from_json(filepath="questions.json"):
    """Parses the multi-document questions.json file."""
    print(f"Loading multi-document questions from '{filepath}'...")
    problems = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for i, paper_block in enumerate(data):
            source_papers = paper_block.get("source_papers")
            if not source_papers: continue
            for question_text in paper_block.get("questions", []):
                problem_id = f"synthesis-{len(problems)+1}"
                problems.append({"problem_id": problem_id, "source_papers": source_papers, "question": question_text})
        print(f"✅ Successfully loaded {len(problems)} multi-document problems.")
        return problems
    except Exception as e:
        print(f"\n--- FATAL ERROR --- \nFailed to load multi-document questions file. Error: {e}")
        sys.exit(1)

def call_with_timeout(fn, *args, timeout=60, **kwargs):
    """
    Run fn(*args, **kwargs) using the shared SHORT_CALL_EXECUTOR and enforce a timeout.
    Returns the function return value or raises concurrent.futures.TimeoutError.
    """
    fut = SHORT_CALL_EXECUTOR.submit(fn, *args, **kwargs)
    return fut.result(timeout=timeout)

def extract_json_from_text(text):
    """
    Robust extraction of the first JSON-like object from text.
    Prefer explicit markers <<JUDGE_JSON>> ... <</JUDGE_JSON>> if present.
    Attempts small repairs (closing braces) if truncated.
    Returns parsed dict or raises ValueError.
    """
    import json, re

    if not isinstance(text, str):
        raise ValueError("No text provided for JSON extraction")

    m = re.search(r'<<JUDGE_JSON>>(.*?)<</JUDGE_JSON>>', text, re.S)
    candidate = None
    if m:
        candidate = m.group(1).strip()
    else:
        m2 = re.search(r'(\{(?:.|\s)*\})', text, re.S)
        if m2:
            candidate = m2.group(1)
        else:
            candidate = text.strip()

    for attempt in range(0, 4):
        try:
            return json.loads(candidate)
        except Exception:
            candidate = candidate + "}" * (attempt + 1)
            continue

    braces_stack = []
    start_idx = None
    for i, ch in enumerate(text):
        if ch == '{' and start_idx is None:
            start_idx = i
        if ch == '{':
            braces_stack.append('{')
        elif ch == '}' and braces_stack:
            braces_stack.pop()
            if not braces_stack and start_idx is not None:
                sub = text[start_idx:i+1]
                try:
                    return json.loads(sub)
                except Exception:
                    pass
                start_idx = None
    raise ValueError("Unable to parse JSON from judge response")

def generate_google_response(model, prompt, timeout=90, cache_id: str = ""):
    """
    Generates a response using Google Gemini API with retries and enforced timeout.
    If cache_id is provided, a short reference is prepended to the prompt so that
    provider- or application-side cached document can be used instead of resending text.
    """
    # If a cache_id is present, prefer using provider-side id (if known) so provider uses its cache.
    if cache_id:
        provider_id = _get_provider_id_for_cache(cache_id)
        marker_id = provider_id or cache_id
        prompt = f"[DOCUMENT_CACHE_ID:{marker_id}]\n{prompt}"

    start_time = time.time()
    response_data = {"raw_generation": None, "error": None}
    max_retries = 3
    backoff_seconds = 5

    def _call_generate():
        gemini_model = genai.GenerativeModel(model)
        config = genai.types.GenerationConfig(
            temperature=API_PARAMETERS.get("temperature"),
            max_output_tokens=API_PARAMETERS.get("max_tokens")
        )
        completion = gemini_model.generate_content(prompt, generation_config=config)
        txt = ""
        try:
            txt = completion.text
        except Exception:
            try:
                parts = getattr(completion, "parts", None)
                if parts:
                    txt = "".join(getattr(p, "text", "") for p in parts)
            except Exception:
                txt = ""
        return txt

    for attempt in range(1, max_retries + 1):
        try:
            raw = call_with_timeout(_call_generate, timeout=timeout)
            if not raw:
                raise ValueError("Google API returned empty response")
            response_data["raw_generation"] = raw.strip()
            response_data["error"] = None
            break
        except concurrent.futures.TimeoutError:
            response_data["error"] = f"Google API call timed out after {timeout}s (attempt {attempt})"
            time.sleep(backoff_seconds)
            backoff_seconds *= 2
        except (google.api_core.exceptions.ResourceExhausted, ValueError) as e:
            response_data["error"] = f"Google API Error: {e}. Retrying..."
            time.sleep(backoff_seconds)
            backoff_seconds *= 2
        except Exception as e:
            response_data["error"] = str(e)
            break

    response_data["api_latency_ms"] = int((time.time() - start_time) * 1000)
    return response_data

def evaluate_with_llm_judge(question, model_answer, judge_prompt=None, source_text=None, source_titles=None):
    """
    Evaluate model_answer against the question.
    If a judge_prompt is provided, use it verbatim; otherwise build a default.
    """
    if judge_prompt:
        final_judge_prompt = judge_prompt
    else:
        final_judge_prompt = (
            "You are an evaluator. Given the question and the model's answer, "
            "rate faithfulness, correctness, completeness, clarity, and citation use.\n\n"
            f"QUESTION:\n{question}\n\nMODEL ANSWER:\n{model_answer}\n\n"
            "Respond ONLY with a JSON object between <<JUDGE_JSON>> and <</JUDGE_JSON>> with keys: "
            '{"faithfulness","justification_faithfulness","correctness","justification_correctness",'
            '"completeness","justification_completeness","clarity","justification_clarity",'
            '"citation_following","justification_citation_following"}'
        )

    #print(f"[JUDGE] Called evaluate_with_llm_judge. judge_model={CONFIG.get('judge_model_name')}")
    #print("[JUDGE] Prompt length (chars):", len(final_judge_prompt))

    response = generate_google_response(CONFIG.get("judge_model_name"), final_judge_prompt, timeout=90)
    raw = response.get("raw_generation") or ""
    #print("[JUDGE] Google response error:", response.get("error"))
    #print("[JUDGE] Google raw output (truncated):", raw[:1000])

    eval_metrics = {
        "raw_judge_response": raw,
        "score_faithfulness_rating": None,
        "score_faithfulness_justification": "",
        "score_correctness_rating": None,
        "score_correctness_justification": "",
        "score_completeness_rating": None,
        "score_completeness_justification": "",
        "score_clarity_rating": None,
        "score_clarity_justification": "",
        "score_citation_following_rating": None,
        "score_citation_following_justification": "",
        "judge_error": response.get("error")
    }

    if raw:
        try:
            parsed = extract_json_from_text(raw)
            def get_int(k):
                v = parsed.get(k)
                try:
                    return int(v) if v is not None else None
                except Exception:
                    return None

            eval_metrics["score_faithfulness_rating"] = get_int("faithfulness") or get_int("score_faithfulness") or get_int("score_faithfulness_rating")
            eval_metrics["score_faithfulness_justification"] = parsed.get("justification_faithfulness") or parsed.get("faithfulness_justification") or parsed.get("score_faithfulness_justification") or ""

            eval_metrics["score_correctness_rating"] = get_int("correctness") or get_int("score_correctness_rating")
            eval_metrics["score_correctness_justification"] = parsed.get("justification_correctness") or parsed.get("correctness_justification") or ""

            eval_metrics["score_completeness_rating"] = get_int("completeness") or get_int("score_completeness_rating")
            eval_metrics["score_completeness_justification"] = parsed.get("justification_completeness") or parsed.get("completeness_justification") or ""

            eval_metrics["score_clarity_rating"] = get_int("clarity") or get_int("score_clarity_rating")
            eval_metrics["score_clarity_justification"] = parsed.get("justification_clarity") or parsed.get("clarity_justification") or ""

            eval_metrics["score_citation_following_rating"] = get_int("citation_following") or get_int("score_citation_following_rating")
            eval_metrics["score_citation_following_justification"] = parsed.get("justification_citation_following") or parsed.get("citation_following_justification") or ""
        except Exception as e:
            eval_metrics["judge_error"] = (eval_metrics.get("judge_error") or "") + f" | parse_error: {e}"
            print(f"[JUDGE] JSON parse failed: {e}")
    else:
        print("[JUDGE] Empty raw response from Google; skipping parse.")

    return eval_metrics

def generate_ollama_response(model, prompt, cache_id: str = ""):
    """Generates a response using a local Ollama server via the /api/chat endpoint."""
    # If a cache_id is present, prefer using provider-side id (if known) so provider uses its cache.
    if cache_id:
        provider_id = _get_provider_id_for_cache(cache_id)
        marker_id = provider_id or cache_id
        prompt = f"[DOCUMENT_CACHE_ID:{marker_id}]\n{prompt}"

    start_time = time.time()
    response_data = {"raw_generation": None, "error": None}
    ollama_url = f"{CONFIG.get('ollama_base_url', 'http://localhost:11434')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": API_PARAMETERS.get("temperature"),
            "num_predict": API_PARAMETERS.get("num_predict")
        }
    }
    response = None
    try:
        # use persistent session
        response = HTTP_SESSION.post(ollama_url, json=payload, timeout=360)
        response.raise_for_status()
        data = response.json()
        response_data["raw_generation"] = data.get("message", {}).get("content", "").strip()
    except requests.exceptions.RequestException as e:
        response_data["error"] = f"Ollama API request failed: {e}"
    finally:
        # close response object if we created one
        try:
            if response is not None:
                response.close()
        except Exception:
            pass
    response_data["api_latency_ms"] = int((time.time() - start_time) * 1000)
    return response_data

def _master_cache_path():
    # Master cache lives in the user's sop-research home folder so it is shared across runs
    p = Path.home() / "sop-research" / "master_doc_cache.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        try:
            p.write_text(json.dumps({}, indent=2), encoding="utf-8")
        except Exception:
            pass
    return p

def _load_master_cache():
    global MASTER_DOC_CACHE
    if MASTER_DOC_CACHE is not None:
        return MASTER_DOC_CACHE
    p = _master_cache_path()
    try:
        MASTER_DOC_CACHE = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        MASTER_DOC_CACHE = {}
    return MASTER_DOC_CACHE

def _save_master_cache():
    p = _master_cache_path()
    try:
        p.write_text(json.dumps(_load_master_cache(), indent=2), encoding="utf-8")
    except Exception:
        pass

def _get_provider_id_for_cache(cid: str):
    """
    Return a provider-side id for this cache hash if known (either in local index or master cache).
    """
    if not cid:
        return ""
    idx = _load_doc_cache_index()
    if cid in idx:
        pid = idx[cid].get("provider_id")
        if pid:
            return pid
    m = _load_master_cache()
    if cid in m:
        return m[cid].get("provider_id", "")
    return ""

def _ensure_doc_cache_dir():
    d = Path(CONFIG.get("doc_cache_dir", "doc_cache"))
    d.mkdir(parents=True, exist_ok=True)
    idx = d / "index.json"
    if not idx.exists():
        idx.write_text(json.dumps({}, indent=2))
    return d, idx

def _load_doc_cache_index():
    global DOC_CACHE_INDEX
    if DOC_CACHE_INDEX is not None:
        return DOC_CACHE_INDEX
    d, idx = _ensure_doc_cache_dir()
    try:
        DOC_CACHE_INDEX = json.loads(idx.read_text(encoding="utf-8"))
    except Exception:
        DOC_CACHE_INDEX = {}
    return DOC_CACHE_INDEX

def _save_doc_cache_index():
    d, idx = _ensure_doc_cache_dir()
    try:
        idx.write_text(json.dumps(_load_doc_cache_index(), indent=2), encoding="utf-8")
    except Exception:
        pass

def _compute_cache_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _register_with_provider(text: str, metadata: dict = None):
    """
    Best-effort attempt to call a provider-side 'create cached content' endpoint and
    return a provider-side cache id. Tries several plausible genai function names and
    extracts common id/name fields from the response. Failures are swallowed (best-effort).
    """
    # allow either "google" or "gemini" tokens in judge_model_name for provider-side ingestion
    if not text:
        return None
    jm = CONFIG.get("judge_model_name", "").lower()
    if ("google" not in jm) and ("gemini" not in jm):
        return None

    metadata = metadata or {}
    candidate_funcs = [
        getattr(genai, "create_cached_content", None),
        getattr(genai, "CreateCachedContent", None),
        getattr(genai, "upload_document", None),
        getattr(genai, "uploadDocument", None),
        getattr(genai, "cached_content_create", None),
        # try nested attributes if present
        getattr(getattr(genai, "CachedContent", None), "create", None) if getattr(genai, "CachedContent", None) else None,
    ]

    for fn in candidate_funcs:
        if not callable(fn):
            continue
        try:
            resp = fn(text=text, metadata=metadata)
            # resp may be dict-like or object with attributes. Try common return fields.
            provider_id = None
            if isinstance(resp, dict):
                provider_id = resp.get("name") or resp.get("id") or resp.get("cache_id") or resp.get("provider_id")
            else:
                provider_id = getattr(resp, "name", None) or getattr(resp, "id", None) or getattr(resp, "cache_id", None) or getattr(resp, "provider_id", None)
            if provider_id:
                return provider_id
        except Exception:
            # ignore and try next candidate
            continue
    return None

def get_or_create_doc_cache(text: str, metadata: dict = None, push_to_api: bool = False) -> str:
    """
    Create a local cache entry for `text` (if not existing) and return a stable cache_id (sha256).
    If push_to_api=True and CONFIG['use_api_document_cache'] is True, attempt to push to provider
    (best-effort). This function will consult a master cache log in ~/sop-research/master_doc_cache.json
    to see if the provider already has the document and will reuse that provider id when present.
    """
    if not text:
        return ""
    idx = _load_doc_cache_index()
    cid = _compute_cache_id(text)

    # If local entry exists, ensure provider info is synced from master if available
    if cid in idx:
        # if caller requested pushing to provider but local lacks provider_id, consult master
        if push_to_api and not idx[cid].get("provider_id"):
            master = _load_master_cache()
            mrec = master.get(cid)
            if mrec and mrec.get("provider_id"):
                idx[cid]["provider_id"] = mrec["provider_id"]
                _save_doc_cache_index()
        return cid

    # create local cached file
    d = Path(CONFIG.get("doc_cache_dir", "doc_cache"))
    d.mkdir(parents=True, exist_ok=True)
    txt_path = d / f"{cid}.txt"
    try:
        txt_path.write_text(text, encoding="utf-8")
    except Exception:
        pass

    idx[cid] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "metadata": metadata or {},
        "local_path": str(txt_path)
    }
    _save_doc_cache_index()

    # If push_to_api requested and provider caching enabled, first check master cache to avoid duplicate uploads
    if push_to_api and CONFIG.get("use_api_document_cache", False):
        try:
            master = _load_master_cache()
            mrec = master.get(cid)
            if mrec and mrec.get("provider_id"):
                # provider already has this document; propagate to local index
                idx[cid]["provider_id"] = mrec["provider_id"]
                _save_doc_cache_index()
            else:
                # Best-effort: try to upload/register with provider-side cache using explicit wrapper.
                provider_id = _register_with_provider(text, metadata=metadata)
                if provider_id:
                    idx[cid]["provider_id"] = provider_id
                    master[cid] = {
                        "provider_id": provider_id,
                        "metadata": metadata or {},
                        "local_path": str(txt_path),
                        "ingested_at": datetime.utcnow().isoformat() + "Z"
                    }
                    _save_doc_cache_index()
                    _save_master_cache()
        except Exception:
            pass

    return cid

def load_source_text_for_papers(source_papers_list):
    """
    Attempts to load full text files for the provided paper titles/ids from CONFIG['source_text_dir'].
    Expected file names: sanitized_title.txt or sanitized_title.md
    Returns concatenated text or empty string if none found.
    """
    if not isinstance(source_papers_list, (list, tuple)) or not source_papers_list:
        return ""
    base = Path(CONFIG.get("source_text_dir", "sources"))
    parts = []
    for title in source_papers_list:
        if not title:
            continue
        # sanitize filename
        safe = re.sub(r'[^A-Za-z0-9_\-]+', '_', title).strip('_')
        candidates = [base / f"{safe}.txt", base / f"{safe}.md", base / safe]
        found = None
        for c in candidates:
            if c.exists():
                try:
                    found = c.read_text(encoding="utf-8")
                    break
                except Exception:
                    continue
        if found:
            parts.append(f"<!-- SOURCE: {title} -->\n{found}\n\n")
    return "\n".join(parts)

def write_result_row_to_csv(log_filename, row, columns):
    """
    Append a single result row to CSV using csv.DictWriter and a module-level lock.
    Keeps memory usage low by avoiding creation of a temporary DataFrame.
    """
    with GLOBAL_FILE_LOCK:
        file_exists = os.path.exists(log_filename)
        # Open in append mode and write only the row
        with open(log_filename, 'a', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            if not file_exists:
                writer.writeheader()
            # Ensure all columns exist in row (fill missing with empty string/None)
            out = {c: row.get(c, "") for c in columns}
            writer.writerow(out)

def load_judge_prompt(filepath="config/judge_sop.json"):
    """Loads the judge prompt template from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"--- FATAL ERROR --- \nJudge prompt file not found at '{filepath}'.")
        sys.exit(1)
    except Exception as e:
        print(f"--- FATAL ERROR --- \nFailed to load judge prompt file. Error: {e}")
        sys.exit(1)

def get_or_create_judge_prompt_cache(filepath="config/judge_sop.json"):
    """
    Create a cache entry for the judge prompt template (if not existing) and return a stable cache_id.
    Uses the same master cache logic as document caching.
    """
    try:
        # Load the judge prompt content
        with open(filepath, 'r', encoding='utf-8') as f:
            judge_prompt_content = f.read()

        # Compute the hash of the content
        cache_id = _compute_cache_id(judge_prompt_content)

        # Check if the cache ID exists in the master cache
        master_cache = _load_master_cache()
        if cache_id in master_cache:
            print(f"✅ Judge prompt template already cached with ID: {master_cache[cache_id]['provider_id']}")
            return master_cache[cache_id]['provider_id']

        # If not cached, upload the judge prompt to the provider
        provider_id = _register_with_provider(judge_prompt_content, metadata={"type": "judge_prompt"})
        if provider_id:
            # Store the provider ID in the master cache
            master_cache[cache_id] = {
                "provider_id": provider_id,
                "metadata": {"type": "judge_prompt"},
                "local_path": filepath,
                "ingested_at": datetime.utcnow().isoformat() + "Z"
            }
            _save_master_cache()
            print(f"✅ Judge prompt template cached with new ID: {provider_id}")
            return provider_id
        else:
            print(f"--- WARNING --- \nFailed to cache judge prompt template with the provider.")
            return None
    except FileNotFoundError:
        print(f"--- FATAL ERROR --- \nJudge prompt file not found at '{filepath}'.")
        sys.exit(1)
    except Exception as e:
        print(f"--- FATAL ERROR --- \nFailed to process judge prompt file. Error: {e}")
        sys.exit(1)

def main():
    # Load config first so behavior (auto_judge etc.) is available immediately.
    load_config()

    auto_judge = bool(CONFIG.get("auto_judge", True))
    if auto_judge:
        # only initialize judge client when auto_judge is enabled
        setup_google_api_client()
    else:
        print("[INFO] auto_judge disabled in config; skipping judge client setup.")

    # Cache the judge prompt template and get its cache ID
    judge_prompt_cache_id = get_or_create_judge_prompt_cache("config/judge_sop.json")
    print(f"✅ Judge prompt cache ID: {judge_prompt_cache_id}")

    # Load the judge prompt template content
    judge_prompt_template = None
    try:
        with open("config/judge_sop.json", "r", encoding="utf-8") as f:
            judge_prompt_template = f.read()
    except FileNotFoundError:
        print("--- FATAL ERROR --- \nJudge prompt file not found at 'config/judge_sop.json'.")
        sys.exit(1)
    except Exception as e:
        print(f"--- FATAL ERROR --- \nFailed to load judge prompt file. Error: {e}")
        sys.exit(1)

    dataset = load_questions_from_json("questions.json")

    obs = CONFIG.get("obs")
    if obs is not None:
        try:
            obs = int(obs)
            dataset = dataset[:obs]
            print(f"⚡ Limiting to first {obs} observations as specified in config.")
        except Exception as e:
            print(f"--- WARNING --- \nInvalid 'obs' parameter in config: {e}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"results/full_log_{CONFIG['experiment_name']}_{timestamp}.csv"
    os.makedirs("results", exist_ok=True)

    if os.path.exists(log_filename):
        # low-memory read: only load the two columns needed to form completed_pairs
        try:
            completed_df = pd.read_csv(log_filename, comment="#", usecols=['problem_id', 'test_name'])
            completed_pairs = set(zip(completed_df['problem_id'], completed_df['test_name']))
            del completed_df
        except Exception:
            # fallback if CSV malformed or columns missing
            completed_pairs = set()
    else:
        completed_pairs = set()

    results_columns = [
        'problem_id', 'test_name', 'ablation_condition', 'model_name', 'model_version', 'api_type', 'api_version',
        'sop_template_file', 'prompt_used', 'generation_parameters', 'raw_model_output', 'source_papers', 'question',
        'error', 'api_latency_ms', 'judge_prompt', 'raw_judge_response',
        'score_faithfulness_rating', 'score_faithfulness_justification',
        'score_correctness_rating', 'score_correctness_justification',
        'score_completeness_rating', 'score_completeness_justification',
        'score_clarity_rating', 'score_clarity_justification',
        'score_citation_following_rating', 'score_citation_following_justification'
    ]
    # Only write header if the results file is new (avoid truncating existing results)
    if not os.path.exists(log_filename):
        pd.DataFrame(columns=results_columns).to_csv(log_filename, index=False)
    print(f"✅ Results will be saved progressively to '{log_filename}'")

    ablation_classes = CONFIG.get("models_to_test", {})
    if not ablation_classes:
        print("--- FATAL ERROR --- \nNo ablation classes defined in config under 'models_to_test'.")
        sys.exit(1)

    with open(log_filename, "a") as f:
        f.write(f"# System: {platform.platform()}, Python: {platform.python_version()}\n")

    error_log_file = log_filename.replace('.csv', '_errors.log')

    # common file lock for safe CSV writes (used by main writer and optionally by judge updater)
    file_lock = threading.Lock()

    # Only create judge background infrastructure when auto_judge is enabled
    if auto_judge:
        # Thread pool for asynchronous judge evaluations
        judge_workers = int(CONFIG.get("judge_workers", 4))
        judge_executor = concurrent.futures.ThreadPoolExecutor(max_workers=judge_workers)

        # cap for outstanding judge tasks to avoid unbounded memory growth
        JUDGE_QUEUE_CAP = int(CONFIG.get("judge_queue_cap", judge_workers * 8))

        # secondary progress bar to track judge tasks (position=1 places it below the main inference pbar)
        judge_pbar = tqdm(total=0, desc="Judge Tasks", position=1)

        future_lock = threading.Lock()
        future_to_meta = {}

        def _on_judge_done(fut):
            meta = None
            with future_lock:
                meta = future_to_meta.pop(fut, None)
            try:
                judge_pbar.update(1)
            except Exception:
                pass

        def _run_judge_and_update(problem_id, test_name, question, model_answer, judge_prompt):
            """
            Run the judge evaluation and update the results.
            """
            try:
                # Evaluate the model answer using the provided judge prompt
                eval_metrics = evaluate_with_llm_judge(question, model_answer, judge_prompt=judge_prompt)
            except Exception as e:
                # Handle errors during evaluation
                eval_metrics = {
                    "raw_judge_response": "",
                    "score_faithfulness_rating": None,
                    "score_faithfulness_justification": "",
                    "score_correctness_rating": None,
                    "score_correctness_justification": "",
                    "score_completeness_rating": None,
                    "score_completeness_justification": "",
                    "score_clarity_rating": None,
                    "score_clarity_justification": "",
                    "score_citation_following_rating": None,
                    "score_citation_following_justification": "",
                    "judge_error": str(e)
                }

            # Prepare the values to update in the results file
            update_values = {
                'judge_prompt': judge_prompt,
                'raw_judge_response': eval_metrics.get('raw_judge_response'),
                'score_faithfulness_rating': eval_metrics.get('score_faithfulness_rating'),
                'score_faithfulness_justification': eval_metrics.get('score_faithfulness_justification'),
                'score_correctness_rating': eval_metrics.get('score_correctness_rating'),
                'score_correctness_justification': eval_metrics.get('score_correctness_justification'),
                'score_completeness_rating': eval_metrics.get('score_completeness_rating'),
                'score_completeness_justification': eval_metrics.get('score_completeness_justification'),
                'score_clarity_rating': eval_metrics.get('score_clarity_rating'),
                'score_clarity_justification': eval_metrics.get('score_clarity_justification'),
                'score_citation_following_rating': eval_metrics.get('score_citation_following_rating'),
                'score_citation_following_justification': eval_metrics.get('score_citation_following_justification'),
            }

            # Update the results file with the evaluation metrics
            with file_lock:
                try:
                    # Load the results file
                    df = pd.read_csv(log_filename, comment="#")
                    mask = (df['problem_id'] == problem_id) & (df['test_name'] == test_name)

                    # If the row does not exist, create a new one
                    if not mask.any():
                        new_row = {
                            'problem_id': problem_id,
                            'test_name': test_name,
                            'ablation_condition': '',
                            'model_name': '',
                            'model_version': '',
                            'api_type': '',
                            'api_version': '',
                            'sop_template_file': '',
                            'prompt_used': '',
                            'generation_parameters': '',
                            'raw_model_output': model_answer,
                            'source_papers': '',
                            'question': question,
                            'error': '',
                            'api_latency_ms': None,
                            'judge_prompt': update_values['judge_prompt'],
                            'raw_judge_response': update_values['raw_judge_response'],
                            'score_faithfulness_rating': update_values['score_faithfulness_rating'],
                            'score_faithfulness_justification': update_values['score_faithfulness_justification'],
                            'score_correctness_rating': update_values['score_correctness_rating'],
                            'score_correctness_justification': update_values['score_correctness_justification'],
                            'score_completeness_rating': update_values['score_completeness_rating'],
                            'score_completeness_justification': update_values['score_completeness_justification'],
                            'score_clarity_rating': update_values['score_clarity_rating'],
                            'score_clarity_justification': update_values['score_clarity_justification'],
                            'score_citation_following_rating': update_values['score_citation_following_rating'],
                            'score_citation_following_justification': update_values['score_citation_following_justification']
                        }
                        pd.DataFrame([new_row]).to_csv(log_filename, mode='a', header=False, index=False)
                        return

                    # Update the existing row
                    idx = df.index[mask][0]
                    for col, val in update_values.items():
                        if col in df.columns:
                            # Ensure compatibility with the column's data type
                            if pd.api.types.is_numeric_dtype(df[col]):
                                try:
                                    # Convert to numeric if possible, otherwise set to NaN
                                    df.at[idx, col] = pd.to_numeric(val, errors='coerce')
                                except Exception:
                                    df.at[idx, col] = None  # Assign None for incompatible values
                            else:
                                # For non-numeric columns, assign the value directly
                                df.at[idx, col] = val
                    df.to_csv(log_filename, index=False)
                except Exception as e:
                    print(f"[JUDGE] Failed to update results file: {e}")

    # Initialize the generator progress bar (position=0 for the top bar)
    generator_pbar = tqdm(total=len(dataset) * CONFIG["iters"], desc="Generator Tasks", position=0)

    # Initialize the judge progress bar (position=1 for the bottom bar)
    if auto_judge:
        judge_pbar = tqdm(total=0, desc="Judge Tasks", position=1)

    for iteration in range(CONFIG["iters"]):  # Outer loop for iterations
        for obs_index, item in enumerate(dataset[:CONFIG["obs"]]):  # Loop over observations
            source_papers_list = item.get("source_papers", [])

            for ablation_name, ablation_config in ablation_classes.items():  # Loop over test cases
                # Load SOP template file for the current ablation
                sop_template_file = ablation_config.get("sop_template_file", "")
                template_variants = []

                if sop_template_file:
                    try:
                        if sop_template_file.lower().endswith(".json"):
                            sop_template_obj = load_sop_template(sop_template_file)
                            sop_content = json.dumps(sop_template_obj, indent=2, ensure_ascii=False)

                            # Define SOP variants
                            template_variants = [
                                ("json_original", sop_content),
                                ("json_with_grounding_prefix", f"Treat the following JSON as a grounding object:\n{sop_content}")
                            ]
                        else:
                            with open(sop_template_file, "r", encoding="utf-8") as f:
                                sop_content = f.read()
                            template_variants = [("txt_original", sop_content)]
                    except FileNotFoundError:
                        print(f"--- WARNING --- \nSOP template file not found for '{ablation_name}': '{sop_template_file}'. Skipping this ablation.")
                        continue
                    except Exception as e:
                        print(f"--- WARNING --- \nError loading SOP template for '{ablation_name}': {e}. Skipping.")
                        continue
                else:
                    # No template provided (e.g., control). Use a single empty/text-only variant.
                    template_variants = [("no_sop", "")]

                for variant_suffix, variant_sop_content in template_variants:  # Loop over SOP variants
                    # Generate a unique problem_id for each iteration
                    problem_id = f"{item['problem_id']}-iter{iteration + 1}"

                    # Build the prompt and process as usual
                    prompt_str = (
                        f"{variant_sop_content}\n\n---\n"
                        f"**USER'S QUESTION:**\n{item['question']}\n\n---\n\n"
                        "Based on the instructions in the JSON object, synthesize an answer to the user's question."
                    )

                    # Call the model and process the response
                    if ablation_config["api_type"] == "ollama":
                        response_data = generate_ollama_response(ablation_config["model_name"], prompt_str)
                    else:
                        response_data = generate_google_response(ablation_config["model_name"], prompt_str)

                    # Extract raw generation for judge
                    raw_generation_for_judge = response_data.get('raw_generation', "")

                    # Write the result to the CSV
                    result_entry = {
                        'problem_id': problem_id,
                        'test_name': f"{ablation_name}::{variant_suffix}",
                        'ablation_condition': ablation_config.get("description", ""),
                        'model_name': ablation_config["model_name"],
                        'model_version': "",
                        'api_type': ablation_config["api_type"],
                        'api_version': "",
                        'sop_template_file': sop_template_file,
                        'prompt_used': prompt_str,
                        'generation_parameters': json.dumps(API_PARAMETERS),
                        'raw_model_output': raw_generation_for_judge,
                        'source_papers': ", ".join(item.get('source_papers', [])),
                        'question': item['question'],
                        'error': response_data.get('error'),
                        'api_latency_ms': response_data.get('api_latency_ms'),
                        'judge_prompt': f"[DOCUMENT_CACHE_ID:{judge_prompt_cache_id}]\n\n" if judge_prompt_cache_id else "",
                        'raw_judge_response': "",
                        'score_faithfulness_rating': None,
                        'score_faithfulness_justification': "",
                        'score_correctness_rating': None,
                        'score_correctness_justification': "",
                        'score_completeness_rating': None,
                        'score_completeness_justification': "",
                        'score_clarity_rating': None,
                        'score_clarity_justification': "",
                        'score_citation_following_rating': None,
                        'score_citation_following_justification': ""
                    }

                    write_result_row_to_csv(log_filename, result_entry, results_columns)

                    # Assign the judge prompt to the task
                    judge_prompt_for_task = judge_prompt_template

                    # Submit the task to the thread pool
                    if auto_judge:
                        submit_time = time.time()
                        fut = judge_executor.submit(
                            _run_judge_and_update,
                            problem_id,
                            result_entry['test_name'],
                            item['question'],
                            raw_generation_for_judge,
                            judge_prompt_for_task  # Pass explicitly
                        )
                        with future_lock:
                            future_to_meta[fut] = (problem_id, result_entry['test_name'], submit_time)
                        fut.add_done_callback(_on_judge_done)

                        # Update judge progress bar total to reflect a new outstanding task (no stdout)
                        try:
                            judge_pbar.total += 1
                            judge_pbar.refresh()
                        except Exception:
                            pass

                        # If too many outstanding judges, wait for some to finish before continuing
                        if len(future_to_meta) >= JUDGE_QUEUE_CAP:
                            print(f"[JUDGE] Reached queue cap ({JUDGE_QUEUE_CAP}). Waiting for at least one judge to finish...")
                            # Wait for first completed future (non-blocking uses small timeout)
                            done, _ = concurrent.futures.wait(list(future_to_meta.keys()), timeout=5, return_when=concurrent.futures.FIRST_COMPLETED)
                            # Give GC a chance after clearing references
                            gc.collect()

                    # Update the generator progress bar
                    generator_pbar.update(1)

    # Close the progress bars at the end of the program
    generator_pbar.close()
    if auto_judge:
        judge_pbar.close()

    # at program end, ensure executor shutdown and close HTTP session
    try:
        if auto_judge:
            print(f"[JUDGE] Waiting for outstanding judge tasks ({len(future_to_meta)}) to finish...")
            judge_executor.shutdown(wait=True)
    finally:
        # shutdown shared short-call executor
        try:
            SHORT_CALL_EXECUTOR.shutdown(wait=False)
        except Exception:
            pass

        try:
            HTTP_SESSION.close()
        except Exception:
            pass
        gc.collect()
        try:
            if auto_judge:
                judge_pbar.close()
        except Exception:
            pass

    print("\n--- Experiment Complete ---")
    print("\n--- Final Score Summary ---")
    final_df = pd.read_csv(log_filename, comment="#")
    score_cols = [
        'score_faithfulness_rating', 'score_correctness_rating',
        'score_completeness_rating', 'score_clarity_rating',
        'score_citation_following_rating'
    ]
    for col in score_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
    summary = final_df.groupby('test_name')[score_cols].mean()
    print(summary)
    print(f"\n✅ Full, detailed results have been saved to '{log_filename}'")

if __name__ == "__main__":
    main()
