import os
import sys
import time
import json
import requests
import logging
from abc import ABC, abstractmethod

import google.generativeai as genai

# Use a relative import to access other modules within the same package
from .data_structures import ModelResponse

# --- Caching for Ollama Context ---
# This simple cache helps avoid redundant context passing in prompts
CONTEXT_CACHE = {}

class LanguageModel(ABC):
    """Abstract base class for language models."""
    def __init__(self, model_name: str, api_parameters: dict, config: dict):
        self.model_name = model_name
        self.api_parameters = api_parameters
        self.config = config

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
            raw_text = response.text
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason.name if hasattr(candidate, 'finish_reason') else "UNKNOWN"
            safety_ratings = {r.category.name: r.probability.name for r in candidate.safety_ratings}
            return ModelResponse(
                raw_text=raw_text, api_latency_ms=int(latency),
                finish_reason=finish_reason, safety_ratings=safety_ratings
            )
        except ValueError:
            # Handle cases where response.text fails (e.g., blocked response)
            return ModelResponse(error="Response blocked or empty.", api_latency_ms=int((time.time() - start_time) * 1000))
        except Exception as e:
            logging.error(f"Error in GoogleModel for {self.model_name}: {e}")
            return ModelResponse(error=str(e), api_latency_ms=int((time.time() - start_time) * 1000))

class OllamaModel(LanguageModel):
    """LanguageModel implementation for a local Ollama model."""
    def generate(self, prompt: str, timeout: int = 120) -> ModelResponse:
        start_time = time.time()
        url = f"{self.config.get('ollama_base_url', 'http://localhost:11434')}/api/generate"
        payload = {
            "model": self.model_name, "prompt": prompt, "stream": False,
            "options": {
                "temperature": self.api_parameters.get("temperature"),
                "num_predict": self.api_parameters.get("max_tokens")
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            response_json = response.json()
            return ModelResponse(
                raw_text=response_json.get("response", ""),
                api_latency_ms=int((time.time() - start_time) * 1000)
            )
        except requests.exceptions.RequestException as e:
            logging.error(f"Ollama API request failed for {self.model_name}: {e}")
            return ModelResponse(error=str(e), api_latency_ms=int((time.time() - start_time) * 1000))

def get_model_instance(model_identifier: str, config: dict, api_params: dict) -> LanguageModel:
    """Factory function to get the correct model class instance."""
    # A more robust way to check for an Ollama model
    if "ollama" in model_identifier.lower() or model_identifier in config.get("ollama_models", []):
        print(f"INFO: Identified '{model_identifier}' as an Ollama model.")
        # The model_name for Ollama is the identifier itself.
        return OllamaModel(model_identifier, api_params, config)
    else:
        print(f"INFO: Identified '{model_identifier}' as a Google model.")
        return GoogleModel(model_identifier, api_params, config)

    def _build_prompt_for_criterion(self, criterion_name: str, question: str, answer: str, source_text_key: str) -> str:
        """Builds a prompt for the given criterion, question, and answer."""
        source_text = self.config.get("source_texts", {}).get(source_text_key, "")
        template_path = os.path.join(
            self.config.get("template_dir", "."), f"{criterion_name}_template.json"
        )
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