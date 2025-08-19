from dataclasses import dataclass, field

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
    """Represents a single, complete experimental trial."""
    problem_id: str
    test_name: str
    question: str
    source_text: str
    ablation_condition: str
    model_response: ModelResponse | None = None
    judge_evaluation: JudgeEvaluation | None = None

    def to_csv_row(self, columns: list[str]) -> dict:
        """
        Flattens the Trial object into a dictionary suitable for CSV writing,
        dynamically handling all judge evaluation scores.
        """
        row = {
            'problem_id': self.problem_id,
            'test_name': self.test_name,
            'ablation_condition': self.ablation_condition,
            'raw_model_output': self.model_response.raw_text,
            'question': self.question,
            'error': self.model_response.error,
            'api_latency_ms': self.model_response.api_latency_ms,
            'source_text': self.source_text,
            'raw_judge_response': self.judge_evaluation.raw_judge_response,
            'judge_error': self.judge_evaluation.judge_error
        }

        # Unpack all scores from the judge_evaluation object
        if self.judge_evaluation and self.judge_evaluation.scores:
            for criterion, details in self.judge_evaluation.scores.items():
                # Ensure details is a dictionary before accessing keys
                if isinstance(details, dict):
                    row[f'score_{criterion}_rating'] = details.get('rating')
                    row[f'score_{criterion}_justification'] = details.get('justification')

        # Ensure all columns from the header are present in the final row dict,
        # filling missing ones with empty strings.
        final_row = {col: row.get(col, '') for col in columns}
        return final_row

    def __repr__(self):
        return (f"Trial(problem_id={self.problem_id!r}, test_name={self.test_name!r}, "
                f"question={self.question!r}, source_text={self.source_text!r}, "
                f"ablation_condition={self.ablation_condition!r}, "
                f"model_response={self.model_response!r}, "
                f"judge_evaluation={self.judge_evaluation!r})")