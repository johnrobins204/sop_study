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