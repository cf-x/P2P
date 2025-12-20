from dataclasses import asdict, dataclass
from typing import Optional, Sequence

from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

from ..models import GenerationRequest, Message, Model
from .base import Task, TaskResult


@dataclass
class RougeSample:

    prompt: str
    response: str


@dataclass
class RougeScorerConfig:
    # init args for rouge_scorer.RougeScorer, see the following for details:
    # https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py#L50
    rouge_types: Sequence[str] = ("rouge1", "rougeL")
    use_stemmer: bool = False
    split_summaries: bool = False
    tokenizer: Tokenizer | None = None


class RougeTask(Task):

    def __init__(
        self,
        samples: Sequence[RougeSample],
        rouge_scorer_config: RougeScorerConfig,
        context_messages: Sequence[Message] = (),
    ) -> None:
        self.samples = list(samples)
        self.scorer = rouge_scorer.RougeScorer(**asdict(rouge_scorer_config))
        self.context_messages = context_messages
        self.rouge_types = rouge_scorer_config.rouge_types

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def evaluate(self, model: Model, sample_ids: Optional[Sequence[int]] = None) -> TaskResult:
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]
        requests = []
        for sample in samples:
            messages = list(self.context_messages)
            messages.append(Message(role="user", content=sample.prompt))
            requests.append(GenerationRequest(messages=messages))

        sample_details = []
        for sample, result in zip(samples, model.generate(requests)):
            output = result.generation
            scores = self.scorer.score(sample.response, output)
            details = dict(problem=sample.prompt, output=output, response=sample.response)
            for rouge_type in self.rouge_types:
                details[f"{rouge_type}_precision"] = scores[rouge_type].precision
                details[f"{rouge_type}_recall"] = scores[rouge_type].recall
                details[f"{rouge_type}_fmeasure"] = scores[rouge_type].fmeasure
            sample_details.append(details)

        perf_keys = [
            f"{rouge_type}_{metric}"
            for rouge_type in self.rouge_types
            for metric in ["precision", "recall", "fmeasure"]
        ]

        agg_metrics = dict()
        for k in perf_keys:
            agg_metrics[k] = sum([float(sd[k]) for sd in sample_details]) / len(sample_details)

        return TaskResult(aggregate_metrics=agg_metrics, sample_details=sample_details)
