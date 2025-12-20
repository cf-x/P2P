from typing import Optional, Sequence

from ..models import Message, Model, NLLRequest
from .base import Task, TaskResult


class NLLTask(Task):

    def __init__(self, samples: Sequence[Sequence[Message]]) -> None:
        self.samples = [list(messages) for messages in samples]

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def evaluate(self, model: Model, sample_ids: Optional[Sequence[int]] = None) -> TaskResult:
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        requests = [NLLRequest(messages=self.samples[i]) for i in sample_ids]

        results = model.nll(requests)

        sum_nll = 0.0
        num_considered_tokens = 0
        for result in results:
            sum_nll += result.sum_nll
            num_considered_tokens += result.num_considered_tokens

        return TaskResult(
            aggregate_metrics={
                "mean_nll": sum_nll / num_considered_tokens,
                "num_considered_tokens": num_considered_tokens,
            },
            sample_details=[
                {"mean_nll": result.sum_nll, "num_tokens": result.num_considered_tokens}
                for result in results
            ],
        )
