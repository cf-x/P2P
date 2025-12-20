import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from ...logging import get_logger
from ...models.base import GenerationResult, Model
from ..base import Task, TaskResult
from .config import CS_BENCH_FORMATS, CSBenchFormat, CSBenchTaskConfig
from .data import CSBenchSample, load_dataset


logger = get_logger(__name__)

ASSERTION_PAT = re.compile(r"\b(true|false)\b")
MULTICHOICE_PAT = re.compile(r"\b[ABCD]\b", re.IGNORECASE)


class CSBenchTask(Task):
    """CS-Bench: https://github.com/csbench/csbench"""

    def __init__(
        self, samples: Sequence[CSBenchSample], config: Optional[CSBenchTaskConfig] = None
    ) -> None:
        self.samples = list(samples)

        if config is None:
            config = CSBenchTaskConfig()
        self.num_shots = config.num_shots

        self.fewshot_dataset = self._get_fewshot_samples(
            load_dataset(
                hf_repo=config.fewshot_hf_repo, hf_data_files=config.fewshot_hf_data_files
            ),
            num_samples=self.num_shots,
        )

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def evaluate(
        self,
        model: Model,
        sample_ids: Optional[Sequence[int]] = None,
    ) -> TaskResult:
        """
        To avoid running GPT-4 inside the task, it only evaluates
        "Assertion" and "Multiple-choice" task format,
        and ignore "Fill-in-the-blank" and "Open-ended" task format.
        """
        samples = (
            [sample for sample in self.samples if sample.index in set(sample_ids)]
            if sample_ids is not None
            else self.samples
        )
        samples = self._filter_formats(samples)

        outputs = model.generate(
            [sample.to_request(self.fewshot_dataset[sample.format]) for sample in samples]
        )

        is_correct_ls = []
        sample_details: List[Dict[str, Any]] = []
        for sample, output in zip(samples, outputs):
            is_correct = self._model_is_correct(sample, output)
            is_correct_ls.append(is_correct)
            sample_details.append(
                dict(
                    index=sample.index,
                    is_correct=is_correct,
                    answer=sample.answer,
                    model_output=output.generation,
                    format=sample.format,
                )
            )

        return TaskResult(
            aggregate_metrics=self._calc_accuracies(samples, is_correct_ls),
            sample_details=sample_details,
        )

    def _model_is_correct(self, sample: CSBenchSample, output: GenerationResult) -> bool:
        """
        Uses regex match with word boundary, following official CS bench repo:
        https://github.com/csbench/csbench/blob/4b8be69c4b915e4d8cd69825c122c710c6c651b8/gen_judgment.py#L18-L24
        """
        generation = output.generation
        answer = sample.answer
        if isinstance(answer, bool):
            answer = "True" if answer else "False"
        match sample.format:
            case "Assertion":
                match = ASSERTION_PAT.search(generation.lower())
                if match:
                    return match.group(0) == answer.lower()
                else:
                    return False
            case "Multiple-choice":
                matches = MULTICHOICE_PAT.findall(generation)
                if matches:
                    fst_match: str = matches[0]
                    return fst_match.lower() == answer.lower()
                else:
                    return False
            case _:
                raise RuntimeError(
                    f"Internal error: sample format, {sample.format}, is not correctly filtered"
                )

    def _calc_accuracies(
        self, samples: List[CSBenchSample], is_correct_ls: List[bool]
    ) -> Dict[str, float]:
        """Calculate accuracies for each domain/subdomain/format/tag"""
        assert len(samples) == len(
            is_correct_ls
        ), "Length of samples and is_correct_ls should be same."

        accuracies: Dict[str, float] = defaultdict(int)
        counts: Dict[str, int] = defaultdict(int)
        for sample, is_correct in zip(samples, is_correct_ls):
            counts["acc_total"] += 1
            if is_correct:
                accuracies["acc_total"] += 1

            for category_type in ("domain", "sub_domain", "format", "tag"):
                category_name = getattr(sample, category_type)
                metric_name = f"acc_{category_type}_{category_name}".replace(" ", "_").lower()
                counts[metric_name] += 1
                if is_correct:
                    accuracies[metric_name] += 1

        for metric_name in counts:
            accuracies[metric_name] /= counts[metric_name]

        # convert defaultdict to plain dict to avoid unexpected behavior on user's side
        return dict(accuracies)

    def _filter_formats(self, samples: List[CSBenchSample]) -> List[CSBenchSample]:
        filtered_samples = list(
            sample
            for sample in samples
            if sample.format not in ["Fill-in-the-blank", "Open-ended"]
        )
        if len(filtered_samples) < len(samples):
            logger.info(
                (
                    "Unsupported task formats 'Fill-in-the-blank', "
                    f"'Open-ended' are ignored for {len(samples) - len(filtered_samples)} tasks."
                )
            )

        return filtered_samples

    def _get_fewshot_samples(
        self, samples: Sequence[CSBenchSample], num_samples: int
    ) -> Dict[CSBenchFormat, List[CSBenchSample]]:
        sample_by_fmt: Dict[CSBenchFormat, List[CSBenchSample]] = {
            fmt: [] for fmt in CS_BENCH_FORMATS
        }
        for sample in samples:
            if len(sample_by_fmt[sample.format]) < num_samples:
                sample_by_fmt[sample.format].append(sample)

        err_msg = ""
        for fmt, samples in sample_by_fmt.items():
            if len(samples) != num_samples:
                err_msg += (
                    f"Insufficient number of samples ({len(samples)}) for "
                    f"format {fmt} in fewshot dataset. Need {num_samples} samples at least!\n"
                )
        if len(err_msg) > 0:
            raise RuntimeError(err_msg)
        return sample_by_fmt
