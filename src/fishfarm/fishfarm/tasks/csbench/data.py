from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, TypeAlias, get_args

import datasets

from ...models.base import GenerationRequest, Message
from .config import CSBenchDomain, CSBenchFormat, CSBenchSubDomain, CSBenchTag


ProbChoice: TypeAlias = Literal["A", "B", "C", "D"]
prob_choices: Tuple[ProbChoice, ...] = get_args(ProbChoice)


@dataclass
class CSBenchSample:
    """
    Users can optionally provide their own prompt by specifying `request` field.
    If not specified, the prompts in CS-Bench paper will be used.

    For `Assertion` problem format, answer should be bool, or should be str "True" or "False".
    """

    index: int

    question: str
    answer: str | bool

    domain: CSBenchDomain
    sub_domain: CSBenchSubDomain
    format: CSBenchFormat
    tag: CSBenchTag

    # Present only for "Multiple-choice" format
    choice_desc: Optional[Dict[ProbChoice, str]] = None

    request: Optional[GenerationRequest] = None

    def to_request(self, fewshots: Optional[List["CSBenchSample"]] = None) -> GenerationRequest:
        """If user provides their own request, that request will be used"""
        if self.request is None:
            self.request = self._get_instruction_request(fewshots)
        return self.request

    def _problem_prompt(self) -> str:
        question = self.question
        match self.format:
            case "Multiple-choice":
                choice_desc = self.choice_desc
                assert (
                    choice_desc is not None
                ), "choice_disc for format type `Multipe-choice` should not be empty."
                return (
                    f"Question:{question}\n"
                    "Options:\n"
                    f"(A){choice_desc['A']}\n"
                    f"(B){choice_desc['B']}\n"
                    f"(C){choice_desc['C']}\n"
                    f"(D){choice_desc['D']}\n"
                    "Please provide the answer to this question directly (a single letter):"
                )
            case "Assertion":
                return f"Statement:{question}" "Please give the answer directly (true or false):"
            case "Fill-in-the-blank":
                return f"Question:{question}Answer:"
            case "Open-ended":
                return (
                    f"This is a subjective question:{question}"
                    "Please provide a brief answer to this question:"
                )

    def _answer_prompt(self) -> str:
        answer = self.answer
        if isinstance(answer, bool):
            answer = "true" if answer else "false"
        match self.format:
            case "Multiple-choice":
                return answer
            case "Assertion":
                return answer.lower()
            case "Fill-in-the-blank":
                return answer
            case "Open-ended":
                raise RuntimeError("Answer for Open-ended format is not supported.")

    def _get_instruction_request(
        self, fewshots: Optional[List["CSBenchSample"]]
    ) -> GenerationRequest:
        """
        Generate request w/ or w/o fewshots prompt.

        Fill-in-the-blank and Open-ended problems are not used by CSBenchTask, but provided here for completeness.
        https://github.com/csbench/csbench/blob/4b8be69c4b915e4d8cd69825c122c710c6c651b8/vllm-main/examples/csbench/gen_model_answer_en.py#L41-L54 # noqa: E501
        """
        qnas = [self]
        if fewshots is not None:
            qnas = fewshots + qnas
            assert all(
                qna.format == qnas[0].format for qna in qnas
            ), "All the fewshots should have the same format"

        messages = []
        for i, qna in enumerate(qnas):
            message = ""
            if i == 0:
                match self.format:
                    case "Multiple-choice":
                        message = (
                            "This is a multiple-choice question. "
                            "Please read the question carefully and choose the correct answer. "
                        )
                    case "Assertion":
                        message = (
                            "This is a true/false question. "
                            "Please determine whether the following statement is true or false. "
                        )
                    case "Fill-in-the-blank":
                        message = (
                            "You are a professor proficient in computer science. "
                            "This is a fill-in-the-blank question. "
                        )
                    case "Open-ended":
                        message = ""

            message += qna._problem_prompt()
            messages.append(Message(role="user", content=message))

            if i != len(qnas) - 1:
                messages.append(Message(role="assistant", content=qna._answer_prompt()))

        return GenerationRequest(messages)


def load_dataset(
    hf_repo: str = "SakanaAI/CS-Bench",
    hf_data_files: str = "strfied_en_test.jsonl",
) -> Sequence[CSBenchSample]:
    dataset = datasets.load_dataset(hf_repo, data_files=hf_data_files, split="train")

    samples = []
    for row in dataset:
        sample = dict()
        sample["index"] = row["ID"]
        sample["question"] = row["Question"]
        sample["answer"] = row["Answer"]
        sample["domain"] = row["Domain"]
        sample["sub_domain"] = row["SubDomain"]
        sample["format"] = row["Format"]
        sample["tag"] = row["Tag"]

        if all(choice in row for choice in prob_choices):
            sample["choice_desc"] = {choice: row[choice] for choice in prob_choices}
        else:
            assert not any(choice in row for choice in prob_choices), (
                f"The problem chioces {', '.join(prob_choices)} "
                "should be either all-present or all-absent"
            )

        samples.append(CSBenchSample(**sample))

    return samples
