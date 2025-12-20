from typing import Iterable, Sequence

from ..imports import try_import
from ..logging import get_logger
from .base import GenerationRequest, GenerationResult, Message, Model


logger = get_logger(__name__)

with try_import() as _imports:
    from langchain.chat_models.base import BaseChatModel
    from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

_imports.check()


def _into_langchain_message(message: Message) -> BaseMessage:
    if message.role == "system":
        return SystemMessage(content=message.content)
    elif message.role == "user":
        return HumanMessage(content=message.content)
    elif message.role == "assistant":
        return AIMessage(content=message.content)
    elif message.role == "assistant_prefill":
        logger.warning(
            "Langchain does not support assistant_prefill role. Using assistant role instead. "
            "This might cause unexpected behavior."
        )
        return AIMessage(content=message.content)
    else:
        raise ValueError(f"Unknown role: {message.role}")


class LangChainModel(Model):

    def __init__(self, langchain_model: BaseChatModel) -> None:
        self.langchain_model = langchain_model

    def generate(self, requests: Sequence[GenerationRequest]) -> Iterable[GenerationResult]:
        langchain_messages = [
            [_into_langchain_message(message) for message in request.messages]
            for request in requests
        ]

        # `BaseChatModel.batch` always raises mypy type errors because its type hint is invariant.
        # TODO: Remove `type:ignore` when the type hint of `BaseChatModel.batch` is fixed.
        responses = self.langchain_model.batch(langchain_messages)  # type: ignore

        results: list[GenerationResult] = []
        for request, response in zip(requests, responses):
            assert isinstance(response.content, str)
            generation: str = response.content
            results.append(GenerationResult(request=request, generation=generation))

        return results
