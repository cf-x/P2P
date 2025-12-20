from typing import Sequence

from ...models.base import Message


def messages_to_str(messages: Sequence[Message]) -> str:
    result = ""
    for m in messages:
        result += f"{m.role}:\n{m.content}\n"
    return result
