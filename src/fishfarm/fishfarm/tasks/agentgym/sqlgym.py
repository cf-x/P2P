from typing import Optional, Sequence, Union

from ...imports import try_import
from ...models import Model
from ...models.base import GenerationRequest, Message, Role
from ..base import Task, TaskResult
from .utils import messages_to_str


with try_import() as _imports:
    from agentenv.envs.sqlgym import SqlGymEnvClient

_imports.check()


class SqlGymTask(Task):
    def __init__(
        self,
        data_indices: Union[int, list[int]],
        env_server_base: str,
        timeout: Optional[int] = None,
    ):

        translate_roles: dict[str, Role] = {
            "gpt": "assistant",
            "human": "user",
        }
        self.conversation_start = [
            Message(role=translate_roles[m["from"]], content=m["value"])
            for m in SqlGymEnvClient.conversation_start
        ]

        if isinstance(data_indices, int):
            data_indices = list(range(data_indices))

        self.data_indices = data_indices
        # Note: data_len is a legacy argument in AgentGym so we do not use it.
        self.client = SqlGymEnvClient(env_server_base, data_len=None, timeout=timeout)

    @property
    def num_samples(self) -> int:
        return len(self.data_indices)

    def evaluate(self, model: Model, sample_ids: Optional[Sequence[int]] = None) -> TaskResult:
        if sample_ids is None:
            sample_ids = self.data_indices

        requests = []
        for idx in sample_ids:
            conversation = self.conversation_start + [
                Message(role="user", content=self.client.reset(idx)[0])
            ]
            requests.append(GenerationRequest(messages=conversation))

        # generate
        reward_sum = 0.0
        sample_details = []
        for sample_idx, gen_result in zip(sample_ids, model.generate(requests)):
            self.client.reset(sample_idx)
            step_output = self.client.step(gen_result.generation)
            state, reward, done = step_output.state, step_output.reward, step_output.done
            assert done, "The episode should end after one step."
            sample_details.append(
                {
                    "prompt": messages_to_str(gen_result.request.messages),
                    "sample_idx": sample_idx,
                    "generation": gen_result.generation,
                    "state": state,
                    "reward": reward,
                }
            )
            reward_sum += reward

        aggregate_metrics = {"avg_reward": reward_sum / len(sample_ids)}
        return TaskResult(aggregate_metrics=aggregate_metrics, sample_details=sample_details)
