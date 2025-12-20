import subprocess
from multiprocessing import Process, Queue
from time import sleep
from typing import Optional, Sequence, Union

from tqdm import tqdm

from ...imports import try_import
from ...models import Model
from ...models.base import GenerationRequest, Message, Role
from ..base import Task, TaskResult
from .utils import messages_to_str


with try_import() as _imports:
    from agentenv.envs.sqlgym import SqlGymEnvClient

_imports.check()


def start_server(port: int, bird_path: str, verbose: bool = False) -> subprocess.Popen:
    command = f"""source ~/.bashrc && \
source $(conda info --base)/etc/profile.d/conda.sh && \
conda activate agentenv-sqlgym && \
export AGENTENV_SQLGYM_BIRD_PATH={bird_path} && \
sqlgym --host 127.0.0.1 --port {port}
"""
    if not verbose:
        with open("/dev/null", "w") as f:
            return subprocess.Popen(
                command, shell=True, executable="/bin/bash", stdout=f, stderr=f
            )
    return subprocess.Popen(command, shell=True, executable="/bin/bash")


def worker(
    port: int,
    bird_path: str,
    task_queue: Queue,
    result_queue: Queue,
    timeout: Optional[int] = None,
) -> None:
    process = start_server(port, bird_path)
    sleep(2)  # wait for the server to start
    assert process.poll() is None, f"The server on port {port} did not start properly."
    # Note: data_len is a legacy argument in AgentGym so we do not use it.
    client = SqlGymEnvClient(f"http://127.0.0.1:{port}", data_len=None, timeout=timeout)
    try:
        last_idx = None
        while True:
            task = task_queue.get()
            if task is None:
                break
            sample_idx, action = task
            if action == "RESET":
                obs, _ = client.reset(sample_idx)
                result_queue.put({"sample_idx": sample_idx, "state": obs})
            else:
                if last_idx != sample_idx:
                    client.reset(sample_idx)
                step_output = client.step(action)
                state, reward, done = step_output.state, step_output.reward, step_output.done
                result_queue.put(
                    {"sample_idx": sample_idx, "state": state, "reward": reward, "done": done}
                )
            last_idx = sample_idx
    finally:
        process.terminate()


class SqlGymMultiClientTask(Task):
    """
    This task will spawn n_workers processes and launch a sqlgym server for each of them.

    The workers, and their servers, will remain active until a user calls task.terminate().
    """

    def __init__(
        self,
        bird_path: str,
        data_indices: Union[int, list[int]],
        n_workers: int = 1,
        timeout: Optional[int] = None,
    ):

        translate_roles: dict[str, Role] = {"gpt": "assistant", "human": "user"}
        self.conversation_start = [
            Message(role=translate_roles[m["from"]], content=m["value"])
            for m in SqlGymEnvClient.conversation_start
        ]

        if isinstance(data_indices, int):
            data_indices = list(range(data_indices))

        self.data_indices = data_indices
        self.task_queue: Queue = Queue()
        self.result_queue: Queue = Queue()
        # Note: data_len is a legacy argument in AgentGym so we do not use it.
        self.workers = [
            Process(
                target=worker,
                args=(36002 + i, bird_path, self.task_queue, self.result_queue, timeout),
            )
            for i in range(n_workers)
        ]
        for w in self.workers:
            w.start()
        sleep(2)  # wait for the servers to start
        for w in self.workers:
            if not w.is_alive():
                self.terminate()
                raise RuntimeError("One of the workers did not start properly.")

    def terminate(self) -> None:
        for _ in self.workers:
            self.task_queue.put(None)
        for w in self.workers:
            w.join()

    @property
    def num_samples(self) -> int:
        return len(self.data_indices)

    def evaluate(self, model: Model, sample_ids: Optional[Sequence[int]] = None) -> TaskResult:
        if sample_ids is None:
            sample_ids = self.data_indices

        # reset all environments
        for idx in sample_ids:
            self.task_queue.put((idx, "RESET"))

        # collect initial state
        requests: list[GenerationRequest] = []
        indices = []
        for _ in range(len(sample_ids)):
            result = self.result_queue.get()
            idx = result["sample_idx"]
            state = result["state"]
            indices.append(result["sample_idx"])
            requests.append(
                GenerationRequest(
                    messages=self.conversation_start + [Message(role="user", content=state)]
                )
            )

        # generate actions
        reward_sum = 0.0
        gen_results = {}
        for sample_idx, gen_result in zip(indices, model.generate(requests)):
            gen_results[sample_idx] = gen_result
            self.task_queue.put((sample_idx, gen_result.generation))

        # evaluate
        sample_details = []
        for _ in tqdm(range(len(sample_ids)), desc="Evaluating"):
            result = self.result_queue.get()
            sample_idx, state, reward, done = [
                result[k] for k in ["sample_idx", "state", "reward", "done"]
            ]
            assert done, "The episode should end after one step."
            gen_result = gen_results[sample_idx]
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
