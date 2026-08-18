import argparse
import json
import math
import os
import tempfile
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn

from sample_factory.utils.typing import PolicyID
from sample_factory.utils.utils import log

PROGRESS_STATE_FILENAME = ".training_progress.json"
TERMINAL_STATUSES: Tuple[str, ...] = ("success", "failure", "interrupted")


def progress_state_path(experiment_dir: str) -> str:
    return os.path.join(experiment_dir, PROGRESS_STATE_FILENAME)


class TrainingProgressWriter:
    def __init__(self, experiment_dir: str, num_policies: int, total_env_steps: int):
        self.state_path = progress_state_path(experiment_dir)
        self.num_policies = num_policies
        self.total_env_steps = total_env_steps
        self._disabled = total_env_steps <= 0
        self._warned = False

    def update(self, env_steps: Mapping[PolicyID, int], status: str = "running") -> None:
        if self._disabled:
            return

        state = {
            "version": 1,
            "status": status,
            "total_env_steps": self.total_env_steps,
            "env_steps": {
                str(policy_id): min(max(int(env_steps.get(policy_id, 0)), 0), self.total_env_steps)
                for policy_id in range(self.num_policies)
            },
            "updated_at": time.time(),
        }
        temp_path = None
        try:
            descriptor, temp_path = tempfile.mkstemp(
                prefix=f"{PROGRESS_STATE_FILENAME}.",
                dir=os.path.dirname(self.state_path),
                text=True,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as state_file:
                json.dump(state, state_file, separators=(",", ":"), sort_keys=True)
                state_file.write("\n")
            os.replace(temp_path, self.state_path)
            temp_path = None
        except Exception as error:
            self._disabled = True
            if not self._warned:
                self._warned = True
                log.warning("Training progress state disabled: %s", error)
        finally:
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


def read_progress_state(experiment_dir: str) -> Optional[Dict[str, Any]]:
    try:
        with open(progress_state_path(experiment_dir), "r", encoding="utf-8") as state_file:
            state = json.load(state_file)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if not isinstance(state, Mapping):
        return None

    total_env_steps = state.get("total_env_steps")
    env_steps = state.get("env_steps")
    if (
        state.get("status") not in ("running", *TERMINAL_STATUSES)
        or type(state.get("version")) is not int
        or state.get("version") != 1
        or type(total_env_steps) is not int
        or total_env_steps <= 0
        or not isinstance(env_steps, Mapping)
        or not env_steps
        or not _is_finite_number(state.get("updated_at"))
    ):
        return None

    policy_ids = {str(policy_id) for policy_id in range(len(env_steps))}
    if set(env_steps) != policy_ids:
        return None

    for policy_id in policy_ids:
        completed_steps = env_steps[policy_id]
        if type(completed_steps) is not int or not 0 <= completed_steps <= total_env_steps:
            return None
    return state


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def update_progress_tasks(progress: Progress, task_ids: Dict[int, TaskID], state: Mapping[str, Any]) -> None:
    total = int(state["total_env_steps"])
    for policy_key, completed in sorted(state["env_steps"].items(), key=lambda item: int(item[0])):
        policy_id = int(policy_key)
        if policy_id not in task_ids:
            task_ids[policy_id] = progress.add_task("training", total=total, policy_id=policy_id)
        progress.update(task_ids[policy_id], completed=int(completed), total=total, refresh=False)


def view_progress(experiment_dir: str, poll_interval: float = 0.5, console: Optional[Console] = None) -> int:
    console = console or Console(highlight=False)
    progress = Progress(
        TextColumn("Policy {task.fields[policy_id]}"),
        BarColumn(),
        TextColumn("{task.percentage:>5.1f}%"),
        TextColumn("{task.completed:,.0f}/{task.total:,.0f} steps"),
        console=console,
        refresh_per_second=2,
        transient=False,
        expand=True,
    )
    task_ids: Dict[int, TaskID] = {}
    last_updated_at = None
    state = None

    with console.status(f"Waiting for training progress in {experiment_dir}"):
        while state is None:
            state = read_progress_state(experiment_dir)
            if state is None:
                time.sleep(poll_interval)

    with progress:
        while True:
            if state.get("updated_at") != last_updated_at:
                update_progress_tasks(progress, task_ids, state)
                progress.refresh()
                last_updated_at = state.get("updated_at")
            if state.get("status") in TERMINAL_STATUSES:
                return 0
            time.sleep(poll_interval)
            next_state = read_progress_state(experiment_dir)
            if next_state is not None:
                state = next_state


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Show Sample Factory policy training progress")
    parser.add_argument("experiment_dir")
    args = parser.parse_args(argv)
    return view_progress(args.experiment_dir)


if __name__ == "__main__":
    raise SystemExit(main())
