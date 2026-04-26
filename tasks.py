from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

# Evidence-card counts drive easy / medium / hard curricula.


@dataclass(frozen=True)
class Task:
    id: str
    difficulty: str
    num_evidence: int
    max_steps: int
    description: str


TASKS: Dict[str, Task] = {
    "easy": Task(
        id="easy",
        difficulty="easy",
        num_evidence=3,
        max_steps=12,
        description="3 evidence items — small branching factor for smoke tests and curriculum start.",
    ),
    "medium": Task(
        id="medium",
        difficulty="medium",
        num_evidence=5,
        max_steps=20,
        description="5 evidence items — default difficulty for baseline comparisons.",
    ),
    "hard": Task(
        id="hard",
        difficulty="hard",
        num_evidence=8,
        max_steps=32,
        description="8 evidence items — longer horizon, larger combinatorial action space.",
    ),
}


def get_task(task_id: str) -> Task:
    if task_id not in TASKS:
        raise KeyError(
            f"Unknown task {task_id!r}. Valid keys: {', '.join(sorted(TASKS))}."
        )
    return TASKS[task_id]


def _episode_return(transcript: dict[str, Any], task: Task) -> float:
    return float(transcript.get("cumulative_reward", 0.0))


def grade_easy(transcript: dict[str, Any]) -> float:
    return _episode_return(transcript, TASKS["easy"])


def grade_medium(transcript: dict[str, Any]) -> float:
    return _episode_return(transcript, TASKS["medium"])


def grade_hard(transcript: dict[str, Any]) -> float:
    return _episode_return(transcript, TASKS["hard"])


def by_difficulty() -> dict[str, Task]:
    return dict(TASKS)
