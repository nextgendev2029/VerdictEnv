from __future__ import annotations

import os
from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from verdict_env.models import VerdictAction, VerdictObservation, VerdictState


class VerdictEnv(EnvClient[VerdictAction, VerdictObservation, VerdictState]):
    """
    WebSocket client for VerdictEnv (OpenEnv / EnvClient protocol).

    Async usage::

        async with VerdictEnv(base_url=\"http://127.0.0.1:8000\") as env:
            r = await env.reset(task=\"hard\")
            while not r.done:
                r = await env.step(my_action)

    Sync usage::

        v = VerdictEnv(\"http://127.0.0.1:8000\").sync()
        with v:
            r = v.reset(task=\"hard\")
            while not r.done:
                r = v.step(my_action)
    """

    def __init__(self, base_url: str, **kwargs: Any) -> None:
        u = base_url
        if os.environ.get("VERDICT_DEFAULT_URL") and base_url in ("", "default"):
            u = os.environ["VERDICT_DEFAULT_URL"]
        super().__init__(u, **kwargs)

    def _step_payload(self, action: VerdictAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, data: dict[str, Any]) -> StepResult[VerdictObservation]:
        return StepResult(
            observation=VerdictObservation(**data["observation"]),
            reward=data.get("reward"),
            done=bool(data.get("done", False)),
        )

    def _parse_state(self, data: dict[str, Any]) -> VerdictState:
        return VerdictState(**data)
