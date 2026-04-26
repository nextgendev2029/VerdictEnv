from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# OpenEnv base classes — use the real ones when openenv-core is installed,
# otherwise fall back to faithful inline replicas so the app starts cleanly
# on HF Spaces and fresh installs before openenv-core is available.
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.types import Action, Observation, State  # type: ignore
except ImportError:
    class Action(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(
            extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
        )
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(
            extra="forbid", validate_assignment=True, arbitrary_types_allowed=True
        )
        done: bool = Field(default=False)
        reward: Optional[float] = Field(default=None)
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):  # type: ignore[no-redef]
        model_config = ConfigDict(
            extra="allow", validate_assignment=True, arbitrary_types_allowed=True
        )
        episode_id: Optional[str] = Field(default=None)
        step_count: int = Field(default=0, ge=0)


class VerdictAction(Action):
    """Agent move in a simplified trial: present evidence, object, or pass."""

    action_type: str = Field(
        ...,
        description="One of: present_evidence, object, pass",
    )
    evidence_id: Optional[str] = Field(
        default=None,
        description="Which exhibit to introduce when using present_evidence",
    )
    objection_type: Optional[str] = Field(
        default=None,
        description="e.g. hearsay, relevance (used when action_type=object)",
    )

    @field_validator("action_type")
    @classmethod
    def _valid_action_type(cls, v: str) -> str:
        allowed: set[str] = {"present_evidence", "object", "pass"}
        if v not in allowed:
            raise ValueError(f"action_type must be one of {sorted(allowed)}; got {v!r}")
        return v


class VerdictObservation(Observation):
    """What the agent sees after each step (and at reset)."""

    phase: str = Field(
        default="selection",
        description="Trial phase label (simplified FSM in v0)",
    )
    step: int = Field(
        default=0,
        ge=0,
        description="In-episode turn index (mirrors state.step_count after steps)",
    )
    available_evidence: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Evidence cards the defense may still introduce",
    )
    opponent_moves: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recent prosecutorial or procedural events (synthetic in v0)",
    )
    jury_sentiment: dict[str, float] = Field(
        default_factory=lambda: {
            "analytical": 0.0,
            "empathetic": 0.0,
            "skeptical": 0.0,
        },
        description="Jury priors/leanings, each in [0,1], loosely normalized in env logic",
    )
    valid_actions: list[dict[str, str]] = Field(
        default_factory=list,
        description="Concrete one-hot style actions the policy may emit this turn",
    )
    case_score: float = Field(
        default=0.0,
        description="Internal scalar readout; useful for analysis / shaping",
    )
    message: str = Field(default="", description="Short human-readable status line")

    reward: float = Field(0.0, description="Reward for the last transition")
    done: bool = Field(False, description="Episode finished")


class VerdictState(State):
    """Server-side, compact state surface for /state and WebSocket state pulls."""

    current_phase: str = Field(
        default="selection",
        description="Mirrors the environment phase string",
    )
    case_id: Optional[str] = Field(
        default=None, description="Synthetic case identifier for logging"
    )
    num_evidence: int = Field(
        default=5, ge=1, le=20, description="Number of cards for this docket (task size)"
    )
