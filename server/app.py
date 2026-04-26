"""
VerdictEnv — FastAPI REST API + Gradio UI (combined server).

Run combined server (API + UI):
    python -m verdict_env.server.app
    → Gradio UI  at  http://localhost:7860/
    → API docs   at  http://localhost:7860/docs
    → Health     at  http://localhost:7860/api/health
    → Reset      at  http://localhost:7860/api/reset  (POST)
    → Step       at  http://localhost:7860/api/step   (POST)

Run OpenEnv API server (console script):
    verdict-server
"""

from __future__ import annotations

import os
import random
from typing import Any, Optional

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from verdict_env.models import VerdictAction, VerdictObservation
from verdict_env.server.environment import VerdictEnvironment
from verdict_env.inference import LearnedAgent

# ---------------------------------------------------------------------------
# OpenEnv HTTP server — kept for `verdict-server` console-script only.
# Import is lazy so the Gradio UI starts cleanly even if openenv-core is
# unavailable in the deployment environment (e.g. HF Spaces cold-build).
# ---------------------------------------------------------------------------

_openenv_app: Any = None


def _get_openenv_app() -> Any:
    global _openenv_app
    if _openenv_app is not None:
        return _openenv_app
    try:
        from openenv.core.env_server.http_server import create_app  # type: ignore
        _openenv_app = create_app(
            VerdictEnvironment,
            VerdictAction,
            VerdictObservation,
            env_name="verdict_env",
            max_concurrent_envs=int(os.getenv("MAX_CONCURRENT_ENVS", "8")),
        )
    except ImportError:
        raise RuntimeError(
            "openenv-core is not installed. "
            "Run `pip install openenv-core[core]` to use the verdict-server console script."
        )
    return _openenv_app


def serve_api() -> None:
    """Launch the OpenEnv FastAPI server (uvicorn)."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(_get_openenv_app(), host=host, port=port, workers=int(os.getenv("WORKERS", "1")))


# ---------------------------------------------------------------------------
# Custom FastAPI app — REST API with explicit routes
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VerdictEnv API",
    description="Courtroom RL environment — REST endpoints under /api/*. Gradio UI at /.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Pydantic schemas ────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "medium"


class StepRequest(BaseModel):
    action_type: str
    evidence_id: Optional[str] = None
    strength: str = "high"       # used to auto-pick evidence when evidence_id is None


class ObsResponse(BaseModel):
    phase: str
    step: int
    jury_sentiment: dict[str, float]
    case_score: float
    valid_actions: list[dict[str, str]]
    available_evidence: list[dict[str, Any]]
    message: str
    reward: float
    done: bool
    task: str


class StepResponse(BaseModel):
    observation: ObsResponse
    reward: float
    done: bool


# ── Global API env state ────────────────────────────────────────────────────

class _APISession:
    """Thread-safe-enough single-session env wrapper for the REST API."""

    def __init__(self) -> None:
        self._env: Optional[VerdictEnvironment] = None
        self._task: str = "medium"
        self._obs: Optional[VerdictObservation] = None

    def reset(self, task: str = "medium") -> VerdictObservation:
        self._env = VerdictEnvironment()
        self._task = task
        self._obs = self._env.reset(task=task)
        return self._obs

    def step(self, action: VerdictAction) -> VerdictObservation:
        if self._env is None or self._obs is None:
            raise RuntimeError("Call POST /reset before POST /step.")
        self._obs = self._env.step(action)
        return self._obs

    @property
    def task(self) -> str:
        return self._task

    @property
    def obs(self) -> Optional[VerdictObservation]:
        return self._obs


_session = _APISession()


def _obs_to_response(obs: VerdictObservation, task: str) -> ObsResponse:
    return ObsResponse(
        phase=obs.phase,
        step=obs.step,
        jury_sentiment=dict(obs.jury_sentiment or {}),
        case_score=float(obs.case_score or 0.0),
        valid_actions=list(obs.valid_actions or []),
        available_evidence=list(obs.available_evidence or []),
        message=obs.message or "",
        reward=float(obs.reward or 0.0),
        done=bool(obs.done),
        task=task,
    )


# ── Routes (/api/*) ─────────────────────────────────────────────────────────

@app.get("/api", summary="API info")
def api_root():
    return {"message": "VerdictEnv API running", "docs": "/docs", "ui": "/"}


@app.get("/api/health", summary="Health check")
def health():
    return {"status": "ok"}


@app.post("/api/reset", response_model=ObsResponse, summary="Reset environment")
def reset(body: ResetRequest = ResetRequest()):
    valid_tasks = {"easy", "medium", "hard"}
    if body.task not in valid_tasks:
        raise HTTPException(
            status_code=422,
            detail=f"task must be one of {sorted(valid_tasks)}, got {body.task!r}",
        )
    obs = _session.reset(body.task)
    return _obs_to_response(obs, body.task)


@app.post("/api/step", response_model=StepResponse, summary="Step environment")
def step(body: StepRequest):
    if _session.obs is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call POST /api/reset first.")
    if _session.obs.done:
        raise HTTPException(status_code=400, detail="Episode finished. Call POST /api/reset to start a new one.")

    valid_action_types = {"present_evidence", "object", "pass"}
    if body.action_type not in valid_action_types:
        raise HTTPException(
            status_code=422,
            detail=f"action_type must be one of {sorted(valid_action_types)}, got {body.action_type!r}",
        )

    obs_current = _session.obs
    valid = obs_current.valid_actions or []

    # Resolve evidence_id: use provided id, or auto-pick by strength
    eid: Optional[str] = body.evidence_id
    if body.action_type == "present_evidence" and eid is None:
        eid = _pick_evidence(obs_current, body.strength)

    # Find matching valid-action row
    row = None
    if body.action_type == "present_evidence" and eid:
        row = next(
            (r for r in valid if r.get("action_type") == "present_evidence" and r.get("evidence_id") == eid),
            None,
        )
    if row is None:
        row = next((r for r in valid if r.get("action_type") == body.action_type), None)
    if row is None:
        raise HTTPException(
            status_code=422,
            detail=f"Action '{body.action_type}' is not legal in the current phase '{obs_current.phase}'. "
                   f"Valid action types: {[r.get('action_type') for r in valid]}",
        )

    at = row.get("action_type", "pass")
    resolved_eid = row.get("evidence_id") or None
    obj_type = random.choice(("hearsay", "relevance", "speculation")) if at == "object" else None
    action = VerdictAction(action_type=at, evidence_id=resolved_eid, objection_type=obj_type)

    obs_next = _session.step(action)
    return StepResponse(
        observation=_obs_to_response(obs_next, _session.task),
        reward=float(obs_next.reward or 0.0),
        done=bool(obs_next.done),
    )


# ---------------------------------------------------------------------------
# Gradio UI — AI Court Simulator
# ---------------------------------------------------------------------------

_NEUTRAL = (0.42 + 0.33 - 0.35) / 3          # ≈ 0.1333
_THRESH = {"easy": 0.07, "medium": 0.06, "hard": 0.08}
_PHASE_LABELS = {
    "opening": "📜 Opening Statement",
    "prosecution_case": "⚔️  Prosecution Case",
    "defense_case": "🛡  Defense Case",
    "closing": "🎤 Closing Arguments",
    "verdict": "🏛  VERDICT",
}
_PHASE_COLORS = {
    "opening": "#4f46e5",
    "prosecution_case": "#dc2626",
    "defense_case": "#0369a1",
    "closing": "#7c3aed",
    "verdict": "#1e3a5f",
}
_STRENGTH_TARGET = {"low": 0.20, "medium": 0.55, "high": 0.90}

# ── State helpers ──────────────────────────────────────────────────────────


def _new_state(task: str = "medium") -> dict[str, Any]:
    env = VerdictEnvironment()
    obs = env.reset(task=task)
    return {"env": env, "obs": obs, "logs": [], "task": task, "jury_delta": {}, "agent": LearnedAgent(epsilon=0.20)}


def _case_score(jury: dict[str, float]) -> float:
    return 0.42 * jury.get("analytical", 0) + 0.33 * jury.get("empathetic", 0) - 0.35 * jury.get("skeptical", 0)


def _verdict_label(score: float, task: str) -> str:
    thr = _THRESH.get(task, 0.06)
    if score > _NEUTRAL + thr:
        return "✅ Defense Winning"
    if score < _NEUTRAL - thr:
        return "❌ Prosecution Winning"
    return "⚖️  Uncertain"


def _pick_evidence(obs: Any, bucket: str) -> str | None:
    target = _STRENGTH_TARGET[bucket]
    valid_ids = {
        r.get("evidence_id")
        for r in (obs.valid_actions or [])
        if r.get("action_type") == "present_evidence" and r.get("evidence_id")
    }
    candidates = [e for e in (obs.available_evidence or []) if e.get("id") in valid_ids]
    if not candidates:
        return None
    return min(candidates, key=lambda e: abs(float(e.get("strength", 0.5)) - target)).get("id")


# ── HTML renderers ─────────────────────────────────────────────────────────


def _jury_html(jury: dict[str, float], deltas: dict[str, float] | None = None) -> str:
    """Render jury sentiment bars with optional delta arrows (green ↑ / red ↓)."""
    a = jury.get("analytical", 0)
    e = jury.get("empathetic", 0)
    s = jury.get("skeptical", 0)
    d = deltas or {}

    def _delta_badge(key: str) -> str:
        val = d.get(key, 0.0)
        if abs(val) < 0.001:
            return ""
        if val > 0:
            color = "#16a34a" if key != "skeptical" else "#dc2626"
            arrow = "▲"
        else:
            color = "#dc2626" if key != "skeptical" else "#16a34a"
            arrow = "▼"
        return (
            f'<span style="margin-left:6px;font-size:0.78em;font-weight:700;'
            f'color:{color}">{arrow} {abs(val):.3f}</span>'
        )

    def bar(val: float, color: str, label: str, key: str) -> str:
        pct = int(val * 100)
        # Highlight the bar background if this metric moved significantly
        delta_val = d.get(key, 0.0)
        glow = ""
        if abs(delta_val) >= 0.005:
            glow_color = "#bbf7d0" if (
                (key != "skeptical" and delta_val > 0) or (key == "skeptical" and delta_val < 0)
            ) else "#fecaca"
            glow = f"box-shadow:0 0 0 2px {glow_color};"
        return (
            f'<div style="margin:8px 0;padding:4px 6px;border-radius:6px;background:#f8fafc;{glow}">'
            f'<div style="display:flex;align-items:center;margin-bottom:4px">'
            f'<span style="font-weight:700;color:{color};width:108px;font-size:0.9em">{label}</span>'
            f'<span style="font-size:0.9em;color:#374151;margin-left:auto">{val:.3f}</span>'
            f'{_delta_badge(key)}'
            f'</div>'
            f'<div style="background:#e5e7eb;border-radius:4px;height:10px">'
            f'<div style="background:{color};width:{pct}%;height:10px;border-radius:4px;'
            f'transition:width 0.4s ease"></div>'
            f'</div>'
            f'</div>'
        )

    return (
        '<div style="padding:2px 0">'
        + bar(a, "#16a34a", "🧠 Analytical", "analytical")
        + bar(e, "#2563eb", "❤️  Empathetic", "empathetic")
        + bar(s, "#dc2626", "🔍 Skeptical",  "skeptical")
        + "</div>"
    )


def _score_html(score: float, task: str) -> str:
    thr = _THRESH.get(task, 0.06)
    if score > _NEUTRAL + thr:
        color, icon, label = "#16a34a", "▲", "Defense Ahead"
    elif score < _NEUTRAL - thr:
        color, icon, label = "#dc2626", "▼", "Prosecution Ahead"
    else:
        color, icon, label = "#d97706", "●", "Contested"
    bar_pct = int(min(max((score + 0.5) * 100, 0), 100))
    return (
        f'<div style="padding:10px 12px;border-radius:8px;background:#f9fafb;'
        f'border:2px solid {color};margin-top:6px">'
        f'<div style="display:flex;align-items:center;justify-content:space-between">'
        f'<span style="font-size:0.8em;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:.05em">Case Score</span>'
        f'<span style="font-size:1.5em;font-weight:800;color:{color}">{icon} {score:.4f}</span>'
        f'</div>'
        f'<div style="background:#e5e7eb;border-radius:4px;height:8px;margin:6px 0">'
        f'<div style="background:{color};width:{bar_pct}%;height:8px;border-radius:4px"></div>'
        f'</div>'
        f'<div style="font-size:0.78em;color:{color};font-weight:600;text-align:right">{label} · neutral≈{_NEUTRAL:.4f}</div>'
        f'</div>'
    )


def _phase_html(phase: str) -> str:
    label = _PHASE_LABELS.get(phase, phase)
    color = _PHASE_COLORS.get(phase, "#1e3a5f")
    return (
        f'<div style="text-align:center;padding:10px 12px;border-radius:8px;'
        f'background:{color};color:white;font-size:1.05em;font-weight:700;'
        f'letter-spacing:0.03em;box-shadow:0 2px 6px rgba(0,0,0,0.15)">'
        f'{label}'
        f'</div>'
    )


def _verdict_html(score: float, task: str, done: bool) -> str:
    thr = _THRESH.get(task, 0.06)
    if not done:
        label = _verdict_label(score, task)
        color = "#6b7280"
        bg = "#f9fafb"
        pulse = ""
    elif score > _NEUTRAL + thr:
        label, color, bg = "✅ DEFENSE WINS", "#15803d", "#f0fdf4"
        pulse = "animation:pulse 1s ease-in-out 3;"
    elif score < _NEUTRAL - thr:
        label, color, bg = "❌ PROSECUTION WINS", "#b91c1c", "#fef2f2"
        pulse = "animation:pulse 1s ease-in-out 3;"
    else:
        label, color, bg = "⚖️  UNDECIDED", "#b45309", "#fffbeb"
        pulse = ""
    return (
        f'<div style="text-align:center;padding:12px 16px;border-radius:10px;'
        f'font-size:1.15em;font-weight:800;color:{color};'
        f'border:2px solid {color};background:{bg};{pulse}">{label}</div>'
    )


def _log_line(step: int, source: str, action_type: str, ev_note: str, reward: float,
              da: float, de: float, ds: float) -> str:
    """Format a single log entry."""
    src_tag = "USER" if source == "USER" else " RL "
    ev_part = f" [{ev_note}]" if ev_note else ""
    r_color = "+" if reward >= 0 else ""
    delta_parts = []
    if abs(da) >= 0.001:
        delta_parts.append(f"ana {'+' if da >= 0 else ''}{da:.3f}")
    if abs(de) >= 0.001:
        delta_parts.append(f"emp {'+' if de >= 0 else ''}{de:.3f}")
    if abs(ds) >= 0.001:
        delta_parts.append(f"skp {'+' if ds >= 0 else ''}{ds:.3f}")
    delta_str = "  Δ " + ", ".join(delta_parts) if delta_parts else ""
    return f"[STEP {step:>2}][{src_tag}] Action: {action_type}{ev_part} → Reward: {r_color}{reward:.3f}{delta_str}"


# ── Action execution ───────────────────────────────────────────────────────


def _step(
    state: dict[str, Any],
    action_type: str,
    evidence_id: str | None,
    source: str,
    agent: LearnedAgent | None = None,
):
    obs = state["obs"]
    if obs.done:
        state["logs"].append("⚠️  Episode finished — reset to continue.")
        return state

    valid = obs.valid_actions or []
    row = None
    if action_type == "present_evidence" and evidence_id:
        row = next(
            (r for r in valid if r.get("action_type") == "present_evidence" and r.get("evidence_id") == evidence_id),
            None,
        )
    if row is None:
        row = next((r for r in valid if r.get("action_type") == action_type), None)
    if row is None:
        types_avail = [r.get("action_type") for r in valid]
        state["logs"].append(f"⚠️  {action_type!r} not legal here. Available: {types_avail}")
        return state

    at = row.get("action_type", "pass")
    eid = row.get("evidence_id") or None
    obj_type = random.choice(("hearsay", "relevance", "speculation")) if at == "object" else None
    action = VerdictAction(action_type=at, evidence_id=eid, objection_type=obj_type)

    strength_map: dict[str, float] = {
        str(e.get("id", "")): float(e.get("strength", 0.0))
        for e in (obs.available_evidence or [])
    }

    j_before = dict(obs.jury_sentiment or {})
    obs_next = state["env"].step(action)
    state["obs"] = obs_next

    if agent is not None:
        agent.update(action, obs, obs_next, float(obs_next.reward or 0.0), strength_map)
        agent.decay_epsilon()

    j_after = obs_next.jury_sentiment or {}
    da = j_after.get("analytical", 0) - j_before.get("analytical", 0)
    de = j_after.get("empathetic", 0) - j_before.get("empathetic", 0)
    ds = j_after.get("skeptical",  0) - j_before.get("skeptical",  0)

    state["jury_delta"] = {"analytical": da, "empathetic": de, "skeptical": ds}

    state["logs"].append(
        _log_line(obs_next.step, source, at, eid or "", obs_next.reward, da, de, ds)
    )

    if obs_next.done:
        score = _case_score(obs_next.jury_sentiment or {})
        thr = _THRESH.get(state["task"], 0.06)
        if score > _NEUTRAL + thr:
            verdict_str = "DEFENSE WINS"
        elif score < _NEUTRAL - thr:
            verdict_str = "PROSECUTION WINS"
        else:
            verdict_str = "UNDECIDED"
        state["logs"].append(f"{'─'*52}")
        state["logs"].append(f"🏛  VERDICT: {verdict_str}  (case score = {score:.4f})")
        state["logs"].append(f"{'─'*52}")

    return state


# ── UI output builder ──────────────────────────────────────────────────────


def _build_outputs(state: dict[str, Any]):
    obs = state["obs"]
    jury = obs.jury_sentiment or {}
    deltas = state.get("jury_delta", {})
    score = _case_score(jury)
    done = bool(obs.done)
    task = state["task"]
    logs_text = "\n".join(state["logs"][-10:])
    return (
        state,
        _phase_html(obs.phase),
        _jury_html(jury, deltas),
        _score_html(score, task),
        _verdict_html(score, task, done),
        logs_text,
    )


# ── Gradio callbacks ──────────────────────────────────────────────────────


def cb_take_action(action: str, strength: str, state: dict[str, Any] | None):
    if state is None:
        state = _new_state()
    obs = state["obs"]
    eid = _pick_evidence(obs, strength) if action == "present_evidence" else None
    state = _step(state, action, eid, "USER")
    return _build_outputs(state)


def cb_ai_decide(state: dict[str, Any] | None):
    if state is None:
        state = _new_state()
    obs = state["obs"]
    if obs.done:
        state["logs"].append("⚠️  Episode finished — reset to continue.")
        return _build_outputs(state)

    agent = state["agent"]
    action = agent.pick(obs)
    if action is None:
        state["logs"].append("⚠️  RL Agent: no valid actions.")
        return _build_outputs(state)

    state = _step(state, action.action_type, action.evidence_id, "RL", agent=agent)
    return _build_outputs(state)


def cb_reset(task: str, _state: Any):
    state = _new_state(task)
    state["logs"].append(f"{'═'*52}")
    state["logs"].append(f"▶  New episode started  (difficulty: {task.upper()})")
    state["logs"].append(f"{'═'*52}")
    return _build_outputs(state)


# ── CSS ────────────────────────────────────────────────────────────────────

_CSS = """
/* Log box */
.log-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 0.80em;
    line-height: 1.6;
    background: #0f172a;
    color: #e2e8f0;
    border-radius: 8px;
}

/* Section cards */
.section-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}

/* Section headings */
.section-heading {
    font-size: 0.72em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7280;
    margin-bottom: 10px;
    border-bottom: 1px solid #f3f4f6;
    padding-bottom: 6px;
}

/* Button spacing */
.btn-row { gap: 6px; }

/* Pulse animation for final verdict */
@keyframes pulse {
    0%   { transform: scale(1); }
    50%  { transform: scale(1.02); box-shadow: 0 0 12px rgba(0,0,0,0.15); }
    100% { transform: scale(1); }
}

/* Gradio overrides */
.gradio-container { max-width: 1200px !important; }
footer { display: none !important; }
"""

# ── Layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="AI Court Simulator") as demo:

    session = gr.State(value=None)

    # ── Header ────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="background:linear-gradient(135deg,#1e3a5f 0%,#2d5986 100%);
                border-radius:12px;padding:20px 28px;margin-bottom:4px;
                display:flex;align-items:center;justify-content:space-between">
      <div>
        <div style="font-size:1.6em;font-weight:800;color:white;line-height:1.2">
          ⚖️ AI Court Simulator
        </div>
        <div style="font-size:0.88em;color:#93c5fd;margin-top:4px">
          Pure Reinforcement Learning · Tabular Q-learning · Influence the jury, win the case
        </div>
      </div>
      <div style="font-size:2.4em;opacity:0.25">🏛</div>
    </div>
    """)

    # ── Top control bar ────────────────────────────────────────────────────
    with gr.Group():
        with gr.Row(equal_height=True):
            task_dd = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="medium",
                label="🎯 Difficulty",
                scale=1,
            )
            reset_btn = gr.Button(
                "🔄 Reset",
                variant="secondary",
                scale=1,
            )
            verdict_html_top = gr.HTML(
                '<div style="text-align:center;padding:10px 16px;border-radius:10px;'
                'font-size:1.1em;font-weight:800;color:#6b7280;'
                'border:2px solid #e5e7eb;background:#f9fafb">⚖️  Uncertain</div>',
                label="Live Verdict",
                scale=2,
            )

    gr.HTML('<div style="height:8px"></div>')

    # ── Main 3-column body ─────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # LEFT — Actions panel
        with gr.Column(scale=1, min_width=210):
            gr.HTML('<div class="section-heading">🎮 Actions</div>')
            with gr.Group():
                action_dd = gr.Dropdown(
                    choices=["present_evidence", "object", "pass"],
                    value="present_evidence",
                    label="Action Type",
                )
                strength_dd = gr.Dropdown(
                    choices=["low", "medium", "high"],
                    value="high",
                    label="Evidence Strength",
                )
                gr.HTML('<div style="height:4px"></div>')
                take_btn = gr.Button("⚡ Step (Manual)", variant="primary", size="lg")
                ai_btn   = gr.Button("🤖 Auto Play (RL Agent)", variant="secondary", size="lg")
                gr.HTML('<div style="height:2px"></div>')
                gr.HTML(
                    '<div style="font-size:0.76em;color:#9ca3af;line-height:1.5;padding:4px 2px">'
                    '🤖 <em>RL Agent uses ε-greedy Q-learning. Learns within the session.</em>'
                    '</div>'
                )

        # CENTER — Trial state panel
        with gr.Column(scale=2, min_width=300):
            gr.HTML('<div class="section-heading">🏛 Trial State</div>')
            with gr.Group():
                gr.HTML('<div style="font-size:0.75em;font-weight:600;color:#9ca3af;'
                        'text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">Current Phase</div>')
                phase_html = gr.HTML()
                gr.HTML('<div style="height:10px"></div>')
                gr.HTML('<div style="font-size:0.75em;font-weight:600;color:#9ca3af;'
                        'text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px">Jury Sentiment</div>')
                jury_html  = gr.HTML()
                gr.HTML('<div style="height:6px"></div>')
                score_html = gr.HTML()

        # RIGHT — Step log panel
        with gr.Column(scale=2, min_width=300):
            gr.HTML('<div class="section-heading">📋 Step Log <span style="font-weight:400;color:#d1d5db">(last 10 actions)</span></div>')
            log_out = gr.Textbox(
                lines=20,
                max_lines=20,
                interactive=False,
                show_label=False,
                placeholder="Steps will appear here after the first action...",
                elem_classes=["log-box"],
            )

    # ── Wire events ────────────────────────────────────────────────────────
    _outs = [session, phase_html, jury_html, score_html, verdict_html_top, log_out]

    take_btn.click(cb_take_action, inputs=[action_dd, strength_dd, session], outputs=_outs)
    ai_btn.click(cb_ai_decide,    inputs=[session],                          outputs=_outs)
    reset_btn.click(cb_reset,     inputs=[task_dd, session],                 outputs=_outs)
    demo.load(cb_reset,           inputs=[task_dd, session],                 outputs=_outs)


# ---------------------------------------------------------------------------
# Mount Gradio at "/" — FastAPI /api/* and /docs routes take priority
# ---------------------------------------------------------------------------

app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"\nVerdictEnv server starting on http://{host}:{port}")
    print(f"  Gradio UI  → http://{host}:{port}/")
    print(f"  API docs   → http://{host}:{port}/docs")
    print(f"  Health     → http://{host}:{port}/api/health")
    print(f"  Reset      → http://{host}:{port}/api/reset  (POST)")
    print(f"  Step       → http://{host}:{port}/api/step   (POST)\n")
    uvicorn.run(app, host=host, port=port)
