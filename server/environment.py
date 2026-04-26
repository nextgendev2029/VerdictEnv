from __future__ import annotations

import random
import string
import uuid
from copy import deepcopy
from typing import Any, Optional

from openenv.core.env_server import Environment

from verdict_env.models import VerdictAction, VerdictObservation, VerdictState
from verdict_env.tasks import Task, get_task

PHASE_OPENING = "opening"
PHASE_PROSECUTION = "prosecution_case"
PHASE_DEFENSE = "defense_case"
PHASE_CLOSING = "closing"
PHASE_VERDICT = "verdict"
PHASE_ORDER: tuple[str, ...] = (
    PHASE_OPENING,
    PHASE_PROSECUTION,
    PHASE_DEFENSE,
    PHASE_CLOSING,
    PHASE_VERDICT,
)


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _normalize_sent(sent: dict[str, float]) -> dict[str, float]:
    a = _clip01(float(sent.get("analytical", 0.0)))
    e = _clip01(float(sent.get("empathetic", 0.0)))
    s = _clip01(float(sent.get("skeptical", 0.0)))
    tot = a + e + s
    if tot <= 1e-6:
        return {"analytical": 1 / 3, "empathetic": 1 / 3, "skeptical": 1 / 3}
    return {"analytical": a / tot, "empathetic": e / tot, "skeptical": s / tot}


def _case_score(sent: dict[str, float]) -> float:
    a = float(sent.get("analytical", 0.0))
    e = float(sent.get("empathetic", 0.0))
    s = float(sent.get("skeptical", 0.0))
    return 0.42 * a + 0.33 * e - 0.35 * s


# Neutral-jury baseline (equal thirds): used to center verdict decisions.
_NEUTRAL_SCORE: float = (0.42 + 0.33 - 0.35) / 3  # ≈ 0.1333


def _fingerprint(a: VerdictAction) -> tuple[str, str]:
    return (a.action_type, a.evidence_id or "")


class VerdictEnvironment(Environment[VerdictAction, VerdictObservation, VerdictState]):
    """
    Phased trial simulator: partial observability, opponent memory, upgraded jury
    updates, and multi-part reward (verdict, shift, timing, procedure, opponent,
    penalties). Naive random policies lose to step cost, wrong objections, and
    early strong evidence.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = VerdictState()
        self._rng = random.Random()
        self.phase: str = PHASE_OPENING
        self.phase_step: int = 0
        self._phase_entry_step: int = 1

        self._jury: dict[str, float] = _normalize_sent(
            {"analytical": 0.33, "empathetic": 0.33, "skeptical": 0.34}
        )
        self._jury_before_step: dict[str, float] = dict(self._jury)
        self._juror_weights: dict[str, dict[str, float]] = {
            "analytical": {
                "evidence_weight": 0.60,
                "emotion_weight": 0.20,
                "consistency_weight": 0.20,
            },
            "empathetic": {
                "evidence_weight": 0.25,
                "emotion_weight": 0.55,
                "consistency_weight": 0.20,
            },
            "skeptical": {
                "evidence_weight": 0.45,
                "emotion_weight": 0.15,
                "consistency_weight": 0.40,
            },
        }
        self._defense_themes: list[str] = []
        self._evidence: list[dict[str, Any]] = []
        self._used_evidence: set[str] = set()
        self._opponent_trail: list[dict[str, Any]] = []
        self.opponent_history: list[dict[str, Any]] = []
        self._horizon: int = 20
        self._verdict_ground_truth: str = "defense"
        self._defense_presents_in_opening: int = 0
        self._opening_present_cap: int = 2
        self._last_fingerprint: Optional[tuple[str, str]] = None
        self._pending_opp_strong: bool = False
        self._objectionable: bool = False
        self._verdict_fired: bool = False
        self.history: list[dict[str, Any]] = []
        self._defense_start_step: int = 0
        self._defense_end_step: int = 0
        self._task_id: str = "medium"

    @property
    def state(self) -> VerdictState:
        s = self._state
        s.current_phase = self.phase
        return s

    def _boundaries(self) -> dict[str, tuple[int, int]]:
        h = max(self._horizon, 8)
        t_open = 2
        t_pro_end = max(t_open, int(0.32 * h))
        t_def_end = max(t_pro_end + 1, int(0.78 * h))
        t_close_end = max(t_def_end + 1, int(0.92 * h))
        v_start = min(t_close_end + 1, h)
        return {
            PHASE_OPENING: (1, t_open),
            PHASE_PROSECUTION: (t_open + 1, t_pro_end),
            PHASE_DEFENSE: (t_pro_end + 1, t_def_end),
            PHASE_CLOSING: (t_def_end + 1, t_close_end),
            PHASE_VERDICT: (v_start, h),
        }

    def _update_phase(self, step_count: int) -> None:
        b = self._boundaries()
        prev = self.phase
        t = int(step_count)
        if t <= 0:
            self.phase = PHASE_OPENING
            self._phase_entry_step = 1
        else:
            for name in PHASE_ORDER:
                lo, hi = b[name]
                if lo <= t <= hi:
                    if self.phase != name:
                        self._phase_entry_step = t
                    self.phase = name
                    break
            else:
                self.phase = PHASE_VERDICT
                if prev != self.phase:
                    self._phase_entry_step = t
        if t > 0:
            self.phase_step = t - self._phase_entry_step + 1
        else:
            self.phase_step = 0
        d0, d1 = b[PHASE_DEFENSE]
        self._defense_start_step, self._defense_end_step = d0, d1

    def _build_evidence(self, n: int) -> list[dict[str, Any]]:
        tags = ("timeline", "physical", "witness", "expert", "character")
        sides = ["defense", "prosecution", "public"]
        # Balanced assignment: cycle through sides then shuffle
        side_list: list[str] = [sides[i % 3] for i in range(n)]
        self._rng.shuffle(side_list)

        items: list[dict[str, Any]] = []
        for i in range(n):
            eid = f"E{i+1}"
            # Uniform strength distribution — no task-specific boost
            strength = 0.20 + 0.75 * self._rng.random()
            side = side_list[i]
            # Hard: hide all prosecution evidence; medium: hide some
            if self._task_id == "hard":
                hidden = bool(side == "prosecution")
            elif self._task_id == "medium":
                hidden = bool(side == "prosecution" and self._rng.random() < 0.5)
            else:
                hidden = False
            is_public = bool(side == "public")
            items.append(
                {
                    "id": eid,
                    "summary": f"Card {eid} — {tags[i % len(tags)]}",
                    "strength": float(strength),
                    "side": side,
                    "hidden_from_defense": hidden and side == "prosecution",
                    "is_public": is_public,
                    "tag": tags[i % len(tags)],
                }
            )
        return items

    def _visible_to_defense(
        self, *, for_available: bool
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for e in self._evidence:
            if e.get("hidden_from_defense") and e.get("side") == "prosecution" and not e.get("is_public"):
                continue
            if for_available and e["id"] in self._used_evidence:
                continue
            d = {k: v for k, v in e.items() if k in ("id", "summary", "strength", "side", "tag", "is_public")}
            if for_available and e.get("side") not in ("defense", "public",) and not e.get("is_public"):
                if e.get("side") == "prosecution" and not e.get("is_public"):
                    continue
            out.append(d)
        return out

    def _remaining_defense_ids(self) -> set[str]:
        s: set[str] = set()
        for e in self._evidence:
            if e["id"] in self._used_evidence:
                continue
            if e.get("side") in ("defense", "public") or e.get("is_public"):
                s.add(e["id"])
        return s

    def _get_ev(self, eid: str) -> Optional[dict[str, Any]]:
        for e in self._evidence:
            if e["id"] == eid:
                return e
        return None

    def _synthetic_opponent(self) -> dict[str, Any]:
        # Prosecution aggression scales with difficulty
        strong_prob = {"easy": 0.22, "medium": 0.38, "hard": 0.58}.get(self._task_id, 0.38)
        strong = self._rng.random() < strong_prob
        line = "People offer exhibit; ties incident to the accused." if strong else "The People add narrative detail."
        objectionable = bool(strong) or (self._rng.random() < 0.5)
        evt: dict[str, Any] = {
            "side": "prosecution",
            "line": line,
            "strong": bool(strong),
            "objectionable": objectionable,
            "strength": 0.55 + 0.4 * self._rng.random() if strong else 0.22,
        }
        self._objectionable = bool(objectionable)
        self._pending_opp_strong = bool(strong)
        # Hard: strong prosecution moves also raise jury skeptical directly
        if strong and self._task_id == "hard":
            self._jury["skeptical"] = _clip01(
                self._jury.get("skeptical", 0.33) + 0.04 * self._rng.random()
            )
            self._jury = _normalize_sent(self._jury)
        self.opponent_history.append(deepcopy(evt))
        if len(self.opponent_history) > 32:
            self.opponent_history.pop(0)
        return evt

    def _opponent_tick(self) -> None:
        if self.phase == PHASE_PROSECUTION:
            self._opponent_trail.append(self._synthetic_opponent())
        if len(self._opponent_trail) > 6:
            self._opponent_trail.pop(0)

    def _action_allowed_in_phase(self, a: VerdictAction) -> bool:
        t = a.action_type
        if self.phase == PHASE_VERDICT:
            return False
        if self.phase == PHASE_OPENING:
            if t == "present_evidence":
                return self._defense_presents_in_opening < self._opening_present_cap
            if t == "pass":
                return self._defense_presents_in_opening >= self._opening_present_cap
            return False
        if self.phase == PHASE_PROSECUTION:
            return t in ("object", "pass")
        if self.phase == PHASE_DEFENSE:
            return t in ("present_evidence", "object", "pass")
        if self.phase == PHASE_CLOSING:
            return t in ("object", "pass")
        return t in ("object", "pass")

    def _check_structural(self, a: VerdictAction) -> tuple[bool, str]:
        if self.phase == PHASE_VERDICT:
            return False, "verdict: no actions"
        if a.action_type == "present_evidence" and a.evidence_id:
            eid = a.evidence_id
            if eid in self._used_evidence:
                return False, "duplicate exhibit"
            ev = self._get_ev(eid)
            if ev is None:
                return False, "unknown exhibit"
            if eid not in self._remaining_defense_ids():
                return False, "exhibit not available"
            if self.phase == PHASE_OPENING and self._defense_presents_in_opening >= self._opening_present_cap:
                return False, "opening cap"
            if self.phase in (PHASE_PROSECUTION, PHASE_CLOSING):
                return False, "new evidence in wrong phase"
        if a.action_type == "object":
            if not (self._objectionable or self._pending_opp_strong):
                return False, "nothing to object to"
        return True, "ok"

    def _jury_update_from_evidence(
        self, ev: dict[str, Any], *, is_defense: bool, contradiction: bool
    ) -> None:
        """Risk-sensitive, symmetric jury update.

        Strong evidence (>= 0.7) gives a small positive shift.
        Medium evidence (0.4–0.7) is near-neutral.
        Weak evidence (< 0.4) actively hurts the presenter.
        Contradictions and repetitions always hurt.
        """
        s = self._jury
        st = float(ev.get("strength", 0.4))
        noise = 0.02 * (self._rng.random() - 0.5)

        if st >= 0.70:
            delta_good = 0.08 + 0.12 * (st - 0.70)
            delta_skep = -0.06 - 0.08 * (st - 0.70)
        elif st >= 0.40:
            frac = (st - 0.40) / 0.30
            delta_good = 0.015 + 0.03 * frac
            delta_skep = -0.015 - 0.04 * frac
        else:
            delta_good = -0.03 - 0.05 * (0.40 - st)
            delta_skep = 0.04 + 0.06 * (0.40 - st)

        if self._task_id == "easy":
            delta_good *= 1.25
            delta_skep *= 1.25

        if is_defense:
            s["analytical"] = _clip01(s["analytical"] + delta_good + noise)
            s["empathetic"] = _clip01(s["empathetic"] + delta_good * 0.6 + noise)
            s["skeptical"] = _clip01(s["skeptical"] + delta_skep + noise)
        else:
            s["analytical"] = _clip01(s["analytical"] - delta_good * 0.5 + noise)
            s["empathetic"] = _clip01(s["empathetic"] - delta_good * 0.3 + noise)
            s["skeptical"] = _clip01(s["skeptical"] - delta_skep * 0.5 + noise)

        if contradiction:
            s["skeptical"] = _clip01(s["skeptical"] + 0.15 + 0.04 * self._rng.random())
            s["empathetic"] = _clip01(s["empathetic"] - 0.08)
            s["analytical"] = _clip01(s["analytical"] - 0.05)

        self._jury = _normalize_sent(s)

    def _check_contradiction(self, ev: dict[str, Any]) -> bool:
        tag = str(ev.get("tag", ""))
        if tag in self._defense_themes:
            return True
        self._defense_themes.append(tag)
        if len(self._defense_themes) > 12:
            self._defense_themes.pop(0)
        return False

    def _terminal_winner(self) -> str:
        score = _case_score(self._jury)
        delta = score - _NEUTRAL_SCORE
        if self._task_id == "easy":
            thr = 0.07
        elif self._task_id == "medium":
            thr = 0.06
        else:
            thr = 0.08
        if delta > thr:
            return "defense"
        if delta < -thr:
            return "prosecution"
        # Uncertain zone: unbiased coin flip
        return "defense" if self._rng.random() < 0.5 else "prosecution"

    def _verdict_once(self) -> float:
        if self._verdict_fired:
            return 0.0
        self._verdict_fired = True
        w = self._terminal_winner()
        return 8.0 if w == self._verdict_ground_truth else -8.0

    def _jury_shift_sum(self) -> float:
        new = _normalize_sent(self._jury)
        old = _normalize_sent(self._jury_before_step)
        return float(sum(new[k] - old[k] for k in new))

    def _timing_bonus(
        self, a: VerdictAction, ev: Optional[dict[str, Any]], step_count: int
    ) -> float:
        if a.action_type != "present_evidence" or ev is None:
            return 0.0
        if float(ev.get("strength", 0.0)) <= 0.8 + 1e-9:
            return 0.0
        lo, hi = int(self._defense_start_step), int(self._defense_end_step)
        mid = max(1, (lo + hi) // 2)
        if not (lo <= step_count <= hi):
            return -1.0
        if step_count >= int(0.75 * hi):
            return 3.0
        if step_count > mid:
            return 2.0
        return -1.0

    def _procedural(self, a: VerdictAction, had_opportunity: bool) -> float:
        if a.action_type != "object":
            return 0.0
        if not had_opportunity:
            return -2.0
        return 2.0

    def _opp_reaction(
        self,
        a: VerdictAction,
        had_strong: bool,
        action_valid: bool,
        objectionable_snapshot: bool,
    ) -> float:
        if not action_valid or not had_strong:
            return 0.0
        if a.action_type == "object" and objectionable_snapshot:
            return 1.0
        if a.action_type == "pass":
            return -0.5
        return 0.0

    def _reset_internal(self, eid: str, n_ev: int, task: Task, **kwargs: Any) -> None:
        self._state = VerdictState(
            episode_id=eid,
            step_count=0,
            current_phase=PHASE_OPENING,
            case_id="CASE-" + "".join(self._rng.choices(string.ascii_uppercase, k=4)),
            num_evidence=n_ev,
        )
        self.phase = PHASE_OPENING
        self.phase_step = 0
        self._phase_entry_step = 1
        self._jury = _normalize_sent(
            {
                "analytical": 0.33 + 0.04 * (self._rng.random() - 0.5),
                "empathetic": 0.33 + 0.04 * (self._rng.random() - 0.5),
                "skeptical": 0.33 + 0.04 * (self._rng.random() - 0.5),
            }
        )
        # Agent IS the defense — goal is always to win as defense.
        self._verdict_ground_truth = "defense"
        g = kwargs.get("ground_truth")
        if g in ("defense", "prosecution"):
            self._verdict_ground_truth = str(g)
        self._defense_themes = []
        # Task-specific jury starting state
        if self._task_id == "easy":
            # Skeptical in ~[0.35, 0.40]
            base_skeptical = 0.375 + 0.05 * (self._rng.random() - 0.5)
        elif self._task_id == "hard":
            base_skeptical = 0.42 + 0.04 * (self._rng.random() - 0.5)
        elif self._task_id == "medium":
            # Skeptical in ~[0.38, 0.45]
            base_skeptical = 0.415 + 0.07 * (self._rng.random() - 0.5)
        else:
            base_skeptical = 0.33 + 0.03 * (self._rng.random() - 0.5)
        remainder = 1.0 - base_skeptical
        split = self._rng.uniform(0.42, 0.58)
        self._jury = _normalize_sent(
            {
                "analytical": remainder * split,
                "empathetic": remainder * (1.0 - split),
                "skeptical": base_skeptical,
            }
        )
        self._jury_before_step = dict(self._jury)
        self._evidence = self._build_evidence(n_ev)
        self._used_evidence = set()
        self._opponent_trail = []
        self.opponent_history = []
        self._horizon = int(kwargs.get("max_episode_steps", task.max_steps))
        self._defense_presents_in_opening = 0
        self._verdict_fired = False
        self._last_fingerprint = None
        self._pending_opp_strong = False
        self._objectionable = False
        self.history = []
        d0, d1 = self._boundaries()[PHASE_DEFENSE]
        self._defense_start_step, self._defense_end_step = d0, d1

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> VerdictObservation:
        if seed is not None:
            self._rng = random.Random(int(seed))
        eid = episode_id or str(uuid.uuid4())
        t_param = kwargs.get("task", kwargs.get("task_id", "medium"))
        try:
            task = get_task(str(t_param))
        except KeyError:
            task = get_task("medium")
        n_ev = task.num_evidence
        if "num_evidence" in kwargs and kwargs["num_evidence"] is not None:
            n_ev = int(kwargs["num_evidence"])
        self._task_id = task.id
        rest = {k: v for k, v in kwargs.items() if k not in ("task", "task_id")}
        self._reset_internal(eid, n_ev, task, **rest)
        self._update_phase(0)
        return self._build_obs(0.0, False, 0, {})

    def _build_valid_actions(self) -> list[dict[str, str]]:
        if self.phase == PHASE_VERDICT:
            return []
        out: list[dict[str, str]] = []
        if self.phase == PHASE_OPENING:
            if self._defense_presents_in_opening < self._opening_present_cap:
                for e in self._visible_to_defense(for_available=True):
                    eid = str(e.get("id", ""))
                    if not eid or eid in self._used_evidence:
                        continue
                    out.append({"action_type": "present_evidence", "evidence_id": eid})
            else:
                out.append({"action_type": "pass", "evidence_id": ""})
        elif self.phase == PHASE_PROSECUTION:
            out.append({"action_type": "object", "evidence_id": ""})
            out.append({"action_type": "pass", "evidence_id": ""})
        elif self.phase == PHASE_DEFENSE:
            for e in self._visible_to_defense(for_available=True):
                eid = str(e.get("id", ""))
                if not eid or eid in self._used_evidence:
                    continue
                out.append({"action_type": "present_evidence", "evidence_id": eid})
            out.append({"action_type": "object", "evidence_id": ""})
            out.append({"action_type": "pass", "evidence_id": ""})
        elif self.phase == PHASE_CLOSING:
            out.append({"action_type": "object", "evidence_id": ""})
            out.append({"action_type": "pass", "evidence_id": ""})
        seen: set[tuple[str, str]] = set()
        uniq: list[dict[str, str]] = []
        for o in out:
            k = (o["action_type"], o.get("evidence_id", ""))
            if k in seen:
                continue
            seen.add(k)
            uniq.append(o)
        if not uniq and self.phase not in (PHASE_VERDICT, PHASE_OPENING):
            return [{"action_type": "pass", "evidence_id": ""}]
        return uniq

    def _build_obs(
        self,
        reward: float,
        done: bool,
        step_index: int,
        breakdown: dict[str, Any],
    ) -> VerdictObservation:
        vis = self._visible_to_defense(for_available=True)
        return VerdictObservation(
            phase=self.phase,
            step=step_index,
            available_evidence=vis,
            opponent_moves=self._opponent_trail[-5:],
            jury_sentiment=dict(self._jury),
            valid_actions=self._build_valid_actions(),
            case_score=_case_score(self._jury),
            message=f"phase={self.phase} phase_step={self.phase_step} visible={len(vis)}",
            reward=reward,
            done=done,
            metadata={
                "last_reward_breakdown": breakdown,
                "history_len": len(self.history),
            },
        )

    def step(
        self,
        action: VerdictAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> VerdictObservation:
        self._state.step_count += 1
        step_count = int(self._state.step_count)
        self._jury_before_step = dict(self._jury)
        self._update_phase(step_count)
        b = _fingerprint(action)
        is_rep = self._last_fingerprint is not None and b == self._last_fingerprint
        strong0 = self._pending_opp_strong
        obj0 = self._objectionable
        phase_ok = self._action_allowed_in_phase(action)
        ok1, _ = self._check_structural(action)
        valid = bool(phase_ok and ok1 and self.phase != PHASE_VERDICT)
        r_inv = 0.0
        r_rep = 0.0
        r_proc = 0.0
        r_opp = 0.0
        r_time = 0.0
        r_verdict = 0.0
        r_step = -0.01
        if not valid or not ok1 or not phase_ok:
            r_inv = -3.0
        if is_rep:
            r_rep = -1.0
        if valid and action.action_type == "present_evidence" and action.evidence_id:
            eid = action.evidence_id
            evd = self._get_ev(eid)
            if evd and eid in self._remaining_defense_ids():
                self._used_evidence.add(eid)
                if self.phase == PHASE_OPENING:
                    self._defense_presents_in_opening += 1
                c = self._check_contradiction(evd)
                self._jury_update_from_evidence(
                    evd, is_defense=True, contradiction=c
                )
                r_time = self._timing_bonus(action, evd, step_count)
                if float(evd.get("strength", 0.0)) < 0.4:
                    r_time -= 1.5
            else:
                valid = False
                r_inv = -3.0
        if valid and action.action_type == "object" and (obj0 or self._objectionable):
            for axis in self._jury:
                d = 0.05 if axis != "skeptical" else 0.02
                self._jury[axis] = _clip01(self._jury[axis] + d * (1.0 if self._objectionable else 0.2))
            self._jury = _normalize_sent(self._jury)
            r_proc = self._procedural(
                action, had_opportunity=bool(obj0 or self._objectionable)
            )
            self._objectionable = False
            self._pending_opp_strong = False
        if valid and action.action_type == "pass" and self.phase in (
            PHASE_PROSECUTION,
            PHASE_DEFENSE,
        ):
            for axis in self._jury:
                self._jury[axis] = _clip01(
                    self._jury[axis] - 0.012 * (0.4 if axis == "empathetic" else 0.5)
                )
            self._jury = _normalize_sent(self._jury)
        r_opp = self._opp_reaction(
            action,
            had_strong=bool(strong0),
            action_valid=bool(valid and ok1 and phase_ok),
            objectionable_snapshot=bool(obj0),
        )
        r_shift = self._jury_shift_sum() * 2.5
        self._jury = _normalize_sent(self._jury)
        if valid and ok1 and phase_ok:
            self._last_fingerprint = b
        self._opponent_tick()
        done = bool(step_count >= self._horizon or self.phase == PHASE_VERDICT)
        if done:
            r_verdict = self._verdict_once()
        r_total = (
            r_verdict
            + r_shift
            + r_time
            + r_proc
            + r_opp
            + r_inv
            + r_rep
            + r_step
            + 1.0
        )
        breakdown = {
            "verdict": r_verdict,
            "jury_shift": r_shift,
            "timing": r_time,
            "procedural": r_proc,
            "opponent": r_opp,
            "penalty_invalid": r_inv,
            "penalty_repeat": r_rep,
            "step_drain": r_step,
        }
        self.history.append(
            {
                "action": action.model_dump(),
                "phase": self.phase,
                "phase_step": self.phase_step,
                "reward": float(r_total),
                "jury": dict(self._jury),
                "case_score": _case_score(self._jury),
                "breakdown": deepcopy(breakdown),
                "valid": bool(valid and ok1 and phase_ok and self.phase != PHASE_VERDICT),
            }
        )
        if len(self.history) > 200:
            self.history.pop(0)
        return self._build_obs(float(r_total), bool(done), step_count, breakdown)
