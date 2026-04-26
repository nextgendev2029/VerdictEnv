"""
VerdictEnv inference + lightweight training demo.

Usage
-----
    python -m verdict_env.inference                        # full train + compare
    python -m verdict_env.inference --mode random          # single random episode
    python -m verdict_env.inference --mode greedy          # single greedy episode
    python -m verdict_env.inference --mode train           # training only
    python -m verdict_env.inference --task hard --episodes 300
    python -m verdict_env.inference --remote http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Repo-relative assets directory — always inside the project root regardless
# of where the process is launched from.
#
#   inference.py  →  verdict_env/inference.py
#   __file__      →  .../verdict_env/inference.py
#   .parent       →  .../verdict_env/          (package dir)
#   .parent.parent→  .../                      (project root)
# ---------------------------------------------------------------------------

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_ASSETS_DIR.mkdir(exist_ok=True)

from verdict_env.models import VerdictAction
from verdict_env.server.environment import VerdictEnvironment

# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def _make_action(row: dict[str, str]) -> VerdictAction:
    at = row.get("action_type", "pass")
    eid = row.get("evidence_id") or None
    obj: Optional[str] = None
    if at == "object":
        obj = random.choice(("hearsay", "relevance", "speculation"))
    return VerdictAction(
        action_type=at,
        evidence_id=eid if eid else None,
        objection_type=obj,
    )


# ---------------------------------------------------------------------------
# Agent 1: Random
# ---------------------------------------------------------------------------

def random_action(obs) -> Optional[VerdictAction]:
    """Select uniformly at random from valid_actions."""
    if not obs.valid_actions:
        return None
    return _make_action(random.choice(obs.valid_actions))


# ---------------------------------------------------------------------------
# Agent 2: Greedy
# ---------------------------------------------------------------------------

def greedy_action(obs) -> Optional[VerdictAction]:
    """
    1. Present the highest-strength unused evidence if any is available.
    2. Object if it is in valid_actions.
    3. Pass if listed in valid_actions; otherwise None (no legal move).
    """
    if not obs.valid_actions:
        return None
    valid = obs.valid_actions or []
    # Map available evidence by id -> strength
    strength_map: dict[str, float] = {
        str(e.get("id", "")): float(e.get("strength", 0.0))
        for e in (obs.available_evidence or [])
    }
    best_ev: Optional[dict[str, str]] = None
    best_st = -1.0
    for row in valid:
        if row.get("action_type") == "present_evidence":
            eid = row.get("evidence_id", "")
            st = strength_map.get(eid, 0.0)
            if st > best_st:
                best_st, best_ev = st, row
    if best_ev is not None:
        return _make_action(best_ev)
    for row in valid:
        if row.get("action_type") == "object":
            return _make_action(row)
    for row in valid:
        if row.get("action_type") == "pass":
            return _make_action(row)
    return None


# ---------------------------------------------------------------------------
# Agent 3: Learned (pure tabular Q, incremental mean per state–action)
# ---------------------------------------------------------------------------

class LearnedAgent:
    """
    Discretized state (phase, rounded skeptical) × action; Q updated with
    sample-mean of environment rewards (no manual shaping).
    """

    def __init__(self, epsilon: float = 0.15) -> None:
        self.epsilon = epsilon
        self.Q: dict[str, float] = {}
        self.N: dict[str, int] = {}

    def key(
        self,
        row: dict[str, str],
        strength_map: dict[str, float],
        obs: Any,
    ) -> str:
        phase = getattr(obs, "phase", "unknown")
        j = getattr(obs, "jury_sentiment", None) or {}
        skeptical = round(float((j or {}).get("skeptical", 0.33)), 1)

        at = row.get("action_type", "pass")

        if at == "present_evidence":
            eid = row.get("evidence_id", "")
            st = strength_map.get(eid, 0.0)
            bucket = "hi" if st >= 0.75 else ("mid" if st >= 0.45 else "lo")
            return f"{phase}|{skeptical}|present{bucket}"

        return f"{phase}|{skeptical}|{at}"

    def pick(self, obs) -> Optional[VerdictAction]:
        valid = obs.valid_actions or []
        if not valid:
            return None
        strength_map: dict[str, float] = {
            str(e.get("id", "")): float(e.get("strength", 0.0))
            for e in (obs.available_evidence or [])
        }
        if random.random() < self.epsilon:
            return _make_action(random.choice(valid))

        q_best = max(valid, key=lambda r: self.Q.get(self.key(r, strength_map, obs), 0.0))
        return _make_action(q_best)

    def update(
        self,
        action: VerdictAction,
        obs: Any,
        obs_next: Any,
        reward: float,
        strength_map: dict[str, float],
    ) -> None:
        if getattr(obs_next, "phase", None) == "verdict":
            return
        row = {
            "action_type": action.action_type,
            "evidence_id": (action.evidence_id or "")
            if action.action_type == "present_evidence"
            else "",
        }
        k = self.key(row, strength_map, obs)
        self.N[k] = self.N.get(k, 0) + 1
        alpha = 1.0 / self.N[k]
        old_q = self.Q.get(k, 0.0)
        r_env = float(reward)
        self.Q[k] = old_q + alpha * (r_env - old_q)

    def decay_epsilon(self, rate: float = 0.99) -> None:
        self.epsilon = max(0.02, self.epsilon * rate)

    def summary(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in sorted(self.Q.items())}


# ---------------------------------------------------------------------------
# Episode runners (sync + remote async)
# ---------------------------------------------------------------------------

def _run_episode(
    env: VerdictEnvironment,
    *,
    task: str,
    seed: Optional[int],
    action_fn,
    agent: Optional[LearnedAgent] = None,
) -> dict[str, Any]:
    obs = env.reset(seed=seed, task=task)
    total = 0.0
    steps = 0
    while not obs.done:
        if not obs.valid_actions or obs.phase == "verdict":
            break

        strength_map: dict[str, float] = {
            str(e.get("id", "")): float(e.get("strength", 0.0))
            for e in (obs.available_evidence or [])
        }

        act = action_fn(obs)
        if act is None:
            break

        obs_next = env.step(act)
        r = float(obs_next.reward or 0.0)
        total += r

        if agent is not None:
            agent.update(act, obs, obs_next, r, strength_map)

        obs = obs_next
        steps += 1
    case_score = float(obs.case_score or 0.0)

    # Mirror _terminal_winner logic from VerdictEnvironment
    _NEUTRAL = (0.42 + 0.33 - 0.35) / 3  # ≈ 0.1333, matches VerdictEnvironment
    delta = case_score - _NEUTRAL
    thr = {"easy": 0.07, "medium": 0.06, "hard": 0.08}.get(task, 0.08)
    if delta > thr:
        result_verdict = "defense"
    elif delta < -thr:
        result_verdict = "prosecution"
    else:
        result_verdict = "defense" if random.random() < 0.5 else "prosecution"

    won = result_verdict == "defense"
    return {
        "total_reward": total,
        "steps": steps,
        "case_score": case_score,
        "won": won,
        "verdict": result_verdict,
    }


async def _run_episode_remote(
    base_url: str, *, task: str, seed: Optional[int]
) -> dict[str, Any]:
    from verdict_env.client import VerdictEnv

    total = 0.0
    steps = 0
    async with VerdictEnv(base_url=base_url) as client:
        r = await client.reset(task=task, seed=seed)
        obs = r.observation
        while not r.done:
            if not obs.valid_actions or obs.phase == "verdict":
                break
            act = random_action(obs)
            if act is None:
                break
            r = await client.step(act)
            obs = r.observation
            total += float(r.reward or 0.0)
            steps += 1
    case_score = float(getattr(obs, "case_score", 0.0) or 0.0)
    _NEUTRAL = (0.42 + 0.33 - 0.35) / 3
    delta = case_score - _NEUTRAL
    thr = {"easy": 0.07, "medium": 0.06, "hard": 0.08}.get(task, 0.08)
    if delta > thr:
        result_verdict = "defense"
    elif delta < -thr:
        result_verdict = "prosecution"
    else:
        result_verdict = "defense" if random.random() < 0.5 else "prosecution"
    return {
        "total_reward": total,
        "steps": steps,
        "case_score": case_score,
        "won": result_verdict == "defense",
        "verdict": result_verdict,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class Metrics:
    def __init__(self) -> None:
        self.episode_rewards: list[float] = []
        self.win_flags: list[int] = []
        self.case_scores: list[float] = []
        self.verdicts: list[str] = []

    def record(self, result: dict[str, Any]) -> None:
        self.episode_rewards.append(result["total_reward"])
        self.win_flags.append(1 if result["won"] else 0)
        self.case_scores.append(result["case_score"])
        self.verdicts.append(result.get("verdict", "unknown"))

    def window_avg(self, data: list[float], w: int = 20) -> list[float]:
        out: list[float] = []
        for i in range(len(data)):
            start = max(0, i - w + 1)
            out.append(sum(data[start : i + 1]) / max(1, i - start + 1))
        return out

    def win_rate(self, w: int = 20) -> list[float]:
        return self.window_avg([float(x) for x in self.win_flags], w)

    def avg_reward(self, w: int = 20) -> list[float]:
        return self.window_avg(self.episode_rewards, w)

    def avg_case_score(self, w: int = 20) -> list[float]:
        return self.window_avg(self.case_scores, w)

    def verdict_dist(self) -> dict[str, int]:
        d: dict[str, int] = {"defense": 0, "prosecution": 0, "unknown": 0}
        for v in self.verdicts:
            d[v] = d.get(v, 0) + 1
        return d

    def summary(self) -> dict[str, float]:
        n = len(self.episode_rewards)
        if n == 0:
            return {}
        return {
            "episodes": float(n),
            "mean_reward": sum(self.episode_rewards) / n,
            "win_rate": sum(self.win_flags) / n,
            "mean_case_score": sum(self.case_scores) / n,
        }


def run_training(
    *,
    task: str,
    episodes: int,
    seed_base: Optional[int],
    epsilon_start: float = 0.30,
    epsilon_decay: float = 0.99,
) -> tuple[Metrics, LearnedAgent]:
    env = VerdictEnvironment()
    agent = LearnedAgent(epsilon=epsilon_start)
    metrics = Metrics()
    for ep in range(episodes):
        seed = None if seed_base is None else seed_base + ep
        result = _run_episode(
            env, task=task, seed=seed, action_fn=agent.pick, agent=agent
        )
        metrics.record(result)
        agent.decay_epsilon(epsilon_decay)
    return metrics, agent


def run_baseline(
    *,
    task: str,
    episodes: int,
    seed_base: Optional[int],
    mode: str = "random",
) -> Metrics:
    env = VerdictEnvironment()
    fn = random_action if mode == "random" else greedy_action
    metrics = Metrics()
    for ep in range(episodes):
        seed = None if seed_base is None else seed_base + ep
        result = _run_episode(env, task=task, seed=seed, action_fn=fn)
        metrics.record(result)
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(
    metrics_random: Metrics,
    metrics_greedy: Metrics,
    metrics_trained: Metrics,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping plots.")
        return

    episodes = list(range(1, len(metrics_trained.episode_rewards) + 1))

    # ---- reward curve ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, metrics_trained.avg_reward(), label="Trained (rolling avg)", linewidth=2)
    ax.axhline(
        sum(metrics_random.episode_rewards) / max(1, len(metrics_random.episode_rewards)),
        color="red", linestyle="--", label="Random baseline (mean)",
    )
    ax.axhline(
        sum(metrics_greedy.episode_rewards) / max(1, len(metrics_greedy.episode_rewards)),
        color="orange", linestyle="--", label="Greedy baseline (mean)",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("VerdictEnv — Reward Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    path = _ASSETS_DIR / "reward_curve.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {path}")

    # ---- win rate ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, metrics_trained.win_rate(), label="Trained (rolling win-rate)", linewidth=2)
    ax.axhline(
        sum(metrics_random.win_flags) / max(1, len(metrics_random.win_flags)),
        color="red", linestyle="--", label="Random baseline",
    )
    ax.axhline(
        sum(metrics_greedy.win_flags) / max(1, len(metrics_greedy.win_flags)),
        color="orange", linestyle="--", label="Greedy baseline",
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate")
    ax.set_title("VerdictEnv — Win Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    path = _ASSETS_DIR / "win_rate.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="VerdictEnv inference / training demo")
    p.add_argument("--task", default="medium", choices=("easy", "medium", "hard"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=200, help="Training episodes")
    p.add_argument(
        "--mode",
        default="all",
        choices=("all", "random", "greedy", "train"),
        help="all = train + compare; random/greedy = single-run baselines; train = training only",
    )
    p.add_argument(
        "--remote",
        default=None,
        metavar="URL",
        help="Connect to a running server instead of in-process (random agent, one episode)",
    )
    args = p.parse_args()

    if args.remote:
        result = asyncio.run(
            _run_episode_remote(args.remote, task=args.task, seed=args.seed)
        )
        print(
            f"[remote] episode_return={result['total_reward']:.4f} "
            f"steps={result['steps']} won={result['won']}"
        )
        return

    if args.mode == "random":
        env = VerdictEnvironment()
        res = _run_episode(env, task=args.task, seed=args.seed, action_fn=random_action)
        print(f"[random] episode_return={res['total_reward']:.4f}  won={res['won']}")
        return

    if args.mode == "greedy":
        env = VerdictEnvironment()
        res = _run_episode(env, task=args.task, seed=args.seed, action_fn=greedy_action)
        print(f"[greedy] episode_return={res['total_reward']:.4f}  won={res['won']}")
        return

    # ---- Training run + comparison ----
    N = args.episodes
    EVAL_N = max(20, N // 5)
    print(f"\nVerdictEnv  task={args.task}  train_episodes={N}  eval_episodes={EVAL_N}")
    print("=" * 58)

    print(f"[1/4] Random baseline ({EVAL_N} episodes)...")
    m_random = run_baseline(task=args.task, episodes=EVAL_N, seed_base=args.seed, mode="random")

    print(f"[2/4] Greedy baseline ({EVAL_N} episodes)...")
    m_greedy = run_baseline(task=args.task, episodes=EVAL_N, seed_base=args.seed, mode="greedy")

    print(f"[3/4] Training learned agent ({N} episodes)...")
    m_trained, agent = run_training(
        task=args.task,
        episodes=N,
        seed_base=args.seed,
    )

    agent.epsilon = 0.05
    print(f"[4/4] Evaluating trained agent ({EVAL_N} episodes, ε={agent.epsilon})...")
    m_eval = run_baseline(task=args.task, episodes=EVAL_N, seed_base=args.seed + 1000, mode="random")
    env_eval = VerdictEnvironment()
    m_eval_trained = Metrics()
    for ep in range(EVAL_N):
        seed = args.seed + 1000 + ep
        res = _run_episode(
            env_eval, task=args.task, seed=seed, action_fn=agent.pick, agent=None
        )
        m_eval_trained.record(res)

    rs = m_random.summary()
    gs = m_greedy.summary()
    ts = m_trained.summary()
    es = m_eval_trained.summary()

    print("\n" + "=" * 58)
    print("RESULTS SUMMARY")
    print("=" * 58)
    print(f"{'Agent':<22} {'Mean Reward':>12} {'Win Rate':>10} {'Case Score':>12}")
    print("-" * 58)
    print(f"{'Random (baseline)':<22} {rs.get('mean_reward', 0):>12.3f} {rs.get('win_rate', 0):>10.2%} {rs.get('mean_case_score', 0):>12.4f}")
    print(f"{'Greedy (baseline)':<22} {gs.get('mean_reward', 0):>12.3f} {gs.get('win_rate', 0):>10.2%} {gs.get('mean_case_score', 0):>12.4f}")
    print(f"{'Trained (during train)':<22} {ts.get('mean_reward', 0):>12.3f} {ts.get('win_rate', 0):>10.2%} {ts.get('mean_case_score', 0):>12.4f}")
    print(f"{'Trained (eval, ε=0)':<22} {es.get('mean_reward', 0):>12.3f} {es.get('win_rate', 0):>10.2%} {es.get('mean_case_score', 0):>12.4f}")
    print("=" * 58)

    improvement_reward = es.get("mean_reward", 0) - rs.get("mean_reward", 0)
    improvement_wr = es.get("win_rate", 0) - rs.get("win_rate", 0)
    sign_r = "+" if improvement_reward >= 0 else ""
    sign_w = "+" if improvement_wr >= 0 else ""
    print(f"\nImprovement over random  →  reward: {sign_r}{improvement_reward:.3f}   win-rate: {sign_w}{improvement_wr:.2%}")

    print("\nVerdict distribution (defense wins / total):")
    for label, m in (
        ("Random", m_random),
        ("Greedy", m_greedy),
        ("Trained (train)", m_trained),
        ("Trained (eval)", m_eval_trained),
    ):
        vd = m.verdict_dist()
        n_ep = max(1, int(m.summary().get("episodes", 1)))
        print(f"  {label:<22}: defense={vd.get('defense', 0):>4}/{n_ep}  prosecution={vd.get('prosecution', 0):>4}/{n_ep}")

    print("\nLearned action scores:")
    for k, v in agent.summary().items():
        print(f"  {k:<22}: {v:>+.4f}")

    _plot(m_random, m_greedy, m_trained)
    print("\nDone. Plots saved to:", _ASSETS_DIR)


if __name__ == "__main__":
    main()
