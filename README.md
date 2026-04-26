---
title: VerdictEnv — AI Courtroom Simulator
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.13.0"
app_file: server/app.py
app_port: 7860
pinned: false
license: mit
---

# ⚖️ VerdictEnv — AI Courtroom RL Environment

[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/nextgendev2029/VerdictEnv)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nextgendev2029/VerdictEnv/blob/main/VerdictEnv_Colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](openenv.yaml)

> **A turn-based courtroom trial environment for training AI agents to reason under uncertainty.**
> The agent plays defense counsel — selecting evidence, raising objections, and managing jury sentiment
> across structured trial phases to win the verdict.

---

## 🧠 Why This Matters

LLMs struggle with **sequential decision-making under partial information** — exactly what a trial lawyer does every day.
A defense attorney must:

- **Read the room** — gauge jury sentiment and adapt strategy in real time
- **Prioritize evidence** — not all evidence helps in all phases
- **Time objections** — objecting at the wrong moment backfires
- **Build a narrative** — across four distinct phases (opening → prosecution → defense → closing)

This is not a game. It is a **structured test of causal reasoning, strategic sequencing, and contextual judgment** —
capabilities that matter far beyond courtrooms (negotiation, medical diagnosis, policy planning).

> While we demonstrate training using **tabular Q-learning**, the environment is designed for
> **LLM-based policy learning using TRL / Unsloth**. The Q-learning baseline proves the reward
> signal is learnable and the environment is non-trivial.

---

## 🎮 Environment Design

### How it works

```
┌─────────────────────────────────────────────────────────────┐
│                     TRIAL PHASES                            │
│  opening → prosecution_case → defense_case → closing → verdict │
└─────────────────────────────────────────────────────────────┘
         ↓ each step ↓
   Agent observes:          Agent chooses:
   • jury_sentiment          • present_evidence[id]
     (analytical/skeptical/  • object(hearsay|relevance
      emotional)               |speculation)
   • available_evidence      • pass
   • current phase
   • case_score
         ↓
   Reward = Δ case_score + phase bonus + procedure penalty
```

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `jury_sentiment` | dict | `{analytical, skeptical, emotional}` floats ∈ [0,1] |
| `available_evidence` | list | Each item has `id`, `strength` ∈ [0,1], `type` |
| `phase` | str | Current trial phase |
| `case_score` | float | Scalar measure of defense case strength |
| `valid_actions` | list | Legal moves in this state |
| `done` | bool | Episode terminal flag |

### Action Space

| Action | When effective | Risk |
|--------|---------------|------|
| `present_evidence` | All phases | Low-strength evidence hurts |
| `object` | During prosecution | Wrong objection type = penalty |
| `pass` | Any phase | Safe but gives up momentum |

### Reward Signal

The reward is **multi-component** — not a single end-of-episode signal:

```
reward = Δ analytical_sentiment × 2.0
       + verdict_bonus (±5.0 at terminal)
       + procedure_reward (correct objection type)
       + timing_bonus (phase-appropriate action)
       - penalty (invalid objection, weak evidence in wrong phase)
```

**Measured range across 200 training episodes:** `[-12.0, +9.0]`

### Task Difficulty

| Task | Evidence items | Max steps | Description |
|------|---------------|-----------|-------------|
| `easy` | 3 | 12 | Tight record, strong evidence pool |
| `medium` | 5 | 20 | Mixed docket, balanced difficulty |
| `hard` | 8 | 32 | Weak evidence, skeptical jury |

---

## 📊 Training Results

Trained using **tabular Q-learning with ε-greedy exploration** (ε: 0.30 → 0.04 over 200 episodes).
Task: `medium` | Seed: 42 | Eval episodes: 40.

### Results Table

| Agent | Mean Reward | Win Rate | Case Score |
|-------|-------------|----------|------------|
| Random (baseline) | -3.315 | 60.0% | 0.1721 |
| Greedy (baseline) | -13.765 | 60.0% | 0.1746 |
| **Trained — Q-learning (during train)** | **+7.433** | **73.5%** | **0.1927** |
| **Trained — Q-learning (eval, ε=0.05)** | **+7.998** | **72.5%** | **0.1915** |

**Improvement over random baseline → reward: +11.313 | win-rate: +12.5%**

Verdict distribution (defense wins / total):

| Agent | Defense wins | Prosecution wins |
|-------|-------------|-----------------|
| Random | 24 / 40 | 16 / 40 |
| Greedy | 24 / 40 | 16 / 40 |
| Trained (train) | **147 / 200** | 53 / 200 |
| Trained (eval) | **29 / 40** | 11 / 40 |

### What the agent learned

The Q-table reveals clear learned behavior — not random improvement:

- `prosecution_case + object` → **+1.24 to +1.28** — agent learned to object during prosecution
- `defense_case + presentmid` → **+0.99** — medium-strength evidence is most reliable
- `defense_case + presentlo` → **-0.44 to -0.51** — agent learned to avoid weak evidence
- `closing + object` → **-2.01** — objections in closing = penalty (correctly avoided)

### Training Plots

**Reward Curve** — trained agent consistently outperforms both baselines after ~50 episodes:

![Reward Curve](assets/reward_curve.png)

**Win Rate** — rolling win rate climbs from 60% (random) to ~73% (trained):

![Win Rate](assets/win_rate.png)

---

## 🚀 Quick Start

### Run locally

```bash
git clone https://github.com/nextgendev2029/VerdictEnv.git
cd VerdictEnv
pip install -e .
```

**Train the agent (200 episodes, medium task):**

```bash
python -m verdict_env.inference --task medium --episodes 200
```

**Launch the interactive UI + API server:**

```bash
python -m verdict_env.server.app
# Gradio UI  → http://localhost:7860/
# API docs   → http://localhost:7860/docs
# Health     → http://localhost:7860/api/health
```

### Docker

```bash
docker build -t verdictenv .
docker run -p 7860:7860 verdictenv
```

### API endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/reset` | Reset environment, get initial observation |
| `POST` | `/api/step` | Take action, get next observation + reward |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 🏗️ Architecture

```
verdict_env/
├── models.py          # VerdictAction, VerdictObservation, VerdictState (Pydantic)
├── tasks.py           # Task configs (easy/medium/hard) + grading functions
├── inference.py       # Q-learning agent, training loop, baselines, plotting
├── client.py          # WebSocket client (EnvClient protocol)
└── server/
    ├── environment.py # VerdictEnvironment — core game logic and reward
    ├── app.py         # FastAPI (REST API) + Gradio UI, served together
    └── requirements.txt
```

**OpenEnv compliance:**
- `VerdictEnvironment` subclasses `Environment[VerdictAction, VerdictObservation, VerdictState]`
- `reset()` → seeded, reproducible initial observation
- `step()` → full observation with reward, done flag, valid actions
- `state` property → `VerdictState` with current phase
- All entry points in `openenv.yaml` resolve to live code

---

## 📓 Training Notebook

The full training pipeline runs end-to-end in Google Colab — no setup required:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nextgendev2029/VerdictEnv/blob/main/VerdictEnv_Colab.ipynb)

The notebook covers:
1. Install and import
2. Random + greedy baselines
3. Q-learning training (200 episodes)
4. Evaluation (ε = 0.05)
5. Results table + verdict distribution
6. Q-table inspection
7. Single episode walkthrough (step-by-step)
8. Inline reward curve + win rate plots

---

## 🔗 Links

| Resource | Link |
|----------|------|
| 🚀 HF Space (live demo) | [nextgendev2029/VerdictEnv](https://huggingface.co/spaces/nextgendev2029/VerdictEnv) |
| 📓 Colab Notebook | [VerdictEnv_Colab.ipynb](https://colab.research.google.com/github/nextgendev2029/VerdictEnv/blob/main/VerdictEnv_Colab.ipynb) |
| 📝 Blog / Writeup | [BLOG.md](BLOG.md) |
| ⚙️ OpenEnv Manifest | [openenv.yaml](openenv.yaml) |
| 📜 License | [MIT](LICENSE) |

---

## 🧪 OpenEnv Protocol

```python
import asyncio
from verdict_env.client import VerdictEnv
from verdict_env.models import VerdictAction

async def main():
    async with VerdictEnv("http://localhost:7860") as env:
        r = await env.reset(task="medium", seed=42)
        obs = r.observation
        while not r.done:
            act = VerdictAction(action_type="pass")
            r = await env.step(act)
        print("Final case score:", r.observation.case_score)

asyncio.run(main())
```

---

## 📋 Citation

```bibtex
@misc{verdictenv2026,
  title   = {VerdictEnv: A Courtroom RL Environment for Sequential Legal Reasoning},
  author  = {nextgendev2029},
  year    = {2026},
  url     = {https://huggingface.co/spaces/nextgendev2029/VerdictEnv}
}
```

---

*Built for the OpenEnv Hackathon 2026 · MIT License*
