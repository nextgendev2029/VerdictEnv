---
title: VerdictEnv — AI Courtroom Simulator
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ⚖️ VerdictEnv — AI Courtroom RL Environment

[![Open in Spaces](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/tuhindev2029/VerdictEnv)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nextgendev2029/VerdictEnv/blob/main/VerdictEnv_Colab.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](openenv.yaml)

> **A turn-based courtroom trial environment where an AI agent learns to present evidence,
> raise objections, and manage jury sentiment to win the verdict — purely through
> reinforcement learning.**

---

## 🧠 Why This Matters

India has **5.58 crore (55.8M) pending court cases** — a backlog growing at 5.8% per year.
Over 1.8 lakh cases have been stuck for **30+ years**. The system has 15 judges per million
people; the US has 150. Justice is delayed at scale, and the bottleneck is human bandwidth.

Meanwhile, frontier LLMs fail at real legal reasoning. Yale/MIT's *CourtReasoner* benchmark
(EMNLP 2025) found that **>60% of GPT-4/Claude outputs contain invalid legal arguments**
when asked to construct full judicial analyses — not answer trivia, but *reason through cases*.

The gap is not knowledge. It is **sequential judgment under uncertainty**: reading the room,
timing moves, building a case across phases. That is what trial lawyers do. That is what LLMs
cannot do. That is what VerdictEnv trains.

> While we demonstrate training using **tabular Q-learning**, the environment is designed for
> **LLM-based policy learning using TRL / Unsloth**. The Q-learning baseline proves the reward
> signal is learnable and the environment is genuinely non-trivial.

---

## 🎮 How the Environment Works

A courtroom trial unfolds across **four phases**. Each phase has different strategic implications.
The agent plays defense counsel.

```
┌─────────────────────────────────────────────────────────────────┐
│                        TRIAL PHASES                             │
│  opening → prosecution_case → defense_case → closing → verdict  │
└─────────────────────────────────────────────────────────────────┘
         ↓ each step ↓
   Agent observes:               Agent chooses:
   • jury_sentiment               • present_evidence[id]
     {analytical, skeptical,      • object(hearsay|relevance|speculation)
      emotional}                  • pass
   • available_evidence
   • current phase
   • case_score
         ↓
   Reward = Δ case_score + verdict bonus + procedure reward + timing bonus
```

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `jury_sentiment` | dict | `{analytical, skeptical, emotional}` — floats ∈ [0,1] |
| `available_evidence` | list | Each item: `{id, strength ∈ [0,1], type}` |
| `phase` | str | Current trial phase |
| `case_score` | float | Running defense case strength |
| `valid_actions` | list | Legal moves in this state |
| `done` | bool | Episode terminal flag |

### Action Space

| Action | When effective | Risk |
|--------|---------------|------|
| `present_evidence` | Defense case, opening | Low-strength evidence hurts |
| `object` | During prosecution | Wrong objection type = penalty |
| `pass` | Any phase | Safe but surrenders momentum |

### Reward Design

**Multi-component, dense signal** — not a single 0/1 at episode end:

```
reward = Δ analytical_sentiment × 2.0     (jury reacts to your move)
       + verdict_bonus (±5.0 at terminal)  (win or lose the case)
       + procedure_reward                  (correct objection type)
       + timing_bonus                      (phase-appropriate action)
       - penalty                           (bad objections, weak evidence)
```

**Measured range:** `[-12.0, +9.0]` across 200 episodes.

### Why This Environment is Hard

The **greedy baseline performs worse than random**:

| Agent | Mean Reward |
|-------|-------------|
| Random | -3.3 |
| **Greedy** | **-13.8** |
| Trained | **+8.0** |

A greedy agent dumps all evidence early, ignores phase context, and never objects.
The environment *punishes* brute-force strategies. If greedy beats trained, your env is
trivial. In VerdictEnv, greedy doesn't even beat random.

### Task Difficulty

| Task | Evidence | Max steps | Description |
|------|----------|-----------|-------------|
| `easy` | 3 items | 12 | Strong evidence pool, lenient jury |
| `medium` | 5 items | 20 | Balanced — recommended for training |
| `hard` | 8 items | 32 | Weak evidence, skeptical jury |

---

## 📊 Training Results

**Algorithm:** Tabular Q-learning, ε-greedy (ε: 0.30 → 0.04 over 200 episodes)
**Task:** `medium` | **Seed:** 42 | **Eval episodes:** 40

### Agent Comparison

| Agent | Mean Reward | Win Rate | Case Score |
|-------|-------------|----------|------------|
| Random (baseline) | -3.315 | 60.0% | 0.1721 |
| Greedy (baseline) | -13.765 | 60.0% | 0.1746 |
| **Trained (during training)** | **+7.433** | **73.5%** | **0.1927** |
| **Trained (eval, ε=0.05)** | **+7.998** | **72.5%** | **0.1915** |

**Improvement over random → reward: +11.313 | win-rate: +12.5%**

### Verdict Distribution

| Agent | Defense wins | Prosecution wins |
|-------|-------------|-----------------|
| Random | 24 / 40 | 16 / 40 |
| Greedy | 24 / 40 | 16 / 40 |
| Trained (train) | **147 / 200** | 53 / 200 |
| Trained (eval) | **29 / 40** | 11 / 40 |

### What the Agent Learned

The Q-table reveals an interpretable strategy — not noise:

| State × Action | Q-value | What it means |
|---------------|---------|--------------|
| `prosecution` + `object` | **+1.28** | Disrupts the opponent — correctly timed |
| `defense` + `present_mid` | **+0.99** | Medium-strength evidence is most reliable |
| `defense` + `present_lo` | **-0.44** | Weak evidence hurts — learned to avoid |
| `closing` + `object` | **-2.01** | Objections in closing = desperation penalty |
| `opening` + `pass` | **+0.99** | Patience early — let prosecution overextend |

### Training Plots

**Reward Curve** — agent consistently outperforms both baselines after ~50 episodes:

![Reward Curve](assets/reward_curve.png)

**Win Rate** — rolling win rate climbs from 60% (random) to stable ~73%:

![Win Rate](assets/win_rate.png)

---

## 🚀 Quick Start

### Live on HF Space

| | URL |
|--|-----|
| Gradio UI | https://tuhindev2029-verdictenv.hf.space/ |
| API docs (Swagger) | https://tuhindev2029-verdictenv.hf.space/docs |
| Health check | https://tuhindev2029-verdictenv.hf.space/api/health |

### Run locally

```bash
git clone https://github.com/nextgendev2029/VerdictEnv.git
cd VerdictEnv
pip install -e .
```

**Train (200 episodes, medium task):**

```bash
python -m verdict_env.inference --task medium --episodes 200
```

**Launch UI + API:**

```bash
python -m verdict_env.server.app
# Gradio UI  → http://localhost:7860/
# API docs   → http://localhost:7860/docs
```

### Docker

```bash
docker build -t verdictenv .
docker run -p 7860:7860 verdictenv
```

### API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/reset` | Reset environment, get initial observation |
| `POST` | `/api/step` | Take action, get next state + reward |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 🏗️ Architecture

```
verdict_env/
├── models.py          # VerdictAction, VerdictObservation, VerdictState (Pydantic)
├── tasks.py           # Task configs (easy/medium/hard) + grading functions
├── inference.py       # Q-learning agent, training loop, baselines, plotting
├── client.py          # WebSocket client (EnvClient / OpenEnv protocol)
└── server/
    ├── environment.py # VerdictEnvironment — core game logic, phases, reward
    ├── app.py         # FastAPI REST API + Gradio UI (served together)
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

Full training pipeline — runs end-to-end in Google Colab, no setup required:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nextgendev2029/VerdictEnv/blob/main/VerdictEnv_Colab.ipynb)

**Notebook covers:**
1. Install and import
2. Random + greedy baselines (40 episodes each)
3. Q-learning training (200 episodes, ε-greedy)
4. Evaluation at ε = 0.05
5. Results table + verdict distribution
6. Q-table interpretation
7. Single episode step-by-step walkthrough
8. Inline reward curve + win rate plots

---

## 🔗 All Links

| Resource | Link |
|----------|------|
| 🚀 HF Space | [tuhindev2029/VerdictEnv](https://huggingface.co/spaces/tuhindev2029/VerdictEnv) |
| 🌐 Gradio UI (direct) | [tuhindev2029-verdictenv.hf.space](https://tuhindev2029-verdictenv.hf.space/) |
| 📖 API Docs | [Swagger](https://tuhindev2029-verdictenv.hf.space/docs) |
| 📓 Colab Notebook | [VerdictEnv_Colab.ipynb](https://colab.research.google.com/github/nextgendev2029/VerdictEnv/blob/main/VerdictEnv_Colab.ipynb) |
| 📝 Blog / Writeup | [BLOG.md](https://huggingface.co/spaces/tuhindev2029/VerdictEnv/blob/main/BLOG.md) |
| ⚙️ OpenEnv Manifest | [openenv.yaml](openenv.yaml) |
| 📜 License | [MIT](LICENSE) |

---

## 🧪 OpenEnv Protocol Example

```python
import asyncio
from verdict_env.client import VerdictEnv
from verdict_env.models import VerdictAction

async def main():
    async with VerdictEnv("https://tuhindev2029-verdictenv.hf.space") as env:
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
  author  = {tuhindev2029},
  year    = {2026},
  url     = {https://huggingface.co/spaces/tuhindev2029/VerdictEnv}
}
```

---

*Built for the OpenEnv Hackathon India 2026.*
*5.58 crore cases are waiting. The system needs help. AI can learn to provide it.*
