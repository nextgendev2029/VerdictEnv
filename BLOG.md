# VerdictEnv: Teaching AI to Win in Court

*OpenEnv Hackathon 2026 — Submission Blog*

---

## The Problem

LLMs are surprisingly good at answering legal questions in isolation. Ask GPT "is this evidence
admissible?" and it gives a reasonable answer. But put it in a courtroom — where every decision
affects the next, where the jury's mood shifts with each move, where a mistimed objection can
undo ten minutes of good work — and it falls apart.

The gap is not knowledge. It is **sequential judgment under uncertainty**.

A defense attorney does not just know the law. They read the jury, choose *which* evidence to
present *when*, decide whether to object or let a weak prosecution point slide, and build a
narrative across four distinct phases. That is the kind of multi-step reasoning that current LLMs
consistently struggle with — and that RL is uniquely suited to train.

VerdictEnv puts an agent in that seat.

---

## Why Courtroom RL is Hard

Three things make this environment genuinely non-trivial:

**1. Phase-dependent strategy.** An action that scores well in the defense case phase can be
catastrophic in closing. Presenting weak evidence early poisons the jury. Objecting in closing
looks desperate. The agent must learn *when* each action is appropriate — not just *what* each
action does.

**2. Partial observability.** The agent sees jury sentiment (analytical / skeptical / emotional)
and available evidence — but not the prosecution's next move or the jury's internal threshold.
Every decision is made under uncertainty.

**3. A reward signal that is easy to game badly.** A naive agent learns to spam `pass` — it avoids
penalties and gets small positive rewards but never wins. The environment is designed so that
genuinely winning (defense verdict) requires building real case strength, not exploiting the
reward structure.

---

## Environment Design

The trial runs across four phases: `opening → prosecution_case → defense_case → closing → verdict`.

At each step the agent receives an observation containing jury sentiment, available evidence with
strength scores, the current phase, a running case score, and the list of valid actions. It then
chooses one of:

- **`present_evidence[id]`** — plays a piece of evidence; reward tracks the jury's analytical
  shift in response
- **`object(type)`** — raises hearsay, relevance, or speculation objection during prosecution;
  correctly typed objections score, wrong types penalise
- **`pass`** — yields the step; safe but gives up momentum

The reward is a **multi-component signal**: change in analytical sentiment, a verdict bonus
(±5.0 at terminal), procedure reward for correct objection type, and a timing bonus for
phase-appropriate actions. The measured range across training is **[-12.0, +9.0]** —
wide enough to provide a rich learning signal across the full episode.

---

## Training Results

We trained a tabular Q-learning agent with ε-greedy exploration (ε: 0.30 → 0.04) for 200
episodes on the `medium` task and evaluated for 40 episodes at ε = 0.05.

| Agent | Mean Reward | Win Rate |
|-------|-------------|----------|
| Random baseline | -3.3 | 60% |
| Greedy baseline | -13.8 | 60% |
| **Trained agent** | **+8.0** | **72.5%** |

**The trained agent improves mean reward by +11.3 and win rate by +12.5 percentage points
over the random baseline.** The Q-table reveals interpretable learned behavior: it objects
during prosecution (+1.28 Q-value), avoids weak evidence in defense (-0.44), and never
objects in closing (-2.01 — correctly learned as a penalty).

The greedy baseline — which always plays the highest-strength evidence — actually performs
*worse* than random because it ignores phase context entirely. This confirms the environment
tests genuine sequential reasoning, not just evidence quality ranking.

---

## Future: LLM Training with TRL / Unsloth

The tabular agent proves the reward signal is learnable and the environment is non-trivial.
The natural next step is to replace the Q-table with a language model policy trained via
**GRPO or PPO using Hugging Face TRL or Unsloth**.

The environment is already structured for this:

- Observations serialize cleanly to text (jury sentiment, phase, evidence list)
- Actions are a small, typed vocabulary — ideal for constrained LLM decoding
- The reward signal is dense and informative — no sparse terminal-only reward
- The OpenEnv client protocol makes it straightforward to wire a TRL training loop

A small model (Qwen-1.5B or Phi-3-mini with QLoRA) trained on this environment would develop
**phase-aware legal reasoning** — a capability with direct applications in contract analysis,
compliance review, and structured argumentation.

---

## Try It

- **Live demo:** [HF Space](https://huggingface.co/spaces/nextgendev2029/VerdictEnv)
- **Training notebook:** [Google Colab](https://colab.research.google.com/github/nextgendev2029/VerdictEnv/blob/main/VerdictEnv_Colab.ipynb)
- **Code:** [GitHub / HF Hub](https://huggingface.co/spaces/nextgendev2029/VerdictEnv)

---

*Built for the OpenEnv Hackathon 2026.*
