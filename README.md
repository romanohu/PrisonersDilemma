# PrisonersDilemma Environment

This package provides a Gymnasium multi-agent environment for repeated population Prisoner's Dilemma (PD) experiments with partner selection.

## Components

- `core.py`
  - `PairwisePrisonersDilemmaCore`: 2x2 PD payoff core.
- `prisoners_dilemma_env.py`
  - `PrisonersDilemmaEnv`: fixed 2-agent repeated PD environment.
- `population_prisoners_dilemma_env.py`
  - `PopulationPrisonersDilemmaEnv`: population environment (`num_agents >= 2`).

## Setup

```bash
pip install gymnasium numpy
```

## Quick Start

### 2-agent repeated PD

```python
from PrisonersDilemma import PrisonersDilemmaEnv

env = PrisonersDilemmaEnv(
    num_agents=2,
    max_steps=150,
    history_h=1,
)

obs, infos = env.reset(seed=7)
obs, rewards, terminations, truncations, infos = env.step([0, 1])  # 0:C, 1:D
```

### Population PD

```python
from PrisonersDilemma import PopulationPrisonersDilemmaEnv

env = PopulationPrisonersDilemmaEnv(
    num_agents=8,
    max_steps=150,      # number of rounds per episode
    history_h=1,
    partner_scheduler="from_actions",  # or "random"
)

obs, infos = env.reset(seed=7)

# 1) Round-start step:
#    partner map is fixed for this round and selector 0 interaction is executed.
dilemma_actions = [(0, 1)] * env.num_agents
obs, rewards, terminations, truncations, infos = env.step(dilemma_actions)

# 2) Remaining dilemma substeps: one directed interaction per step.
for _ in range(env.num_agents - 1):
    dilemma_actions = [(0, 1)] * env.num_agents
    obs, rewards, terminations, truncations, infos = env.step(dilemma_actions)
```