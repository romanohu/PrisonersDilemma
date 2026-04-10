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

## Partner Head Encoding

- action head `partner` is `Discrete(num_agents - 1)`
- each agent chooses among all other agents in ascending id order with self removed

Example for `num_agents=4`:

- agent 0: `0->1, 1->2, 2->3`
- agent 1: `0->0, 1->2, 2->3`
- agent 2: `0->0, 1->1, 2->3`
- agent 3: `0->0, 1->1, 2->2`

## Matching Modes

- `from_actions`
  - partner map comes from the partner action head each round.
- `random`
  - partner map is sampled uniformly at each round boundary.

Compatibility aliases `random_with_replacement` and `random_with_replacement_each_step` are normalized to `random`.

## Partner Override API

You can override the partner map for the next round once:

```python
env.set_partners([1, 0, 0, 2, 3, 4, 5, 6])
```

Or pass at reset:

```python
obs, infos = env.reset(options={"partners": [1, 0, 0, 2, 3, 4, 5, 6]})
```

Constraints for `partners`:

- length must be `num_agents`
- each id must be in `[0, num_agents)`
- self-partnering (`partners[i] == i`) is not allowed

## Observation / Action Summary

- Action space per agent:
  - `Tuple(Discrete(num_agents - 1), Discrete(2))`
  - first head: partner
  - second head: PD action (`0:C`, `1:D`)

- Observation per agent:
  - shape: `((num_agents - 1) * 2 * history_h + 1,)`
  - concatenated one-hot PD history of all other agents + one phase flag
  - phase flag: `0.0` (round boundary / partner selection), `1.0` (dilemma)
  - during dilemma, only the currently interacting pair observes each other; non-participants are zeros

See `docs/environment_api.md` for detailed transition semantics and info fields.
