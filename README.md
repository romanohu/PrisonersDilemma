# PrisonersDilemma Environment

This repository provides Gymnasium environments for repeated Prisoner's Dilemma (PD) experiments.

## Components

- `core.py`
  - `PairwisePrisonersDilemmaCore`: 2x2 PD payoff core for directed interactions.
- `population_prisoners_dilemma_env.py`
  - `PopulationPrisonersDilemmaEnv`: population environment (`num_agents >= 2`).
- `prisoners_dilemma_env.py`
  - `PrisonersDilemmaEnv`: minimal fixed 2-agent environment.

Both environments export a Sample Factory-compatible multi-agent style API:

- `reset(...) -> (observations, infos)`
- `step(actions) -> (observations, rewards, terminations, truncations, infos)`

## Setup

```bash
pip install gymnasium numpy
```

## Quick Start

### Population Environment (recommended)

```python
from PrisonersDilemma import PopulationPrisonersDilemmaEnv

env = PopulationPrisonersDilemmaEnv(
    num_agents=8,
    max_steps=150,
    history_h=2,
    partner_scheduler="random_with_replacement",
)

obs, infos = env.reset(seed=7)
actions = [0] * env.num_agents  # 0 = C, 1 = D
obs, rewards, terminations, truncations, infos = env.step(actions)
```

You can also pass explicit partners at reset:

```python
obs, infos = env.reset(options={"partners": [1, 0, 0, 2, 3, 4, 5, 6]})
```

Constraints for `options["partners"]`:

- length must be `num_agents`
- each id must be in `[0, num_agents)`
- self-partnering (`partners[i] == i`) is not allowed

### Fixed 2-Agent Environment

```python
from PrisonersDilemma import PrisonersDilemmaEnv

env = PrisonersDilemmaEnv(num_agents=2, max_steps=150, history_h=1)
obs, infos = env.reset(seed=7)
obs, rewards, terminations, truncations, infos = env.step([0, 1])
```

## Current Behavior Notes

- Action space is `Discrete(2)` for both envs:
  - `0`: Cooperate (`C`)
  - `1`: Defect (`D`)
- Observation is partner-action history encoded as one-hot features of shape `(2 * history_h,)`.
- `PopulationPrisonersDilemmaEnv` currently supports only:
  - `partner_scheduler="random_with_replacement"`
- In the population env, partners are chosen at `reset` and remain fixed within an episode.
- If `step` is called after termination, env auto-resets and returns zero rewards with non-terminal flags.

See `docs/environment_api.md` for complete transition semantics and info fields.
