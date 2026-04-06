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
    partner_scheduler="random_with_replacement_each_step",
)

obs, infos = env.reset(seed=7)
actions = [0] * env.num_agents  # backward-compatible shortcut (broadcast to all opponents)
obs, rewards, terminations, truncations, infos = env.step(actions)

# per-opponent actions (row i has actions vs every opponent except i)
actions_per_opponent = [
    [0] * (env.num_agents - 1) for _ in range(env.num_agents)
]
obs, rewards, terminations, truncations, infos = env.step(actions_per_opponent)
```

You can also pass explicit partners at reset:

```python
obs, infos = env.reset(options={"partners": [1, 0, 0, 2, 3, 4, 5, 6]})
```

Constraints for `options["partners"]`:

- length must be `num_agents`
- each id must be in `[0, num_agents)`
- self-partnering (`partners[i] == i`) is not allowed

You can also set partners directly before a step:

```python
env.set_partners([1, 0, 1, 2, 3, 4, 5, 6])
```

### Fixed 2-Agent Environment

```python
from PrisonersDilemma import PrisonersDilemmaEnv

env = PrisonersDilemmaEnv(num_agents=2, max_steps=150, history_h=1)
obs, infos = env.reset(seed=7)
obs, rewards, terminations, truncations, infos = env.step([0, 1])
```

## Current Behavior Notes

- `PrisonersDilemmaEnv` action space is `Discrete(2)`:
  - `0`: Cooperate (`C`)
  - `1`: Defect (`D`)
- `PopulationPrisonersDilemmaEnv` action space is `MultiDiscrete([2] * (num_agents - 1))`:
  - each agent outputs one C/D action per possible opponent (excluding self)
  - legacy input `[a_0, ..., a_{N-1}]` is still accepted and broadcast per opponent
- Observation in population env is all-other-agents action history, shape
  `(2 * history_h * (num_agents - 1))`.
- `PopulationPrisonersDilemmaEnv` supports:
  - `partner_scheduler="random_with_replacement"`
  - `partner_scheduler="random_with_replacement_each_step"`
- Scheduler behavior:
  - `random_with_replacement`: partners are sampled at `reset` and fixed for the episode.
  - `random_with_replacement_each_step`: partners are resampled every non-terminal step.
- `set_partners(...)` overrides the current partner assignment immediately (after validation).
- If `step` is called after termination, env auto-resets and returns zero rewards with non-terminal flags.

See `docs/environment_api.md` for complete transition semantics and info fields.
