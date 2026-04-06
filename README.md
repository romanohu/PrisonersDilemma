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
actions = [0] * env.num_agents  # one C/D action per agent
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
- `PopulationPrisonersDilemmaEnv` action space is also `Discrete(2)` per agent:
  - each agent outputs one C/D action at every environment step
- Population step/round semantics:
  - one environment step processes one full round
  - one round contains `num_agents` directed interactions (`i -> partners[i]` for all `i`)
  - if agent `j` is selected by multiple selectors, selected-side rewards are accumulated in the same step
  - each agent outputs one action per step, reused across all incoming interactions in that step
- Observation in population env is partner-action history for each agent's selected partner,
  shape `(2 * history_h)` (all agents active on non-terminal steps).
- History update detail:
  - the history stored for agent `i` at round `t` is the selector-side action used in
    `i -> partners[i]` in that round
  - selected-side responses from multiple incoming interactions are not stored
    as separate history entries
- `PopulationPrisonersDilemmaEnv` supports:
  - `partner_scheduler="random_with_replacement"`
  - `partner_scheduler="random_with_replacement_each_step"`
- Scheduler behavior:
  - `random_with_replacement`: partners are sampled at `reset` and fixed unless overridden.
  - `random_with_replacement_each_step`: partners are resampled for the next step round.
- `set_partners(...)` stores a pending override that takes effect on the following step.
- If `step` is called after termination, env auto-resets and returns zero rewards with non-terminal flags.

See `docs/environment_api.md` for complete transition semantics and info fields.
