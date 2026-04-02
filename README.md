# PrisonersDilemma Environment

This repository provides a multi-agent repeated Prisoner's Dilemma environment with a clear separation of concerns:

- `core.py`: pairwise 2x2 PD reward core (`PairwisePrisonersDilemmaCore`)
- `schedulers.py`: partner-selection scheduler interface and defaults
- `prisoners_dilemma_env.py`: Gymnasium environment wrapper (`PrisonersDilemmaEnv`)

The game core is always pairwise PD. Partner selection is delegated to an external scheduler.

## Setup

Initialize submodules:

```bash
git submodule update --init --recursive
```

Install runtime dependencies:

```bash
pip install gymnasium numpy
```

## Minimal Usage

```python
from PrisonersDilemma import PrisonersDilemmaEnv

env = PrisonersDilemmaEnv(
    num_agents=20,
    max_steps=150,
    history_h=1,
)

obs, infos = env.reset(seed=7)
for _ in range(5):
    # 0 = Cooperate, 1 = Defect
    actions = [0] * env.num_agents
    obs, rewards, terminations, truncations, infos = env.step(actions)
    if all(terminations):
        break

env.close()
```

## Custom Scheduler Example

```python
import numpy as np
from PrisonersDilemma import PrisonersDilemmaEnv, InteractionScheduler


class RoundRobinScheduler(InteractionScheduler):
    def select_partners(self, *, num_agents, rng, action_history, step):
        del rng, action_history
        idx = np.arange(num_agents, dtype=np.int32)
        return (idx + step + 1) % num_agents


env = PrisonersDilemmaEnv(num_agents=8, scheduler=RoundRobinScheduler())
```

## Notes

- The default scheduler is uniform random partner selection with replacement.
- Each step runs `num_agents` directed pairwise interactions (`i -> partner[i]`).
- Observation is partner-behavior history only, shaped `(2 * history_h,)`.

For full transition semantics, see `docs/environment_api.md`.
