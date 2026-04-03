# PrisonersDilemma Environment

This repository provides a minimal repeated Prisoner's Dilemma environment for Sample Factory integration:

- `core.py`: pairwise 2x2 PD payoff definition (`PairwisePrisonersDilemmaCore`)
- `prisoners_dilemma_env.py`: Gymnasium environment wrapper (`PrisonersDilemmaEnv`, 2-agent fixed duel)

The environment is intentionally minimal: exactly 2 agents play one PD game per step.

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
    num_agents=2,
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

## Notes

- `num_agents` must be `2`.
- Pairing is fixed as `0 <-> 1`.
- Observation is partner-behavior history only, shaped `(2 * history_h,)`.

For full transition semantics, see `docs/environment_api.md`.
