# PrisonersDilemma Environment

This repository provides a repeated Prisoner's Dilemma environment implemented with Gymnasium conventions and Axelrod game primitives.

## Repository Layout

- `prisoners_dilemma_env.py`: core environment (`PrisonersDilemmaEnv`)
- `axelrod_adapter.py`: loading Axelrod `Action` and `Game`
- `docs/environment_api.md`: detailed API and transition semantics
- `Axelrod/`: upstream Axelrod submodule

## Setup

Initialize submodules:

```bash
git submodule update --init --recursive
```

Install runtime dependencies:

```bash
pip install gymnasium numpy
```

If `axelrod` is installed in your Python environment, it is used directly. Otherwise, this repository falls back to loading minimal `Action`/`Game` classes from `Axelrod/axelrod/`.

## Minimal Usage

```python
from prisoners_dilemma_env import PrisonersDilemmaEnv

env = PrisonersDilemmaEnv(num_agents=2, max_steps=5)
obs, infos = env.reset(seed=7)
print("reset obs shape:", obs[0]["obs"].shape)

for t in range(5):
    # 0 = Cooperate, 1 = Defect
    obs, rewards, terminations, truncations, infos = env.step([0, 1])
    print(t, rewards, terminations, truncations)
    if all(terminations):
        break

env.close()
```

## `reset` and `step` Return Contract (Quick View)

`reset(seed=None, options=None)` returns:

1. `observations`: `list[dict[str, np.ndarray]]` with length `num_agents`
2. `infos`: `list[dict]` with length `num_agents`

`step(actions)` returns:

1. `observations`: same structure as `reset`
2. `rewards`: `list[np.float32]`, one reward per agent
3. `terminations`: `list[bool]`
4. `truncations`: `list[bool]`
5. `infos`: `list[dict]`

For full details (observation layout, reward semantics, scripted policies, post-termination behavior), see:

- `docs/environment_api.md`

## References

- Gymnasium Env API: https://gymnasium.farama.org/main/api/env/
- Sample Factory custom environment integration: https://www.samplefactory.dev/03-customization/custom-environments/
- Axelrod documentation: https://axelrod.readthedocs.io/en/stable/index.html
