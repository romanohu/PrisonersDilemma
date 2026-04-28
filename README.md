# PrisonersDilemma Environment

GymnasiumベースのマルチエージェントPD環境群です。

公開クラス:
- `PairwisePrisonersDilemmaCore`
- `PrisonersDilemmaEnv`（2-agent）
- `PopulationPrisonersDilemmaEnv`
- `PairedPopulationPrisonersDilemmaEnv`

## Install

```bash
pip install gymnasium numpy
```

## Quick Start

### 1) 2-agent repeated PD

```python
from PrisonersDilemma import PrisonersDilemmaEnv

env = PrisonersDilemmaEnv(num_agents=2, max_steps=150, history_h=1)
obs, infos = env.reset(seed=7)
obs, rewards, terminations, truncations, infos = env.step([0, 1])  # 0:C, 1:D
```

### 2) Population PD (sequential directed interactions)

```python
from PrisonersDilemma import PopulationPrisonersDilemmaEnv

env = PopulationPrisonersDilemmaEnv(
    num_agents=8,
    max_steps=150,
    history_h=1,
    partner_scheduler="from_actions",  # or "random"
)

obs, infos = env.reset(seed=7)
actions = [(0, 1)] * env.num_agents  # (partner_rel, pd)
obs, rewards, terminations, truncations, infos = env.step(actions)
```

### 3) Paired Population PD (matching + slot-packed local games)

```python
from PrisonersDilemma import PairedPopulationPrisonersDilemmaEnv

env = PairedPopulationPrisonersDilemmaEnv(
    num_agents=8,
    pd_horizon=32,
    use_opening_signal=False,
)

obs, infos = env.reset(seed=7)
actions = [(0, 1)] * env.num_agents  # (partner_rel, pd)
obs, rewards, terminations, truncations, infos = env.step(actions)  # matching step: rewards are all zero
```

## Notes

- `PopulationPrisonersDilemmaEnv` と `PrisonersDilemmaEnv` は、終了後に `step()` が呼ばれると自動 `reset()` します。
- `PairedPopulationPrisonersDilemmaEnv` は、終了後に `step()` すると `RuntimeError` です。次エピソードは明示的に `reset()` してください。
- 詳細仕様は `docs/environment_api.md` を参照してください。
