# PrisonersDilemma Environment

Gymnasiumベースの 2人 repeated Prisoner's Dilemma 環境です。

current public API:
- `PairwisePrisonersDilemmaCore`
- `PrisonersDilemmaEnv`（2-agent）

`tankyu2026` の N 人集団化・selection/game phase 追加は、この package の外で
`romanohu/envs/wrappers/population_match_phase.py` が担当します。

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

## Notes

- `PrisonersDilemmaEnv` は、終了後に `step()` が呼ばれると自動 `reset()` します。
- N 人集団化が必要なら、`tankyu2026/romanohu` 側の wrapper を使ってください。
- 詳細仕様は `docs/environment_api.md` を参照してください。
