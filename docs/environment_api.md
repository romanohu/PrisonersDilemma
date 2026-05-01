# Environment API and Transition Semantics

current public API:

- `PairwisePrisonersDilemmaCore`
- `PrisonersDilemmaEnv`（2-agent）

現在の `tankyu2026` は、2人 base game `PrisonersDilemmaEnv` を `romanohu` 側 wrapper から呼び出します。

## 1. Shared Conventions

- PD action code:
  - `0`: Cooperate (`C`)
  - `1`: Defect (`D`)
- Gymnasium I/O:
  - `reset(...) -> (observations, infos)`
  - `step(actions) -> (observations, rewards, terminations, truncations, infos)`

## 2. Reward Core

### `PairwisePrisonersDilemmaCore`

```python
PairwisePrisonersDilemmaCore(
    payoff_matrix=((3.0, 0.0), (4.0, 1.0)),
)
```

- `payoff_matrix` は shape `(2, 2)` 必須。
- directed interaction `i -> j` の報酬:
  - selector reward: `payoff[a_i, a_j]`
  - selected reward: `payoff[a_j, a_i]`

`compute_round_rewards(actions, partners)`:
- `actions`: shape `(N,)`, 各要素 `0/1`
- `partners`: shape `(N,)`, `partners[i] != i`
- returns:
  - `rewards`: shape `(N,)`, `float32`
  - `interaction_counts`: shape `(N,)`, `int32`

## 3. `PrisonersDilemmaEnv` (2-agent)

### Constructor

```python
PrisonersDilemmaEnv(
    num_agents=2,
    max_steps=150,
    payoff_matrix=((3.0, 0.0), (4.0, 1.0)),
    history_h=1,
    seed=0,
)
```

- `num_agents` は `2` 固定（それ以外は `ValueError`）。
- action space: `Discrete(2)`（`0:C`, `1:D`）
- observation space: `Dict({"obs": Box(shape=(2*history_h,), dtype=float32)})`
  - 相手の過去行動履歴を one-hot 化
  - 未観測 (`-1`) はゼロ埋め

### Step Semantics

- 1 step = 1回の2人PD対戦。
- `max_steps` 到達で `terminated=True` / `truncated=True`（全agent同値）。
- 終了後に `step(...)` が呼ばれた場合:
  - 自動 `reset()` して
  - `rewards=0`, `terminations=False`, `truncations=False` を返す。

### Infos

各 `infos[i]` に主に以下を含む:
- `true_objective`
- `played_partner`
- `selected_partner`
- `interaction_count`
- `episode_extra_stats.*`（協調/裏切り回数・比率など）

N 人集団化、selection/game phase 追加、`phase_id`/`action_mask` を含む学習用観測の構築は
`tankyu2026/romanohu` 側の `PopulationMatchPhaseEnv` が担当します。
