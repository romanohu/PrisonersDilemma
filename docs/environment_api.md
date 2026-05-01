# Environment API and Transition Semantics

このドキュメントは、現在リポジトリで公開している以下の環境APIをまとめたものです。

- `PairwisePrisonersDilemmaCore`
- `PrisonersDilemmaEnv`（2-agent）
- `PopulationPrisonersDilemmaEnv`
- `PairedPopulationPrisonersDilemmaEnv`

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

## 4. `PopulationPrisonersDilemmaEnv`

### Constructor

```python
PopulationPrisonersDilemmaEnv(
    num_agents=8,
    max_steps=150,
    payoff_matrix=((3.0, 0.0), (5.0, 1.0)),
    history_h=1,
    seed=0,
    partner_scheduler="from_actions",
)
```

- `partner_scheduler`:
  - `"from_actions"`: partner head を使用
  - `"random"`: ランダムマッチング
  - 互換エイリアス:
    - `"random_with_replacement"`
    - `"random_with_replacement_each_step"`

### Observation / Action

- action space: `Tuple(Discrete(num_agents-1), Discrete(2))`
  - head0: 相対 partner id
  - head1: PD action
- observation space:
  - `Dict({"obs": Box(shape=((num_agents-1)*2*history_h + 1,), dtype=float32)})`
  - 末尾1要素は phase flag:
    - `0.0`: selection
    - `1.0`: dilemma
  - dilemma中は「現在対戦中の2agentだけ」相手履歴スロットが埋まり、他はゼロ。

### Round / Step Semantics

1 round は `num_agents` step:
1. round-start step:
  - partner map 決定
  - selector `0 -> partner[0]` を同stepで実行
2. 残り `num_agents-1` step:
  - selector `1..num_agents-1` を順に実行

`max_steps` は「round数」の上限。

### Post-Termination Behavior

- 終了後に `step(...)` が呼ばれた場合は自動 `reset()`（Sample Factory互換挙動）。

### Infos

`step` の `infos[i]` には以下を含む:
- `true_objective`
- `played_partner`
- `selected_partner`
- `interaction_count`
- `episode_extra_stats.*`
- `is_active`（終端stepのみ `False`）

`reset()` 直後の `infos` は preview 用の簡易フィールド中心:
- `selected_partner`（pending override があればその値）
- `played_partner=-1`
- `interaction_count=0`
- `is_active=True`

## 5. `PairedPopulationPrisonersDilemmaEnv`

### Constructor

```python
PairedPopulationPrisonersDilemmaEnv(
    num_agents=8,
    pd_horizon=32,
    ema_alpha=0.1,
    own_reward_prior=0.0,
    partner_reward_prior=0.0,
    payoff_matrix=((3.0, 0.0), (4.0, 1.0)),
    seed=0,
)
```

### Observation / Action

- observation space:
  - `selection_obs`: shape `(num_agents-1, 2)`
  - `pd_obs`: shape `(2,)`
- `pd_obs` encoding:
  - `C -> [1,0]`
  - `D -> [0,1]`
  - 未観測初期値 -> `[0,0]`

- action space:
  - `(partner_choice_rel, pd_action)`

入力形式は `tuple/list/ndarray` または `dict` の両方を受け付ける。

### Episode Semantics

1 episode:
1. matching phase を1step（全員 reward=0）
2. directed match `i -> partner[i]` を全agent分作成
3. greedy packing（同一slotでagent重複なし）
4. 各slotを `pd_horizon` step 実行
5. 全slot終了で episode 終了

補足:
- `i -> j` と `j -> i` は独立な別試合として扱う。
- 各ローカル試合の1step目 `pd_obs` は常に `[0,0]` で始まる。

### Phase Metadata (`infos[i]`)

- `phase`: `0=matching`, `1=pd`
- `can_act`
- `new_match`
- `active_opponent_id`（非参加は `-1`）
- `selected_partner`（絶対ID）
- `true_objective`

### Non-Participant Rule

- PD phaseで `can_act=False` のagentは遷移・報酬計算に使われない（actionは無視）。
- ただし `step()` 呼び出し上、action配列自体は全agent分必要。

### Post-Termination Behavior

- 終了後に `step(...)` を呼ぶと `RuntimeError`。
- 次エピソードへ進むには明示的な `reset()` が必要。

### EMA Stats

- `reset()` ではEMAを保持。
- `reset_population_stats()` でのみEMAを初期化。
