# Environment API and Transition Semantics

This document describes exactly how `PrisonersDilemmaEnv` transitions state and what each API call returns.

## Constructor

```python
PrisonersDilemmaEnv(
    num_agents: int = 2,
    max_steps: int = 150,
    payoff_matrix: Sequence[Sequence[float]] = ((3.0, 0.0), (5.0, 1.0)),
    scripted_opponents: Sequence[str | None] | None = None,
    scripted_seed: int = 0,
    interaction_mode: str = "all_pairs_average",
    reward_aggregation: str | None = None,
    history_h: int = 1,
)
```

### Parameters

- `num_agents`: number of agents. Must be `>= 2`.
- `max_steps`: episode horizon. Must be `> 0`.
- `payoff_matrix`: `2 x 2` payoff matrix interpreted as standard PD `(R, S, T, P)`.
- `scripted_opponents`: optional scripted policy assignment per agent index.
- `scripted_seed`: RNG seed used by scripted random policies.
- `interaction_mode`: interaction rule.
  - `all_pairs_average`: all unordered pairs interact; per-agent reward is averaged by `(num_agents - 1)`.
  - `random_partner_with_replacement`: each round has a random selection stage, then a dilemma stage.
- `reward_aggregation`: reduction for the selected interaction rule (`sum` or `average`).
  - default is `average` for `all_pairs_average`, `sum` for `random_partner_with_replacement`.
- `history_h`: observation history window length. Must be `> 0`.

### Scripted Opponents

Supported scripted policy names:

- `cooperator`
- `defector`
- `titfortat`
- `grudger`
- `random`

Notes:

- Scripted opponents are currently supported only when `num_agents == 2`.
- Use `None` for agents that should stay externally controlled.

## Observation and Action Spaces

- `action_space = spaces.Discrete(2)`
  - `0`: Cooperate (C)
  - `1`: Defect (D)

- `observation_space = spaces.Dict({"obs": Box(shape=(2 * history_h,), dtype=float32)})`

Per-agent observation vector layout (`obs`, length `2 * history_h`):

For each lag `k` from `0` to `history_h - 1`:

- `obs[2*k]`: cooperation feature at lag `k`
- `obs[2*k+1]`: defection feature at lag `k`

`k=0` is the most recent interaction history, `k=1` is one step older, and so on.

Feature semantics by mode:

- `all_pairs_average`: each lag pair stores the ratio of opponents that cooperated/defected at that lag.
- `random_partner_with_replacement`: each lag pair stores one selected partner's action history as one-hot (`[1,0]` for C, `[0,1]` for D, `[0,0]` when history is empty).

## `reset(seed=None, options=None)`

State transition:

1. Optional re-seed.
2. Clear step counter, previous actions, action-history buffer, episode returns.
3. Reset scripted policy internal state.
4. In `random_partner_with_replacement`, sample one selected partner per agent for the first round.

Return:

- `observations`: list of `{"obs": np.ndarray(shape=(2 * history_h,), dtype=np.float32)}`
- `infos`: list of dicts
  - includes `selected_partner` in random-matching mode
  - includes `scripted_strategy` for scripted agents

## `step(actions)`

Input:

- `actions`: iterable of length `num_agents`
- each action must be `0` or `1`

Validation and overrides:

1. validate action length/values
2. apply scripted overrides where configured

Reward calculation:

- `all_pairs_average`
  1. evaluate all unordered pairs
  2. accumulate pair payoffs
  3. divide by `(num_agents - 1)`

- `random_partner_with_replacement`
  1. use pre-sampled `selected_partner` for this round (`played_partner`)
  2. run one directed dilemma interaction per selecting agent
  3. aggregate with `sum` or `average`
  4. sample `selected_partner` for the next round

Return:

`(observations, rewards, terminations, truncations, infos)`

- `observations`: next-round observations
- `rewards`: `list[np.float32]`
- `terminations`: list of bools
- `truncations`: list of bools (mirrors `terminations`)
- `infos`: per-agent dict
  - `true_objective`
  - in random-matching mode:
    - `played_partner`
    - `selected_partner` (next round)
    - `interaction_count`
  - scripted extras when applicable:
    - `scripted_strategy`
    - `scripted_action`

## Post-Termination Behavior

If `step` is called after termination, environment auto-resets and returns zero rewards with non-terminal flags.

## Render

`render()` returns an RGB `np.ndarray` frame showing per-agent previous actions and episode progress.
