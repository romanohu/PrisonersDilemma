# Environment API and Transition Semantics

This document describes the current API of `PrisonersDilemmaEnv`.

## Design

The implementation is split into two layers:

1. Pairwise PD reward core (`PairwisePrisonersDilemmaCore`)
2. Gymnasium environment wrapper (`PrisonersDilemmaEnv`)

`PrisonersDilemmaEnv` is intentionally minimal:
- exactly 2 agents
- fixed pairing `0 <-> 1`
- one undirected PD game per step

## Constructor

```python
PrisonersDilemmaEnv(
    num_agents: int = 2,
    max_steps: int = 150,
    payoff_matrix: Sequence[Sequence[float]] = ((3.0, 0.0), (5.0, 1.0)),
    history_h: int = 1,
    seed: int = 0,
)
```

### Parameters

- `num_agents`: must be `2`
- `max_steps`: episode horizon (`> 0`)
- `payoff_matrix`: 2x2 PD payoff matrix
- `history_h`: observation history window length (`> 0`)
- `seed`: RNG seed (reserved for reproducibility hooks)

## Observation and Action Spaces

- `action_space = spaces.Discrete(2)`
  - `0`: Cooperate (`C`)
  - `1`: Defect (`D`)

- `observation_space = spaces.Dict({"obs": Box(shape=(2 * history_h,), dtype=float32)})`

For each lag `k`:

- `obs[2*k]`: partner cooperated at lag `k`
- `obs[2*k+1]`: partner defected at lag `k`

`k=0` is most recent.

## Step Semantics

One environment step does:

1. Uses fixed partners (`0 <-> 1`)
2. Applies one PD game with actions `(a0, a1)`
3. Assigns rewards:
   - agent 0: `payoff[a0, a1]`
   - agent 1: `payoff[a1, a0]`
4. Updates per-agent cumulative returns and history

## `reset(seed=None, options=None)`

State transition:

1. optional reseed
2. clear episode state/history

Return:

- `observations`: list of `{"obs": np.ndarray}` length `num_agents`
- `infos`: list of dicts containing `selected_partner`

## `step(actions)`

Input:

- iterable length `num_agents`
- each action in `{0, 1}`

Return:

`(observations, rewards, terminations, truncations, infos)`

- `observations`: next-round partner-history observations
- `rewards`: list of `np.float32`
- `terminations`: all-true when `step >= max_steps`
- `truncations`: mirrors `terminations`
- `infos` per agent:
  - `true_objective`
  - `played_partner`
  - `selected_partner` (same as `played_partner`, compatibility field)
  - `interaction_count`
  - `episode_extra_stats.last_action`
  - `episode_extra_stats.partner_last_action`

## Post-Termination Behavior

If `step` is called after termination, environment auto-resets and returns zero rewards with non-terminal flags.

## Render

`render()` returns an RGB `np.ndarray` frame showing per-agent latest action and episode progress.
