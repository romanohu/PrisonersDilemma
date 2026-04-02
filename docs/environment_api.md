# Environment API and Transition Semantics

This document describes the current API of `PrisonersDilemmaEnv`.

## Design

The implementation is split into three layers:

1. Pairwise PD reward core (`PairwisePrisonersDilemmaCore`)
2. Partner-selection scheduler (`InteractionScheduler`)
3. Gymnasium environment wrapper (`PrisonersDilemmaEnv`)

`PrisonersDilemmaEnv` always uses pairwise 2x2 PD rewards.
Who interacts with whom is handled by the scheduler.

## Constructor

```python
PrisonersDilemmaEnv(
    num_agents: int = 20,
    max_steps: int = 150,
    payoff_matrix: Sequence[Sequence[float]] = ((3.0, 0.0), (5.0, 1.0)),
    history_h: int = 1,
    scheduler: InteractionScheduler | None = None,
    seed: int = 0,
)
```

### Parameters

- `num_agents`: number of agents (`>= 2`)
- `max_steps`: episode horizon (`> 0`)
- `payoff_matrix`: 2x2 PD payoff matrix
- `history_h`: observation history window length (`> 0`)
- `scheduler`: external partner-selection scheduler
  - defaults to `RandomPartnerScheduler` when `None`
- `seed`: RNG seed for scheduler sampling

## Scheduler Contract

Scheduler must implement:

```python
select_partners(
    *,
    num_agents: int,
    rng: np.random.Generator,
    action_history: np.ndarray,
    step: int,
) -> np.ndarray
```

Return value requirements:

- shape `(num_agents,)`
- integer partner ids
- `partner[i] != i`

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

1. Uses already-sampled partners `partner[i]` for this round
2. Applies one directed pairwise PD interaction per agent: `(i -> partner[i])`
3. Updates per-agent cumulative returns and history
4. Samples partners for next round via scheduler

This yields `num_agents` directed interactions per step.

## `reset(seed=None, options=None)`

State transition:

1. optional reseed
2. clear episode state/history
3. sample first-round partners using scheduler

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
  - `selected_partner` (for next round)
  - `interaction_count`
  - `episode_extra_stats.last_action`

## Post-Termination Behavior

If `step` is called after termination, environment auto-resets and returns zero rewards with non-terminal flags.

## Render

`render()` returns an RGB `np.ndarray` frame showing per-agent latest action and episode progress.
