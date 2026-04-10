# Environment API and Transition Semantics

This document describes the API for:

- `PopulationPrisonersDilemmaEnv`

## Shared Core

The implementation is split into two layers:

1. Reward core: `PairwisePrisonersDilemmaCore`
2. Gymnasium wrapper: `PopulationPrisonersDilemmaEnv`

Shared conventions:

- PD action values:
  - `0`: Cooperate (`C`)
  - `1`: Defect (`D`)
- `reset(...)` returns `(observations, infos)`
- `step(actions)` returns:
  - `(observations, rewards, terminations, truncations, infos)`
- After termination, calling `step(...)` auto-resets and returns:
  - zero rewards
  - all `False` for `terminations` and `truncations`

## Reward Core (`PairwisePrisonersDilemmaCore`)

Constructor:

```python
PairwisePrisonersDilemmaCore(
    payoff_matrix=((3.0, 0.0), (5.0, 1.0)),
)
```

- `payoff_matrix` must be shape `(2, 2)`.

For one directed interaction `i -> j`:

- selector reward: `payoff[a_i, a_j]`
- selected reward: `payoff[a_j, a_i]`

## `PopulationPrisonersDilemmaEnv`

### Constructor

```python
PopulationPrisonersDilemmaEnv(
    num_agents: int = 8,
    max_steps: int = 150,
    payoff_matrix=((3.0, 0.0), (5.0, 1.0)),
    history_h: int = 1,
    seed: int = 0,
    partner_scheduler: str = "from_actions",
)
```

### Parameters

- `num_agents`: must be `>= 2`
- `max_steps`: number of rounds per episode (`> 0`)
- `history_h`: observation history length (`> 0`)
- `payoff_matrix`: 2x2 matrix
- `seed`: RNG seed
- `partner_scheduler`: partner matching mode
  - `from_actions`: use partner head choices at each round
  - `random`: random partner sampling at each round boundary

Compatibility aliases `random_with_replacement` and `random_with_replacement_each_step` are normalized to `random`.

### Observation and Action Spaces

- `observation_space = Dict({"obs": Box(shape=((num_agents - 1) * 2 * history_h + 1,), dtype=float32)})`
  - base part: concatenated one-hot PD histories of all other agents
  - last scalar: phase flag (`0.0=round boundary`, `1.0=dilemma`)
  - unknown history (`-1`) is encoded as all zeros in that slot
  - in dilemma phase, only currently interacting pair slots are populated; non-participants are zeros

- `action_space = Tuple(Discrete(num_agents - 1), Discrete(2))` per agent
  - head 0 (`partner`): relative partner id among other agents
  - head 1 (`pd`): PD action (`0:C`, `1:D`)

Relative partner decoding for agent `i`:

- if `partner_rel < i` then `partner_abs = partner_rel`
- else `partner_abs = partner_rel + 1`

### Partner Overrides

`set_partners(partners)`:

- validates:
  - shape `(num_agents,)`
  - each id in `[0, num_agents)`
  - no self-partnering
- applies at the next round boundary (phase `0.0`), then clears

`reset(options={"partners": ...})` sets the same pending override.

### Round and Step Semantics

One round has exactly `num_agents` env steps:

1. **Round-start step** (phase is `selection` before stepping)
   - decode all actions
   - resolve partner map (override / matching mode / partner head)
   - store partner map for this round
   - execute selector `0 -> partner[0]` interaction in the same step
   - transition to dilemma phase

2. **Remaining dilemma steps** (`num_agents - 1` env steps)
   - each step executes one directed interaction:
     - selector index advances as `1,2,...,num_agents-1`
     - selected index is `partner[selector]`
   - reward uses selector PD action and selected PD action from that step
   - after last selector, history is committed and round count is incremented
   - transition back to round-boundary phase (`selection`)

### `infos[i]` Fields

Per-agent `infos[i]` contains:

- `true_objective`: cumulative episode return (`np.float32`)
- `played_partner`: currently interacted partner id or `-1` (if not interacting this step)
- `selected_partner`: partner chosen for this round
- `interaction_count`: `1 + (#times agent i is selected by others in the round)`
- `episode_extra_stats.last_action`
- `episode_extra_stats.partner_last_action`
- `episode_extra_stats.cooperate_count`
- `episode_extra_stats.defect_count`
- `episode_extra_stats.cooperate_ratio`
- `episode_extra_stats.defect_ratio`
- `episode_extra_stats.selected_count_episode`
- `episode_extra_stats.env_total_reward`
- `is_active`
  - `True` during non-terminal steps
  - `False` on terminal transition

## Render

`render()` returns an RGB `np.ndarray` frame summarizing:

- latest committed selector-side action color per agent
- episode progress bar
