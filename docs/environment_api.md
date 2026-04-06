# Environment API and Transition Semantics

This document describes the current API for both environments:

- `PopulationPrisonersDilemmaEnv`
- `PrisonersDilemmaEnv`

## Shared Design

The implementation is split into two layers:

1. Reward core: `PairwisePrisonersDilemmaCore`
2. Gymnasium wrappers: population / fixed-2-agent environments

Shared conventions:

- Action space: `Discrete(2)`
  - `0`: Cooperate (`C`)
  - `1`: Defect (`D`)
- Observation space:
  - `Dict({"obs": Box(shape=(2 * history_h,), dtype=float32)})`
  - Encodes partner action history as one-hot features.
- `reset(...)` returns `(observations, infos)`
- `step(actions)` returns:
  - `(observations, rewards, terminations, truncations, infos)`
- After termination, calling `step(...)` auto-resets the env and returns:
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

Round reward computation:

- For directed interaction `i -> j`:
  - selector reward gets `payoff[a_i, a_j]`
  - selected reward gets `payoff[a_j, a_i]`
- In population mode, each agent can be selected by multiple agents in the same round.
  - Those selected-side rewards are accumulated.

## `PopulationPrisonersDilemmaEnv`

### Constructor

```python
PopulationPrisonersDilemmaEnv(
    num_agents: int = 8,
    max_steps: int = 150,
    payoff_matrix=((3.0, 0.0), (5.0, 1.0)),
    history_h: int = 1,
    seed: int = 0,
    partner_scheduler: str = "random_with_replacement",
)
```

### Parameters

- `num_agents`: must be `>= 2`
- `max_steps`: must be `> 0`
- `history_h`: must be `> 0`
- `payoff_matrix`: 2x2 matrix
- `seed`: RNG seed
- `partner_scheduler`: one of
  - `"random_with_replacement"`
  - `"random_with_replacement_each_step"`

### Partner Assignment

Partners are represented by an internal directed assignment vector where each
agent `i` plays against exactly one partner `partners[i]` (`partners[i] != i`).

`reset(seed=None, options=None)`:

- If `options["partners"]` is provided, that array is used after validation.
- Otherwise, partners are sampled according to `partner_scheduler`.

`options["partners"]` constraints:

- shape must be `(num_agents,)`
- each id must be in `[0, num_agents)`
- self partnering is forbidden (`partners[i] != i`)

Sampling rule for `"random_with_replacement"`:

- each agent independently samples one partner from all other agents
- the same partner can be sampled by multiple agents

`"random_with_replacement_each_step"` uses the same sampling rule, but it
resamples partners every non-terminal `step`.

`set_partners(partners)`:

- validates with the same constraints as `options["partners"]`
- updates partner assignment immediately (useful for external policy-mapping control)

### `step(actions)`

Input:

- iterable of length `num_agents`
- each action in `{0, 1}`

State transition:

1. Apply directed interactions `i -> partners[i]` for all `i`.
2. Compute rewards via `PairwisePrisonersDilemmaCore.compute_round_rewards`.
3. Update:
   - episode step count
   - per-agent latest action and action history
   - per-agent cumulative return
4. Set terminal flags when `step >= max_steps`.
5. If scheduler is `"random_with_replacement_each_step"` and episode is not
   terminated, sample partners for the next step.

Note:

- `infos[i]["selected_partner"]` / `infos[i]["played_partner"]` always refer to
  the partner used for reward computation in the current step.
- `observations` are built from the environment's current partner assignment
  after transition updates (which may already be the next-step partner when
  using `"random_with_replacement_each_step"`).

Per-agent `infos[i]` includes:

- `true_objective`: cumulative return (`np.float32`)
- `played_partner`
- `selected_partner` (compatibility field; same value)
- `interaction_count`
  - `1 + (#times agent i was selected by others in this round)`
- `episode_extra_stats.last_action`
- `episode_extra_stats.partner_last_action`

## `PrisonersDilemmaEnv` (fixed 2-agent)

### Constructor

```python
PrisonersDilemmaEnv(
    num_agents: int = 2,
    max_steps: int = 150,
    payoff_matrix=((3.0, 0.0), (5.0, 1.0)),
    history_h: int = 1,
    seed: int = 0,
)
```

### Parameters

- `num_agents`: must be exactly `2`
- `max_steps`: must be `> 0`
- `history_h`: must be `> 0`
- `payoff_matrix`: 2x2 matrix
- `seed`: RNG seed

### Partnering and Step Semantics

- Pairing is fixed: `0 <-> 1`.
- One undirected PD game is played each step.
- Rewards are:
  - agent 0: `payoff[a0, a1]`
  - agent 1: `payoff[a1, a0]`
- `interaction_count` is always `1` for both agents.

Per-agent `infos[i]` includes the same keys as population env:

- `true_objective`
- `played_partner`
- `selected_partner`
- `interaction_count`
- `episode_extra_stats.last_action`
- `episode_extra_stats.partner_last_action`

## Render

Both environments implement `render()` and return an RGB `np.ndarray` frame summarizing:

- latest action color per agent
- episode progress bar
