# Environment API and Transition Semantics

This document describes the current API for both environments:

- `PopulationPrisonersDilemmaEnv`
- `PrisonersDilemmaEnv`

## Shared Design

The implementation is split into two layers:

1. Reward core: `PairwisePrisonersDilemmaCore`
2. Gymnasium wrappers: population / fixed-2-agent environments

Shared conventions:

- Action values use binary PD coding:
  - `0`: Cooperate (`C`)
  - `1`: Defect (`D`)
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

### Observation and Action Spaces

- `observation_space = Dict({"obs": Box(shape=(2 * history_h,), dtype=float32)})`
  - At each environment step, observation is the one-hot history of the
    *current interaction opponent* for that agent.
  - In population mode, all agents are active on non-terminal steps.

- `action_space = Discrete(2)` per agent
  - `0`: Cooperate (`C`)
  - `1`: Defect (`D`)

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
resamples partners at every non-terminal round boundary.

`set_partners(partners)`:

- validates with the same constraints as `options["partners"]`
- stores assignment as pending override for the next round (next env step)

### Partner Assignment Timing (with external controller)

When this environment is controlled by an external partner-selection module:

1. `set_partners(...)` does not immediately replace the current step's assignment.
2. It stores the assignment as a pending value.
3. The pending value is consumed after current-step round processing and becomes the partner assignment of the next step.
4. If `set_partners(...)` is called multiple times before the next step starts, the latest call overwrites earlier pending values.

Operationally, this means one step runs with one fixed assignment, and external updates take effect on the following step.

### `step(actions)`

Input:

- iterable shaped `(num_agents,)`
- each action value in `{0, 1}`

State transition:

1. One environment step processes one full round of directed interactions.
2. For every selector `i`, run interaction `i -> j` where `j=partners[i]`.
3. Each agent outputs one C/D action in the step; that same action is reused for all incoming interactions in that step.
4. Apply directed PD payoff for all selectors:
   - selector reward `+= payoff[a_i, a_j]`
   - selected reward `+= payoff[a_j, a_i]`
5. Update per-agent latest selector action and history (one action per agent per round/step).
6. Set terminal flags when completed rounds reach `max_steps`.
7. If scheduler is `"random_with_replacement_each_step"` and episode is not
   terminated, sample partners for the next round (next step).

Note:

- `infos[i]["selected_partner"]` / `infos[i]["played_partner"]` always refer to
  the current step round's selected partner for agent `i`.
- `episode_extra_stats.last_action` is the selector action used in agent `i`'s
  own directed interaction `i -> partners[i]` (for the round summary).
- If agent `j` is selected by multiple selectors in one round, selected-side
  rewards from all incoming interactions are accumulated in the same step reward of `j`.
- In that case, `j` still uses one action for the whole step.
- Action history update stores only one action per agent per step:
  the selector-side action from that agent's own directed interaction in the round
  `i -> partners[i]`.
- `infos[i]["is_active"]` is `True` for all agents on non-terminal steps.

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
