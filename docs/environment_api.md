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
)
```

### Parameters

- `num_agents`: number of agents. Must be `>= 2`.
- `max_steps`: episode horizon. Must be `> 0`.
- `payoff_matrix`: `2 x 2` payoff matrix interpreted as standard PD `(R, S, T, P)`:
  - `R = payoff_matrix[0][0]` (C, C)
  - `S = payoff_matrix[0][1]` (C, D)
  - `T = payoff_matrix[1][0]` (D, C)
  - `P = payoff_matrix[1][1]` (D, D)
- `scripted_opponents`: optional scripted policy assignment per agent index.
- `scripted_seed`: RNG seed used by scripted random policies.
- `interaction_mode`: interaction rule.
  - `all_pairs_average`: all unordered pairs interact; per-agent reward is averaged by `(num_agents - 1)`.
  - `random_partner_with_replacement`: each agent samples one partner `j != i` per round with replacement.
- `reward_aggregation`: reduction for the selected interaction rule.
  - `sum` or `average`
  - if omitted: defaults to `average` for `all_pairs_average`, `sum` for `random_partner_with_replacement`

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
- Example: `[None, "titfortat"]` means agent 0 is externally controlled, agent 1 is scripted TitForTat.

## Observation and Action Spaces

- `action_space = spaces.Discrete(2)`
  - `0`: Cooperate (C)
  - `1`: Defect (D)

- `observation_space = spaces.Dict({"obs": Box(shape=(6,), dtype=float32)})`

Per-agent observation vector layout (`obs`, length 6):

1. `obs[0:3]`: one-hot of own previous action state
   - `[1, 0, 0]`: no previous action (episode start)
   - `[0, 1, 0]`: previous action was C
   - `[0, 0, 1]`: previous action was D
2. `obs[3]`: ratio of opponents that cooperated at previous step
3. `obs[4]`: ratio of opponents that defected at previous step
4. `obs[5]`: normalized episode progress (`current_step / max_steps`)

## `reset(seed=None, options=None)`

### State Transition

`reset` performs:

1. Re-seed scripted RNG if `seed` is provided.
2. Set internal step counter to `0`.
3. Clear previous actions and episode returns.
4. Mark environment active (`is_terminated = False`).
5. Reset scripted-policy internal state (for example, `grudger` memory).

### Return Value

`reset` returns `(observations, infos)`:

- `observations`: `list[dict]` of length `num_agents`
  - each item: `{ "obs": np.ndarray(shape=(6,), dtype=np.float32) }`
- `infos`: `list[dict]` of length `num_agents`
  - includes `scripted_strategy` for scripted agents
  - empty dict for non-scripted agents

## `step(actions)`

### Input

- `actions`: iterable of length `num_agents`
- each action must be `0` or `1`

### Validation and Overrides

1. Validate number of actions and action values.
2. If scripted opponents are configured, scripted indices override external actions.

### Reward Calculation

Reward logic depends on `interaction_mode`:

- `all_pairs_average`
  1. Enumerate each unordered pair `(i, j)`.
  2. Score by `Game.score((ai, aj))`.
  3. Accumulate rewards for each agent.
  4. Divide each agent total by `(num_agents - 1)`.

- `random_partner_with_replacement`
  1. Each agent `i` samples one partner `j != i` uniformly at random (with replacement).
  2. For each directed selection `(i, j_i)`, score one PD interaction and accumulate rewards for both `i` and `j_i`.
  3. Final per-agent reward is:
     - raw sum if `reward_aggregation=sum`
     - per-agent average over interaction count if `reward_aggregation=average`

### State Update

After rewards are computed:

1. Increment internal step counter.
2. Store encoded previous actions.
3. Accumulate episode returns.
4. Set `terminated = (step >= max_steps)`.
5. Build next observations.

### Return Value

`step` returns `(observations, rewards, terminations, truncations, infos)`:

- `observations`: same structure as `reset`
- `rewards`: `list[np.float32]`, length `num_agents`
- `terminations`: `list[bool]`, length `num_agents`, all equal to `terminated`
- `truncations`: `list[bool]`, length `num_agents`, currently mirrors `terminations`
- `infos`: `list[dict]`, length `num_agents`
  - each includes `true_objective` (cumulative return)
  - in `random_partner_with_replacement`, each also includes:
    - `selected_partner`
    - `interaction_count`
  - scripted agents also include:
    - `scripted_strategy`
    - `scripted_action`

## Post-Termination Behavior

If `step` is called when `is_terminated` is already `True`, the environment auto-resets and returns:

- fresh reset observations and infos
- zero rewards for all agents
- `terminations = [False, ..., False]`
- `truncations = [False, ..., False]`

## Render

`render()` returns an `np.ndarray` RGB frame (`uint8`) showing:

- each agent's previous action color block
- a progress bar indicating episode completion
