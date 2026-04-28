# N-Agent Pair PD Design Transcript (2026-04-28)

This file records the design discussion history for the new N-agent pair PD environment.

Note:
- This document is a historical transcript.
- For current implementation/API behavior, use `docs/environment_api.md` and `docs/train_construct_method.md` as the source of truth.

## User request and scope

**User:** `$grill-me` for a new MARL environment for N-agent pair PD with two phases:
- Matching phase
- PD phase

Key requirements provided by the user:
- Keep each agent action as combined heads:
  - Matching head: choose one partner from `N-1`
  - PD head: choose `C/D`
- In matching phase, use only matching-related action/head.
- In PD phase, use only PD-related action/head.
- Matching uses environment-held features, starting with:
  - `moving_avg_own_reward`
  - `moving_avg_partner_reward_when_j_is_present`
- PD should be repeated for fixed steps (default 32).
- During a pair match, non-participants should not affect learning.
- Episode-level aggregation for matching features is desired.

## Early clarification from user

**User:** Existing env shape is not a hard constraint because this is a separate file/new env.  
`core.py` should be reusable.  
Target training stack is primarily Sample Factory.

**Assistant:** Reframed the highest-impact question around Sample Factory compatibility and inactive-agent handling.

## Agreement on learning-control direction

**User:** Do not embed Sample Factory-specific behavior into the environment.  
Use wrappers in importing projects.  
But keep the environment compatible with Sample Factory usage.

**Assistant and User alignment:**
- Environment should be framework-agnostic.
- Wrapper can map env metadata to Sample Factory behavior.

## Selection observation design

**Assistant question:** Candidate-wise features vs global summary.  
**User:** Choose candidate-wise.

**Resolved:**
- Selection observation is candidate-aligned fixed shape per agent.
- Initial candidate feature vector is exactly 2 scalars:
  - `moving_avg_own_reward[j]`
  - `moving_avg_partner_reward_when_j_is_present[j]`
- Initial implementation should be per-agent-slot (not policy-id keyed inside env).

## EMA semantics

**User and Assistant resolved:**
- `moving_avg_partner_reward_when_j_is_present[j]` should only use matches where `j` is actually involved (not unrelated episode-wide averages).
- `moving_avg_own_reward[j]` uses episode total reward (sum, not per-match average).
- EMA update rule is shared/fixed:
  - `ema <- (1 - alpha) * ema + alpha * episode_value`
- EMA values persist across episodes.
- `reset()` must not clear EMA state.
- A separate `reset_population_stats()` API should clear population-level EMA stats.
- Cold-start should use scalar priors, not extra feature dims:
  - `own_reward_prior`
  - `partner_reward_prior`

## Phase control and metadata

**User concern:** Avoid ambiguity when switching models between selection and PD phase.

**Resolved:**
- Keep phase/control metadata out of core `obs`, and expose via `infos[i]` for wrappers.
- Required control metadata in `infos[i]`:
  - `phase`
  - `can_act`
  - `new_match`
  - `active_opponent_id`
  - `selected_partner`
  - `true_objective`

**Additional resolved details:**
- `selected_partner` and `active_opponent_id` in `infos` are absolute agent IDs.
- `active_opponent_id = -1` when not active.
- `true_objective` is episode cumulative reward (`np.float32` style).

## Non-participant handling

**User request:** Environment must strictly enforce that non-participants do not influence transitions or learning-relevant state.

**Resolved hard rule:**
- `can_act=False` agents are fully masked by env logic:
  - Their actions are ignored for transition/reward logic.
  - Their per-step internal learning-relevant updates are blocked.

## PD observation representation

**User proposal:** Use `[0,0]` for initial/unknown, and action one-hot encoding.

**Resolved encoding:**
- Follow `core.py` action semantics:
  - `C` (action 0) -> `[1,0]`
  - `D` (action 1) -> `[0,1]`
  - Unknown/initial -> `[0,0]`

**User decision:** No extra explicit sequence-position feature for PD (`remaining_steps` etc. not required).

## Opening signal extension

**User request:** Also support version without opening signal.

**Resolved:**
- Optional `opening_signal` feature exists, but can be disabled.
- `opening_signal` is separate from PD action and used only to initialize first PD observation in each local match.
- It is not counted in reward/statistics/history/EMA.
- Per episode, one `opening_signal` per agent (shared across that agent’s local matches).
- `use_opening_signal=False` version must be supported.

## Matching frequency and episode structure

**Resolved:**
- Exactly one matching phase at episode start.
- Then process directed pair matches for the episode.

## Directed matches and compression by slot packing

**User proposal:** Do not force one directed match per gym step if non-overlapping pairs can run in parallel.

**Resolved:**
- Build directed matches `i -> partner[i]` for all agents (`N` directed matches).
- `i -> j` and `j -> i` remain independent matches.
- Pack matches via deterministic greedy packing into groups with no shared agents per group.
- Each group is called a processing bucket/slot (non-overlapping match set).
- Execute slot-by-slot; for each slot, run full `pd_horizon` steps before moving to next slot.
- This compresses episode length and increases active-agent density.

## Independence of reciprocal matches

**Resolved:**
- Reciprocal directed matches are fully independent local PD games.
- No PD internal state sharing across those matches.
- `new_match` marks local boundary transitions.

## Action space decisions

**Resolved:**
- Partner choice is relative ID over `N-1` candidates.
- If `use_opening_signal=True`:
  - action heads are `(partner_choice, opening_signal, pd_action)`.
- If `use_opening_signal=False`:
  - action heads are `(partner_choice, pd_action)`.

## Observation space decisions

**Resolved:**
- Use `Dict` observation with phase-specific payload fields:
  - `selection_obs`: shape `(N-1, 2)` (2D, not flattened)
  - `pd_obs`: shape `(2,)`
- Unused phase branch is zero-filled.

## Episode/step semantics

**Resolved:**
- Matching step has zero rewards for all agents.
- PD rewards only occur during PD steps.
- `pd_horizon` default is 32, configurable.
- `pd_horizon` counts reward-producing PD action steps.
- End-of-episode requires explicit `reset()`.
- No auto-reset on `step()` after done.

## Constructor/API set agreed

Initial constructor arguments agreed:
- `num_agents`
- `pd_horizon=32`
- `ema_alpha`
- `use_opening_signal`
- `own_reward_prior`
- `partner_reward_prior`
- `payoff_matrix=((3,0),(4,1))` (aligned to `core.py` default)
- `seed`

Additional API agreed:
- `reset_population_stats()`

## Notes on model wiring (outside env)

User indicated external stack strategy:
- Separate model families per phase (selection model and PD model).
- Policy assignments are agent-aligned externally.
- Wrapper is responsible for routing/masking using `infos`.

## Final status before implementation

User requested:
1. Save this discussion history into one markdown file.
2. Then start implementation.
