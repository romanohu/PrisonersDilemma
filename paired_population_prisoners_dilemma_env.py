from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .core import PairwisePrisonersDilemmaCore


@dataclass
class _DirectedMatchState:
    selector: int
    opponent: int
    selector_return: float = 0.0
    opponent_return: float = 0.0
    last_selector_action: int = -1
    last_opponent_action: int = -1


class PairedPopulationPrisonersDilemmaEnv(gym.Env):
    """N-agent pair-PD with one matching phase and slot-packed local PD matches.

    Episode flow:
    1) One matching step (zero reward)
    2) N directed matches i -> partner[i], packed into non-overlapping slots
    3) Each slot runs for pd_horizon steps before moving to the next slot

    Design invariants:
    - Non-participants are hard-masked by `can_act=False`.
    - `opening_signal` only initializes first PD observation; it never affects rewards.
    - EMA population stats persist across `reset()` and are cleared only by
      `reset_population_stats()`.
    """

    metadata = {"render_modes": ["rgb_array"]}

    PHASE_MATCHING = 0
    PHASE_PD = 1

    def __init__(
        self,
        *,
        num_agents: int = 8,
        pd_horizon: int = 32,
        ema_alpha: float = 0.1,
        use_opening_signal: bool = False,
        own_reward_prior: float = 0.0,
        partner_reward_prior: float = 0.0,
        payoff_matrix: Sequence[Sequence[float]] = ((3.0, 0.0), (4.0, 1.0)),
        seed: int = 0,
    ):
        self.num_agents = int(num_agents)
        self.pd_horizon = int(pd_horizon)
        self.ema_alpha = float(ema_alpha)
        self.use_opening_signal = bool(use_opening_signal)
        self.own_reward_prior = float(own_reward_prior)
        self.partner_reward_prior = float(partner_reward_prior)

        if self.num_agents < 2:
            raise ValueError(f"num_agents must be >= 2, got {num_agents}")
        if self.pd_horizon <= 0:
            raise ValueError(f"pd_horizon must be positive, got {pd_horizon}")
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must satisfy 0 < ema_alpha <= 1, got {ema_alpha}")

        self.core = PairwisePrisonersDilemmaCore(payoff_matrix=payoff_matrix)
        self._rng = np.random.default_rng(int(seed))

        if self.use_opening_signal:
            self.action_space = spaces.Tuple(
                (
                    spaces.Discrete(self.num_agents - 1),  # partner choice (relative id)
                    spaces.Discrete(2),  # opening signal (C/D)
                    spaces.Discrete(2),  # PD action (C/D)
                )
            )
        else:
            self.action_space = spaces.Tuple(
                (
                    spaces.Discrete(self.num_agents - 1),  # partner choice (relative id)
                    spaces.Discrete(2),  # PD action (C/D)
                )
            )

        self.observation_space = spaces.Dict(
            {
                "selection_obs": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(self.num_agents - 1, 2),
                    dtype=np.float32,
                ),
                "pd_obs": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

        self.is_multiagent = True
        self.is_terminated = False

        # Population-level EMA state (persists across reset()).
        self._ema_own_reward = np.full((self.num_agents,), self.own_reward_prior, dtype=np.float32)
        self._ema_partner_reward_when_present = np.full(
            (self.num_agents,),
            self.partner_reward_prior,
            dtype=np.float32,
        )

        self._all_false = [False for _ in range(self.num_agents)]
        self._all_true = [True for _ in range(self.num_agents)]

        self._clear_episode_state()

    def reset_population_stats(self) -> None:
        self._ema_own_reward[:] = np.float32(self.own_reward_prior)
        self._ema_partner_reward_when_present[:] = np.float32(self.partner_reward_prior)

    def _clear_episode_state(self) -> None:
        self._phase = self.PHASE_MATCHING
        self._episode_done = False
        self.is_terminated = False

        self._episode_returns = np.zeros((self.num_agents,), dtype=np.float32)
        self._selected_partners = np.full((self.num_agents,), -1, dtype=np.int32)
        self._opening_signals = np.zeros((self.num_agents,), dtype=np.int8)

        self._matches: list[_DirectedMatchState] = []
        self._slots: list[list[int]] = []
        self._slot_index = 0
        self._pd_step_in_slot = 0

        self._can_act = np.ones((self.num_agents,), dtype=np.bool_)
        self._new_match = np.zeros((self.num_agents,), dtype=np.bool_)
        self._active_opponents = np.full((self.num_agents,), -1, dtype=np.int32)

    @staticmethod
    def _encode_pd_action(action_code: int) -> np.ndarray:
        encoded = np.zeros((2,), dtype=np.float32)
        if action_code == 0:
            encoded[0] = 1.0
        elif action_code == 1:
            encoded[1] = 1.0
        return encoded

    def _set_phase_metadata(self, *, phase: int, new_match: bool = False) -> None:
        if phase == self.PHASE_MATCHING:
            self._can_act[:] = True
            self._new_match[:] = False
            self._active_opponents[:] = -1
            return

        # PD phase: activate only participants in the current slot.
        self._can_act[:] = False
        self._new_match[:] = False
        self._active_opponents[:] = -1
        if self._slot_index >= len(self._slots):
            return
        for match_idx in self._slots[self._slot_index]:
            match = self._matches[match_idx]
            self._can_act[match.selector] = True
            self._can_act[match.opponent] = True
            self._active_opponents[match.selector] = np.int32(match.opponent)
            self._active_opponents[match.opponent] = np.int32(match.selector)
            if new_match:
                self._new_match[match.selector] = True
                self._new_match[match.opponent] = True

    def _build_observations(self) -> list[dict[str, np.ndarray]]:
        observations = [
            {
                "selection_obs": np.zeros((self.num_agents - 1, 2), dtype=np.float32),
                "pd_obs": np.zeros((2,), dtype=np.float32),
            }
            for _ in range(self.num_agents)
        ]

        if self._phase == self.PHASE_MATCHING:
            for agent_idx in range(self.num_agents):
                row = 0
                for other_idx in range(self.num_agents):
                    if other_idx == agent_idx:
                        continue
                    observations[agent_idx]["selection_obs"][row, 0] = self._ema_own_reward[other_idx]
                    observations[agent_idx]["selection_obs"][row, 1] = self._ema_partner_reward_when_present[other_idx]
                    row += 1
            return observations

        # PD phase
        for match_idx in self._slots[self._slot_index]:
            match = self._matches[match_idx]
            selector_idx = match.selector
            opponent_idx = match.opponent

            if self._pd_step_in_slot == 0:
                if self.use_opening_signal:
                    observations[selector_idx]["pd_obs"] = self._encode_pd_action(int(self._opening_signals[opponent_idx]))
                    observations[opponent_idx]["pd_obs"] = self._encode_pd_action(int(self._opening_signals[selector_idx]))
            else:
                observations[selector_idx]["pd_obs"] = self._encode_pd_action(match.last_opponent_action)
                observations[opponent_idx]["pd_obs"] = self._encode_pd_action(match.last_selector_action)

        return observations

    def _build_infos(self) -> list[dict]:
        infos: list[dict] = []
        for agent_idx in range(self.num_agents):
            infos.append(
                {
                    "phase": int(self._phase),
                    "can_act": bool(self._can_act[agent_idx]),
                    "new_match": bool(self._new_match[agent_idx]),
                    "active_opponent_id": int(self._active_opponents[agent_idx]),
                    "selected_partner": int(self._selected_partners[agent_idx]),
                    "true_objective": np.asarray(self._episode_returns[agent_idx], dtype=np.float32),
                }
            )
        return infos

    def _step_selection_phase(self, *, partners_abs: np.ndarray, opening_signals: np.ndarray):
        self._selected_partners = partners_abs.copy()
        if self.use_opening_signal:
            self._opening_signals = opening_signals.copy()
        else:
            self._opening_signals[:] = 0

        self._matches = [
            _DirectedMatchState(selector=selector_idx, opponent=int(self._selected_partners[selector_idx]))
            for selector_idx in range(self.num_agents)
        ]

        # Greedy packing: no agent appears twice in one slot.
        slots: list[list[int]] = []
        used_agents_per_slot: list[set[int]] = []
        for match_idx, match in enumerate(self._matches):
            placed = False
            for slot_idx, used_agents in enumerate(used_agents_per_slot):
                if match.selector in used_agents or match.opponent in used_agents:
                    continue
                slots[slot_idx].append(match_idx)
                used_agents.add(match.selector)
                used_agents.add(match.opponent)
                placed = True
                break
            if not placed:
                slots.append([match_idx])
                used_agents_per_slot.append({match.selector, match.opponent})
        self._slots = slots

        self._slot_index = 0
        self._pd_step_in_slot = 0
        self._phase = self.PHASE_PD
        self._set_phase_metadata(phase=self.PHASE_PD, new_match=True)

        observations = self._build_observations()
        rewards = [np.float32(0.0) for _ in range(self.num_agents)]
        terminations = list(self._all_false)
        truncations = list(self._all_false)
        infos = self._build_infos()
        return observations, rewards, terminations, truncations, infos

    def _step_game_phase(self, *, pd_actions: np.ndarray):
        rewards_array = np.zeros((self.num_agents,), dtype=np.float32)
        for match_idx in self._slots[self._slot_index]:
            match = self._matches[match_idx]
            selector_idx = match.selector
            opponent_idx = match.opponent

            selector_action = int(pd_actions[selector_idx])
            opponent_action = int(pd_actions[opponent_idx])

            selector_reward = np.float32(self.core.payoff[selector_action, opponent_action])
            opponent_reward = np.float32(self.core.payoff[opponent_action, selector_action])

            rewards_array[selector_idx] += selector_reward
            rewards_array[opponent_idx] += opponent_reward

            match.selector_return += float(selector_reward)
            match.opponent_return += float(opponent_reward)
            match.last_selector_action = selector_action
            match.last_opponent_action = opponent_action

        self._episode_returns += rewards_array

        episode_done = False
        if self._pd_step_in_slot + 1 < self.pd_horizon:
            self._pd_step_in_slot += 1
            self._set_phase_metadata(phase=self.PHASE_PD, new_match=False)
            observations = self._build_observations()
        else:
            if self._slot_index + 1 < len(self._slots):
                self._slot_index += 1
                self._pd_step_in_slot = 0
                self._set_phase_metadata(phase=self.PHASE_PD, new_match=True)
                observations = self._build_observations()
            else:
                partner_reward_sum = np.zeros((self.num_agents,), dtype=np.float32)
                partner_reward_count = np.zeros((self.num_agents,), dtype=np.int32)
                for match in self._matches:
                    # For each agent j, track opponents' returns from matches where j participates.
                    partner_reward_sum[match.selector] += np.float32(match.opponent_return)
                    partner_reward_count[match.selector] += 1
                    partner_reward_sum[match.opponent] += np.float32(match.selector_return)
                    partner_reward_count[match.opponent] += 1

                partner_episode_values = np.full((self.num_agents,), self.partner_reward_prior, dtype=np.float32)
                valid = partner_reward_count > 0
                partner_episode_values[valid] = (
                    partner_reward_sum[valid] / partner_reward_count[valid].astype(np.float32)
                )

                alpha = np.float32(self.ema_alpha)
                one_minus_alpha = np.float32(1.0) - alpha
                self._ema_own_reward = one_minus_alpha * self._ema_own_reward + alpha * self._episode_returns
                self._ema_partner_reward_when_present = (
                    one_minus_alpha * self._ema_partner_reward_when_present + alpha * partner_episode_values
                )

                self._episode_done = True
                self.is_terminated = True
                self._phase = self.PHASE_MATCHING
                self._can_act[:] = False
                self._new_match[:] = False
                self._active_opponents[:] = -1
                observations = [
                    {
                        "selection_obs": np.zeros((self.num_agents - 1, 2), dtype=np.float32),
                        "pd_obs": np.zeros((2,), dtype=np.float32),
                    }
                    for _ in range(self.num_agents)
                ]
                episode_done = True

        rewards = [np.float32(value) for value in rewards_array]
        terminations = list(self._all_true) if episode_done else list(self._all_false)
        truncations = list(self._all_false)
        infos = self._build_infos()
        return observations, rewards, terminations, truncations, infos

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        _ = options

        self._clear_episode_state()
        self._set_phase_metadata(phase=self.PHASE_MATCHING)

        observations = self._build_observations()
        infos = self._build_infos()
        return observations, infos

    def step(self, actions: Iterable):
        if self._episode_done:
            raise RuntimeError("Episode already terminated. Call reset() before step().")

        action_list = list(actions)
        if len(action_list) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(action_list)}")

        partners_abs = np.zeros((self.num_agents,), dtype=np.int32)
        opening_signals = np.zeros((self.num_agents,), dtype=np.int8)
        pd_actions = np.zeros((self.num_agents,), dtype=np.int8)

        for agent_idx, action_entry in enumerate(action_list):
            if isinstance(action_entry, dict):
                if "partner" not in action_entry:
                    raise ValueError(f"Action dict for agent {agent_idx} must contain 'partner': {action_entry}")
                if "pd" not in action_entry:
                    raise ValueError(f"Action dict for agent {agent_idx} must contain 'pd': {action_entry}")
                partner_raw = action_entry["partner"]
                pd_raw = action_entry["pd"]
                opening_raw = action_entry.get("opening", 0)
            elif isinstance(action_entry, (tuple, list, np.ndarray)):
                if self.use_opening_signal:
                    if len(action_entry) != 3:
                        raise ValueError(
                            f"Expected 3 action heads (partner, opening, pd) for agent {agent_idx}, got {action_entry}"
                        )
                    partner_raw, opening_raw, pd_raw = action_entry
                else:
                    if len(action_entry) != 2:
                        raise ValueError(
                            f"Expected 2 action heads (partner, pd) for agent {agent_idx}, got {action_entry}"
                        )
                    partner_raw, pd_raw = action_entry
                    opening_raw = 0
            else:
                raise ValueError(
                    f"Unsupported action format for agent {agent_idx}: {type(action_entry)}. "
                    "Expected dict or tuple/list."
                )

            partner_rel = int(partner_raw)
            if partner_rel < 0 or partner_rel >= self.num_agents - 1:
                raise ValueError(
                    f"Invalid relative partner id {partner_rel} for agent {agent_idx}, "
                    f"expected [0, {self.num_agents - 2}]"
                )

            opening_signal = int(opening_raw)
            pd_action = int(pd_raw)
            if opening_signal not in (0, 1):
                raise ValueError(f"Unsupported opening signal {opening_signal} for agent {agent_idx}, expected 0/1")
            if pd_action not in (0, 1):
                raise ValueError(f"Unsupported PD action {pd_action} for agent {agent_idx}, expected 0/1")

            partners_abs[agent_idx] = partner_rel if partner_rel < agent_idx else partner_rel + 1
            opening_signals[agent_idx] = np.int8(opening_signal)
            pd_actions[agent_idx] = np.int8(pd_action)

        if self._phase == self.PHASE_MATCHING:
            return self._step_selection_phase(partners_abs=partners_abs, opening_signals=opening_signals)
        return self._step_game_phase(pd_actions=pd_actions)

    def render(self):
        frame = np.full((64, 64, 3), 24, dtype=np.uint8)
        return frame

    def close(self):
        pass
