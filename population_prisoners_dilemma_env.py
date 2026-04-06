from __future__ import annotations

from typing import Iterable, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .core import PairwisePrisonersDilemmaCore


class PopulationPrisonersDilemmaEnv(gym.Env):
    """Repeated population PD environment with directed partner assignments.

    Each agent i plays against one selected partner partners[i] at every step.
    Multiple selectors can choose the same partner in the same round.
    """

    metadata = {"render_modes": ["rgb_array"]}
    _SUPPORTED_PARTNER_SCHEDULERS = {"random_with_replacement", "random_with_replacement_each_step"}

    def __init__(
        self,
        *,
        num_agents: int = 8,
        max_steps: int = 150,
        payoff_matrix: Sequence[Sequence[float]] = ((3.0, 0.0), (5.0, 1.0)),
        history_h: int = 1,
        seed: int = 0,
        partner_scheduler: str = "random_with_replacement",
    ):
        self.num_agents = int(num_agents)
        self.max_steps = int(max_steps)
        self.history_h = int(history_h)
        self.partner_scheduler = str(partner_scheduler)

        if self.num_agents < 2:
            raise ValueError(f"num_agents must be >= 2, got {num_agents}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if self.history_h <= 0:
            raise ValueError(f"history_h must be positive, got {history_h}")
        if self.partner_scheduler not in self._SUPPORTED_PARTNER_SCHEDULERS:
            raise ValueError(f"Unsupported partner_scheduler: {partner_scheduler}")

        self.core = PairwisePrisonersDilemmaCore(payoff_matrix=payoff_matrix)
        self._rng = np.random.default_rng(int(seed))

        # Each agent outputs one C/D action for every possible opponent (excluding self).
        self.action_space = spaces.MultiDiscrete(np.full((self.num_agents - 1,), 2, dtype=np.int64))
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2 * self.history_h * (self.num_agents - 1),),
                    dtype=np.float32,
                )
            }
        )
        self.render_mode = "rgb_array"
        self.is_multiagent = True

        self._all_false = [False for _ in range(self.num_agents)]
        self._step = 0
        self._last_actions = np.zeros((self.num_agents,), dtype=np.int8)  # 0: none, 1: C, 2: D
        self._action_history = np.zeros((self.num_agents, self.history_h), dtype=np.int8)
        self._episode_returns = np.zeros((self.num_agents,), dtype=np.float32)
        self._partners = np.zeros((self.num_agents,), dtype=np.int32)
        self._opponent_ids = [
            np.asarray([opponent for opponent in range(self.num_agents) if opponent != agent_idx], dtype=np.int32)
            for agent_idx in range(self.num_agents)
        ]
        self._opponent_to_col = np.full((self.num_agents, self.num_agents), -1, dtype=np.int32)
        for agent_idx in range(self.num_agents):
            for col_idx, opponent_idx in enumerate(self._opponent_ids[agent_idx]):
                self._opponent_to_col[agent_idx, int(opponent_idx)] = int(col_idx)
        self.is_terminated = True

    def _sample_partners(self) -> np.ndarray:
        partners = np.zeros((self.num_agents,), dtype=np.int32)
        candidate_ids = np.arange(self.num_agents, dtype=np.int32)
        for agent_idx in range(self.num_agents):
            valid = candidate_ids[candidate_ids != agent_idx]
            partners[agent_idx] = int(self._rng.choice(valid))
        return partners

    def _validate_partners(self, partners: Iterable[int]) -> np.ndarray:
        partner_array = np.asarray(list(partners), dtype=np.int32)
        if partner_array.shape != (self.num_agents,):
            raise ValueError(f"Expected {self.num_agents} partner ids, got shape {tuple(partner_array.shape)}")
        if np.any(partner_array < 0) or np.any(partner_array >= self.num_agents):
            raise ValueError("partner ids must be within [0, num_agents)")
        if np.any(partner_array == np.arange(self.num_agents, dtype=np.int32)):
            raise ValueError("Self-partnering is not allowed (partners[i] must be != i)")
        return partner_array

    def set_partners(self, partners: Iterable[int]) -> None:
        self._partners = self._validate_partners(partners)

    def _encode_action_history(self, history_codes: np.ndarray) -> np.ndarray:
        features = np.zeros((2 * self.history_h,), dtype=np.float32)
        for lag in range(self.history_h):
            action_code = int(history_codes[lag])
            if action_code == 1:
                features[2 * lag] = 1.0
            elif action_code == 2:
                features[2 * lag + 1] = 1.0
        return features

    def _build_observations(self) -> list[dict[str, np.ndarray]]:
        obs = []
        for agent_idx in range(self.num_agents):
            opp_histories = [self._encode_action_history(self._action_history[int(opponent_idx)]) for opponent_idx in self._opponent_ids[agent_idx]]
            obs_vec = np.concatenate(opp_histories, axis=0).astype(np.float32)
            obs.append({"obs": obs_vec})
        return obs

    def _validate_actions(self, actions: Iterable[int]) -> np.ndarray:
        action_array = np.asarray(list(actions), dtype=np.int8)
        if action_array.shape == (self.num_agents,):
            # Backward-compatibility: one action per agent, broadcast to all opponents.
            if np.any((action_array != 0) & (action_array != 1)):
                raise ValueError("Unsupported action values. Expected only 0 (C) or 1 (D).")
            return np.repeat(action_array.reshape(self.num_agents, 1), self.num_agents - 1, axis=1)

        if action_array.shape != (self.num_agents, self.num_agents - 1):
            raise ValueError(
                f"Expected actions shaped ({self.num_agents}, {self.num_agents - 1}) "
                f"or ({self.num_agents},), got shape {tuple(action_array.shape)}"
            )
        if np.any((action_array != 0) & (action_array != 1)):
            raise ValueError("Unsupported action values. Expected only 0 (C) or 1 (D).")
        return action_array

    def _action_for_pair(self, action_matrix: np.ndarray, actor_idx: int, opponent_idx: int) -> int:
        col_idx = int(self._opponent_to_col[actor_idx, opponent_idx])
        if col_idx < 0:
            raise ValueError(f"Self interaction is not allowed: actor={actor_idx}, opponent={opponent_idx}")
        return int(action_matrix[actor_idx, col_idx])

    def _reset_state(self) -> None:
        self._step = 0
        self._last_actions[:] = 0
        self._action_history[:, :] = 0
        self._episode_returns[:] = 0.0
        self.is_terminated = False

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._reset_state()

        options = options or {}
        if "partners" in options:
            self.set_partners(options["partners"])
        elif self.partner_scheduler in self._SUPPORTED_PARTNER_SCHEDULERS:
            self._partners = self._sample_partners()
        else:
            raise ValueError(f"Unsupported partner scheduler: {self.partner_scheduler}")

        observations = self._build_observations()
        infos = [{"selected_partner": int(self._partners[i])} for i in range(self.num_agents)]
        return observations, infos

    def step(self, actions: Iterable[int]):
        if self.is_terminated:
            obs, infos = self.reset()
            rewards = [np.float32(0.0) for _ in range(self.num_agents)]
            return obs, rewards, list(self._all_false), list(self._all_false), infos

        action_matrix = self._validate_actions(actions)
        played_partners = self._partners.copy()
        payoff = self.core.payoff
        rewards_array = np.zeros((self.num_agents,), dtype=np.float32)
        interaction_counts = np.ones((self.num_agents,), dtype=np.int32)
        np.add.at(interaction_counts, played_partners, 1)
        selector_actions = np.zeros((self.num_agents,), dtype=np.int8)

        for selector_idx in range(self.num_agents):
            partner_idx = int(played_partners[selector_idx])
            selector_action = self._action_for_pair(action_matrix, selector_idx, partner_idx)
            partner_response = self._action_for_pair(action_matrix, partner_idx, selector_idx)
            selector_actions[selector_idx] = selector_action + 1

            rewards_array[selector_idx] += np.float32(payoff[selector_action, partner_response])
            rewards_array[partner_idx] += np.float32(payoff[partner_response, selector_action])

        self._step += 1
        self._last_actions = selector_actions  # 1: C, 2: D (selector action against its chosen partner)
        if self.history_h > 1:
            self._action_history[:, 1:] = self._action_history[:, :-1]
        self._action_history[:, 0] = self._last_actions
        self._episode_returns += rewards_array

        terminated = self._step >= self.max_steps
        self.is_terminated = terminated

        if self.partner_scheduler == "random_with_replacement_each_step" and not terminated:
            self._partners = self._sample_partners()

        observations = self._build_observations()
        rewards = [np.float32(value) for value in rewards_array]
        terminations = [terminated for _ in range(self.num_agents)]
        truncations = [terminated for _ in range(self.num_agents)]

        infos: list[dict] = []
        for i in range(self.num_agents):
            partner = int(played_partners[i])
            selector_action = self._action_for_pair(action_matrix, i, partner)
            partner_response = self._action_for_pair(action_matrix, partner, i)
            infos.append(
                {
                    "true_objective": np.asarray(self._episode_returns[i], dtype=np.float32),
                    "played_partner": partner,
                    "selected_partner": partner,
                    "interaction_count": int(interaction_counts[i]),
                    "episode_extra_stats": {
                        "last_action": int(selector_action),
                        "partner_last_action": int(partner_response),
                    },
                }
            )

        return observations, rewards, terminations, truncations, infos

    def _action_color(self, action_code: int) -> np.ndarray:
        if action_code == 1:
            return np.asarray([62, 180, 96], dtype=np.uint8)
        if action_code == 2:
            return np.asarray([220, 80, 70], dtype=np.uint8)
        return np.asarray([130, 130, 130], dtype=np.uint8)

    def render(self):
        tile_w = 30
        gap = 4
        frame_w = max(240, 8 + self.num_agents * (tile_w + gap))
        frame = np.full((96, frame_w, 3), 25, dtype=np.uint8)

        for agent_idx in range(self.num_agents):
            x0 = 8 + agent_idx * (tile_w + gap)
            frame[8:34, x0 : x0 + tile_w] = self._action_color(int(self._last_actions[agent_idx]))

        progress_width = int((frame_w - 20) * (self._step / float(self.max_steps)))
        frame[74:86, 10 : 10 + progress_width] = np.asarray([82, 156, 255], dtype=np.uint8)
        return frame

    def close(self):
        pass
