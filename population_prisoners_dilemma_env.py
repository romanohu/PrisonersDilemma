from __future__ import annotations

from typing import Iterable, Optional, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .core import PairwisePrisonersDilemmaCore


class PopulationPrisonersDilemmaEnv(gym.Env):
    """Repeated population PD environment with directed partner assignments.

    Design note:
    - Public action space is Discrete(2) per agent (C/D only).
    - One environment step processes one directed interaction in the current round.
    - A round consists of ``num_agents`` directed interactions: selector i plays against
      partner partners[i].
    - Episode length ``max_steps`` is measured in rounds.

    This keeps policy action space C/D-only while allowing selected agents to output
    potentially different responses to multiple selectors within a single round.
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

        # Keep C/D-only policy action space.
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2 * self.history_h,),
                    dtype=np.float32,
                )
            }
        )
        self.render_mode = "rgb_array"
        self.is_multiagent = True

        self._all_false = [False for _ in range(self.num_agents)]
        self._round = 0
        self._last_actions = np.zeros((self.num_agents,), dtype=np.int8)  # 0: none, 1: C, 2: D
        self._action_history = np.zeros((self.num_agents, self.history_h), dtype=np.int8)
        self._episode_returns = np.zeros((self.num_agents,), dtype=np.float32)

        # Partner assignment control.
        self._partners = np.zeros((self.num_agents,), dtype=np.int32)
        self._pending_partners: Optional[np.ndarray] = None

        # Round-local state.
        self._round_initialized = False
        self._interaction_cursor = 0
        self._round_selector_actions = np.full((self.num_agents,), -1, dtype=np.int8)
        self._round_pair_actions = np.full((self.num_agents, self.num_agents), -1, dtype=np.int8)
        self._round_interaction_counts = np.ones((self.num_agents,), dtype=np.int32)
        self._next_active = (0, 1)

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
        # Applied at the next round boundary.
        self._pending_partners = self._validate_partners(partners)

    def _encode_action_history(self, history_codes: np.ndarray) -> np.ndarray:
        features = np.zeros((2 * self.history_h,), dtype=np.float32)
        for lag in range(self.history_h):
            action_code = int(history_codes[lag])
            if action_code == 1:
                features[2 * lag] = 1.0
            elif action_code == 2:
                features[2 * lag + 1] = 1.0
        return features

    def _active_pair(self) -> tuple[int, int]:
        selector = int(self._interaction_cursor)
        partner = int(self._partners[selector])
        return selector, partner

    def _build_observations(self) -> list[dict[str, np.ndarray]]:
        selector, partner = self._next_active
        zero = np.zeros((2 * self.history_h,), dtype=np.float32)

        obs: list[dict[str, np.ndarray]] = []
        for agent_idx in range(self.num_agents):
            if agent_idx == selector:
                opponent_idx = partner
                obs_vec = self._encode_action_history(self._action_history[opponent_idx]).astype(np.float32)
            elif agent_idx == partner:
                opponent_idx = selector
                obs_vec = self._encode_action_history(self._action_history[opponent_idx]).astype(np.float32)
            else:
                obs_vec = zero.copy()
            obs.append({"obs": obs_vec})
        return obs

    def _validate_actions(self, actions: Iterable[int]) -> np.ndarray:
        action_array = np.asarray(list(actions), dtype=np.int8)
        if action_array.shape != (self.num_agents,):
            raise ValueError(f"Expected {self.num_agents} actions, got shape {tuple(action_array.shape)}")
        if np.any((action_array != 0) & (action_array != 1)):
            raise ValueError("Unsupported action values. Expected only 0 (C) or 1 (D).")
        return action_array

    def _reset_state(self) -> None:
        self._round = 0
        self._last_actions[:] = 0
        self._action_history[:, :] = 0
        self._episode_returns[:] = 0.0
        self._pending_partners = None

        self._round_initialized = False
        self._interaction_cursor = 0
        self._round_selector_actions[:] = -1
        self._round_pair_actions[:, :] = -1
        self._round_interaction_counts[:] = 1
        self._next_active = (0, 1)

        self.is_terminated = False

    def _begin_round(self) -> None:
        if self._pending_partners is not None:
            self._partners = self._pending_partners
            self._pending_partners = None
        elif self._round > 0 and self.partner_scheduler == "random_with_replacement_each_step":
            self._partners = self._sample_partners()

        self._interaction_cursor = 0
        self._round_selector_actions[:] = -1
        self._round_pair_actions[:, :] = -1
        self._round_interaction_counts[:] = 1
        np.add.at(self._round_interaction_counts, self._partners, 1)

        self._round_initialized = True
        self._next_active = self._active_pair()

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._reset_state()

        options = options or {}
        if "partners" in options:
            self._partners = self._validate_partners(options["partners"])
        elif self.partner_scheduler in self._SUPPORTED_PARTNER_SCHEDULERS:
            self._partners = self._sample_partners()
        else:
            raise ValueError(f"Unsupported partner scheduler: {self.partner_scheduler}")

        self._begin_round()

        observations = self._build_observations()
        selector, _ = self._next_active
        partner_preview = int(self._partners[selector])
        infos = [
            {
                "selected_partner": int(partner_preview if agent_idx == selector else self._partners[agent_idx]),
                "played_partner": int(partner_preview if agent_idx == selector else self._partners[agent_idx]),
                "interaction_count": int(self._round_interaction_counts[agent_idx]),
                "is_active": bool(agent_idx in self._next_active),
            }
            for agent_idx in range(self.num_agents)
        ]
        return observations, infos

    def step(self, actions: Iterable[int]):
        if self.is_terminated:
            obs, infos = self.reset()
            rewards = [np.float32(0.0) for _ in range(self.num_agents)]
            return obs, rewards, list(self._all_false), list(self._all_false), infos

        if not self._round_initialized:
            self._begin_round()

        action_array = self._validate_actions(actions)
        selector, partner = self._active_pair()

        selector_action = int(action_array[selector])
        partner_action = int(action_array[partner])
        payoff = self.core.payoff

        rewards_array = np.zeros((self.num_agents,), dtype=np.float32)
        rewards_array[selector] += np.float32(payoff[selector_action, partner_action])
        rewards_array[partner] += np.float32(payoff[partner_action, selector_action])
        self._episode_returns += rewards_array

        self._round_selector_actions[selector] = selector_action
        self._round_pair_actions[selector, partner] = selector_action
        self._round_pair_actions[partner, selector] = partner_action

        self._interaction_cursor += 1
        round_finished = self._interaction_cursor >= self.num_agents

        if round_finished:
            # History stores one selector-side action per agent per round.
            self._last_actions = self._round_selector_actions + 1
            if self.history_h > 1:
                self._action_history[:, 1:] = self._action_history[:, :-1]
            self._action_history[:, 0] = self._last_actions

            self._round += 1
            terminated = self._round >= self.max_steps
            self.is_terminated = terminated

            if terminated:
                self._next_active = (0, 1)
            else:
                # Next step starts a new round; partner override (if any) is consumed there.
                self._round_initialized = False
                if self._pending_partners is not None:
                    next_partners = self._pending_partners
                elif self.partner_scheduler == "random_with_replacement_each_step":
                    next_partners = self._sample_partners()
                else:
                    next_partners = self._partners
                next_selector = 0
                next_partner = int(next_partners[next_selector])
                self._next_active = (next_selector, next_partner)
        else:
            terminated = False
            self.is_terminated = False
            self._next_active = self._active_pair()

        observations = self._build_observations()
        rewards = [np.float32(value) for value in rewards_array]
        terminations = [terminated for _ in range(self.num_agents)]
        truncations = [terminated for _ in range(self.num_agents)]

        infos: list[dict] = []
        for i in range(self.num_agents):
            partner_i = int(self._partners[i])

            if round_finished:
                selector_last = int(self._round_selector_actions[i])
                partner_last = int(self._round_pair_actions[partner_i, i])
            elif i == selector:
                selector_last = selector_action
                partner_last = partner_action
            elif i == partner:
                selector_last = partner_action
                partner_last = selector_action
            else:
                selector_last = -1
                partner_last = -1

            infos.append(
                {
                    "true_objective": np.asarray(self._episode_returns[i], dtype=np.float32),
                    "played_partner": partner_i,
                    "selected_partner": partner_i,
                    "interaction_count": int(self._round_interaction_counts[i]),
                    "episode_extra_stats": {
                        "last_action": int(selector_last),
                        "partner_last_action": int(partner_last),
                    },
                    "is_active": bool(i in self._next_active) if not terminated else False,
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

        progress_width = int((frame_w - 20) * (self._round / float(self.max_steps)))
        frame[74:86, 10 : 10 + progress_width] = np.asarray([82, 156, 255], dtype=np.uint8)
        return frame

    def close(self):
        pass
