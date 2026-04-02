from __future__ import annotations

from typing import Iterable, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .core import PairwisePrisonersDilemmaCore
from .schedulers import InteractionScheduler, RandomPartnerScheduler


class PrisonersDilemmaEnv(gym.Env):
    """Multi-agent environment for pairwise Prisoner's Dilemma interactions.

    The game core is always pairwise 2x2 PD.
    Who plays with whom is delegated to an external scheduler.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        num_agents: int = 20,
        max_steps: int = 150,
        payoff_matrix: Sequence[Sequence[float]] = ((3.0, 0.0), (5.0, 1.0)),
        history_h: int = 1,
        scheduler: InteractionScheduler | None = None,
        seed: int = 0,
    ):
        if int(num_agents) < 2:
            raise ValueError(f"PrisonersDilemmaEnv requires at least 2 agents, got {num_agents}")
        if int(max_steps) <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if int(history_h) <= 0:
            raise ValueError(f"history_h must be positive, got {history_h}")

        self.num_agents = int(num_agents)
        self.max_steps = int(max_steps)
        self.history_h = int(history_h)

        self.core = PairwisePrisonersDilemmaCore(payoff_matrix=payoff_matrix)
        self.scheduler = scheduler or RandomPartnerScheduler()
        self._rng = np.random.default_rng(int(seed))

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
        self._step = 0
        self._last_actions = np.zeros((self.num_agents,), dtype=np.int8)  # 0: none, 1: C, 2: D
        self._action_history = np.zeros((self.num_agents, self.history_h), dtype=np.int8)
        self._episode_returns = np.zeros((self.num_agents,), dtype=np.float32)
        self._selected_partners = np.arange(self.num_agents, dtype=np.int32)
        self.is_terminated = True

    def _validate_partner_array(self, partners: np.ndarray) -> np.ndarray:
        arr = np.asarray(partners, dtype=np.int32)
        if arr.shape != (self.num_agents,):
            raise ValueError(
                f"scheduler output must be shape ({self.num_agents},), got {tuple(arr.shape)}"
            )
        if np.any(arr < 0) or np.any(arr >= self.num_agents):
            raise ValueError("scheduler output contains out-of-range partner ids")
        agent_ids = np.arange(self.num_agents, dtype=np.int32)
        if np.any(arr == agent_ids):
            raise ValueError("scheduler output contains self-selection (partner[i] == i)")
        return arr

    def _select_partners(self) -> np.ndarray:
        partners = self.scheduler.select_partners(
            num_agents=self.num_agents,
            rng=self._rng,
            action_history=self._action_history.copy(),
            step=self._step,
        )
        return self._validate_partner_array(partners)

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
            partner_idx = int(self._selected_partners[agent_idx])
            obs_vec = self._encode_action_history(self._action_history[partner_idx]).astype(np.float32)
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
        self._step = 0
        self._last_actions[:] = 0
        self._action_history[:, :] = 0
        self._episode_returns[:] = 0.0
        self.is_terminated = False
        self._selected_partners = self._select_partners()

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._reset_state()
        observations = self._build_observations()
        infos = [{"selected_partner": int(self._selected_partners[i])} for i in range(self.num_agents)]
        return observations, infos

    def step(self, actions: Iterable[int]):
        if self.is_terminated:
            obs, infos = self.reset()
            rewards = [np.float32(0.0) for _ in range(self.num_agents)]
            return obs, rewards, list(self._all_false), list(self._all_false), infos

        action_array = self._validate_actions(actions)
        played_partners = self._selected_partners.copy()

        rewards_array, interaction_counts = self.core.compute_round_rewards(action_array, played_partners)
        rewards_array = rewards_array.astype(np.float32)

        self._step += 1
        self._last_actions = action_array + 1  # 1: C, 2: D
        if self.history_h > 1:
            self._action_history[:, 1:] = self._action_history[:, :-1]
        self._action_history[:, 0] = self._last_actions
        self._episode_returns += rewards_array

        self._selected_partners = self._select_partners()
        terminated = self._step >= self.max_steps
        self.is_terminated = terminated

        observations = self._build_observations()
        rewards = [np.float32(value) for value in rewards_array]
        terminations = [terminated for _ in range(self.num_agents)]
        truncations = [terminated for _ in range(self.num_agents)]

        infos: list[dict] = []
        for i in range(self.num_agents):
            infos.append(
                {
                    "true_objective": np.asarray(self._episode_returns[i], dtype=np.float32),
                    "played_partner": int(played_partners[i]),
                    "selected_partner": int(self._selected_partners[i]),
                    "interaction_count": int(interaction_counts[i]),
                    "episode_extra_stats": {"last_action": int(action_array[i])},
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
        tile_w = 44
        frame_w = max(176, 8 + self.num_agents * (tile_w + 8))
        frame = np.full((96, frame_w, 3), 25, dtype=np.uint8)

        for agent_idx in range(self.num_agents):
            x0 = 8 + agent_idx * (tile_w + 8)
            frame[8:42, x0 : x0 + tile_w] = self._action_color(int(self._last_actions[agent_idx]))

        progress_width = int((frame_w - 20) * (self._step / float(self.max_steps)))
        frame[74:86, 10 : 10 + progress_width] = np.asarray([82, 156, 255], dtype=np.uint8)
        return frame

    def close(self):
        pass
