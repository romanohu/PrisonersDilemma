from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .axelrod_adapter import load_axelrod_core


ActionEnum, GameClass = load_axelrod_core()


@dataclass
class _ScriptedPolicyState:
    name: str
    grudged: bool = False


class PrisonersDilemmaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        *,
        num_agents: int = 2,
        max_steps: int = 150,
        payoff_matrix: Sequence[Sequence[float]] = ((3.0, 0.0), (5.0, 1.0)),
        scripted_opponents: Sequence[str | None] | None = None,
        scripted_seed: int = 0,
        interaction_mode: str = "all_pairs_average",
        reward_aggregation: str | None = None,
    ):
        if int(num_agents) < 2:
            raise ValueError(f"PrisonersDilemmaEnv requires at least 2 agents, got {num_agents}")
        if int(max_steps) <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")

        payoff = np.asarray(payoff_matrix, dtype=np.float32)
        if payoff.shape != (2, 2):
            raise ValueError(f"payoff_matrix must be shaped (2, 2), got {tuple(payoff.shape)}")

        self.num_agents = int(num_agents)
        self.max_steps = int(max_steps)
        self.payoff_matrix = payoff

        self.interaction_mode = self._normalize_interaction_mode(interaction_mode)
        self.reward_aggregation = self._normalize_reward_aggregation(reward_aggregation)

        # Reference: https://wrap.warwick.ac.uk/id/eprint/183331/2/WRAP-learning-partner-selection-rules-that-sustain-cooperation-social-dilemmas-with-the-option-of-opting-out-2024.pdf
        # Adaptation note: each pair plays repeated PD with the same (R,S,T,P), then each agent gets pairwise-average reward.
        r = float(payoff[0, 0])
        s = float(payoff[0, 1])
        t = float(payoff[1, 0])
        p = float(payoff[1, 1])
        self.game = GameClass(r=r, s=s, t=t, p=p)

        self._action_lookup = {
            0: ActionEnum.C,
            1: ActionEnum.D,
        }

        self._rng = np.random.default_rng(int(scripted_seed))
        self._scripted_by_agent = self._build_scripted_policies(scripted_opponents)

        # Reference: https://arxiv.org/abs/1902.03185
        # Adaptation note: memory-one observation with aggregated partner behavior for variable-sized groups.
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(6,),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = spaces.Discrete(2)
        self.render_mode = "rgb_array"
        self.is_multiagent = True

        self._all_false = [False for _ in range(self.num_agents)]
        self._step = 0
        self._last_actions = np.zeros(self.num_agents, dtype=np.int8)
        self._episode_returns = np.zeros(self.num_agents, dtype=np.float32)
        self._selected_partners = np.arange(self.num_agents, dtype=np.int32)
        self.is_terminated = True

    def _normalize_interaction_mode(self, interaction_mode: str) -> str:
        mode = str(interaction_mode).strip().lower()
        supported = {"all_pairs_average", "random_partner_with_replacement"}
        if mode not in supported:
            raise ValueError(f"Unsupported interaction_mode {interaction_mode!r}. Supported: {sorted(supported)}")
        return mode

    def _normalize_reward_aggregation(self, reward_aggregation: str | None) -> str:
        if reward_aggregation is None:
            if self.interaction_mode == "random_partner_with_replacement":
                return "sum"
            return "average"

        agg = str(reward_aggregation).strip().lower()
        supported = {"average", "sum"}
        if agg not in supported:
            raise ValueError(f"Unsupported reward_aggregation {reward_aggregation!r}. Supported: {sorted(supported)}")
        return agg

    def _build_scripted_policies(
        self,
        scripted_opponents: Sequence[str | None] | None,
    ) -> dict[int, _ScriptedPolicyState]:
        if scripted_opponents is None:
            return {}

        scripted = list(scripted_opponents)
        if len(scripted) != self.num_agents:
            raise ValueError(
                f"scripted_opponents length must match num_agents={self.num_agents}, got {len(scripted)}"
            )

        if self.num_agents != 2:
            raise ValueError("scripted_opponents are currently supported only when num_agents=2")

        supported = {"cooperator", "defector", "titfortat", "grudger", "random"}
        result: dict[int, _ScriptedPolicyState] = {}
        for agent_idx, item in enumerate(scripted):
            if item is None:
                continue
            name = str(item).strip().lower()
            if not name:
                continue
            if name not in supported:
                raise ValueError(
                    f"Unsupported scripted strategy: {item!r}. Supported: {sorted(supported)}"
                )
            result[agent_idx] = _ScriptedPolicyState(name=name)
        return result

    def _one_hot_last_action(self, action_code: int) -> np.ndarray:
        vec = np.zeros(3, dtype=np.float32)
        vec[int(action_code)] = 1.0
        return vec

    def _sample_partner(self, agent_idx: int) -> int:
        sampled = int(self._rng.integers(self.num_agents - 1))
        if sampled >= agent_idx:
            sampled += 1
        return sampled

    def _sample_all_partners(self) -> np.ndarray:
        selected = np.zeros((self.num_agents,), dtype=np.int32)
        for agent_idx in range(self.num_agents):
            selected[agent_idx] = self._sample_partner(agent_idx)
        return selected

    def _build_random_mode_partner_features(self, agent_idx: int) -> tuple[float, float]:
        partner_idx = int(self._selected_partners[agent_idx])
        partner_last = int(self._last_actions[partner_idx])
        if partner_last == 1:
            return 1.0, 0.0
        if partner_last == 2:
            return 0.0, 1.0
        return 0.0, 0.0

    def _build_observations(self) -> list[dict[str, np.ndarray]]:
        progress = np.asarray([self._step / float(self.max_steps)], dtype=np.float32)
        observations: list[dict[str, np.ndarray]] = []
        for agent_idx in range(self.num_agents):
            if self.interaction_mode == "random_partner_with_replacement":
                coop_ratio, defect_ratio = self._build_random_mode_partner_features(agent_idx)
            else:
                other_actions = np.delete(self._last_actions, agent_idx)
                coop_ratio = float(np.mean(other_actions == 1)) if other_actions.size > 0 else 0.0
                defect_ratio = float(np.mean(other_actions == 2)) if other_actions.size > 0 else 0.0

            ratios = np.asarray([coop_ratio, defect_ratio], dtype=np.float32)
            obs_vec = np.concatenate(
                [
                    self._one_hot_last_action(int(self._last_actions[agent_idx])),
                    ratios,
                    progress,
                ],
                axis=0,
            )
            observations.append({"obs": obs_vec.astype(np.float32)})
        return observations

    def _reset_state(self) -> None:
        self._step = 0
        self._last_actions[:] = 0
        self._episode_returns[:] = 0.0
        self._selected_partners[:] = np.arange(self.num_agents, dtype=np.int32)
        self.is_terminated = False
        for state in self._scripted_by_agent.values():
            state.grudged = False

    def reset(self, seed: int | None = None, options: dict | None = None):
        del options
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._reset_state()
        if self.interaction_mode == "random_partner_with_replacement":
            self._selected_partners = self._sample_all_partners()

        infos = []
        for agent_idx in range(self.num_agents):
            info = {}
            if self.interaction_mode == "random_partner_with_replacement":
                info["selected_partner"] = int(self._selected_partners[agent_idx])
            if agent_idx in self._scripted_by_agent:
                info["scripted_strategy"] = self._scripted_by_agent[agent_idx].name
            infos.append(info)
        return self._build_observations(), infos

    def _validate_actions(self, actions: Iterable[int]) -> list[int]:
        action_list = [int(action) for action in actions]
        if len(action_list) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(action_list)}")
        for action in action_list:
            if action not in self._action_lookup:
                raise ValueError(f"Unsupported action {action}. Expected 0 (C) or 1 (D).")
        return action_list

    def _scripted_action(self, agent_idx: int, state: _ScriptedPolicyState) -> int:
        partner_idx = 1 - agent_idx
        partner_last = int(self._last_actions[partner_idx])

        if state.name == "cooperator":
            return 0
        if state.name == "defector":
            return 1
        if state.name == "random":
            return int(self._rng.integers(0, 2))
        if state.name == "titfortat":
            if partner_last == 0:
                return 0
            return 0 if partner_last == 1 else 1
        if state.name == "grudger":
            if partner_last == 2:
                state.grudged = True
            return 1 if state.grudged else 0

        raise RuntimeError(f"Unknown scripted strategy state: {state}")

    def _apply_scripted_actions(self, action_list: list[int]) -> list[int]:
        if not self._scripted_by_agent:
            return action_list

        adjusted = list(action_list)
        for agent_idx, state in self._scripted_by_agent.items():
            adjusted[agent_idx] = self._scripted_action(agent_idx, state)
        return adjusted

    def _pairwise_average_rewards(self, action_list: list[int]) -> list[np.float32]:
        scores = np.zeros(self.num_agents, dtype=np.float32)

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                pair = (self._action_lookup[action_list[i]], self._action_lookup[action_list[j]])
                reward_i, reward_j = self.game.score(pair)
                scores[i] += float(reward_i)
                scores[j] += float(reward_j)

        scale = 1.0 / float(self.num_agents - 1)
        scores *= scale
        return [np.float32(value) for value in scores]

    def _random_partner_with_replacement_rewards(
        self,
        action_list: list[int],
        selected_partners: np.ndarray,
    ) -> tuple[list[np.float32], list[int]]:
        scores = np.zeros(self.num_agents, dtype=np.float32)
        interaction_counts = np.zeros(self.num_agents, dtype=np.int32)

        # Reference: https://arxiv.org/pdf/1902.03185
        # Adaptation note: selection happens before dilemma in each round; this mode randomizes selection,
        # then plays N directed dilemma interactions (one per selecting agent).
        for agent_idx in range(self.num_agents):
            partner_idx = int(selected_partners[agent_idx])
            pair = (self._action_lookup[action_list[agent_idx]], self._action_lookup[action_list[partner_idx]])
            reward_i, reward_j = self.game.score(pair)
            scores[agent_idx] += float(reward_i)
            scores[partner_idx] += float(reward_j)
            interaction_counts[agent_idx] += 1
            interaction_counts[partner_idx] += 1

        if self.reward_aggregation == "average":
            active = interaction_counts > 0
            scores[active] = scores[active] / interaction_counts[active]

        rewards = [np.float32(value) for value in scores]
        return rewards, interaction_counts.astype(np.int32).tolist()

    def step(self, actions: Iterable[int]):
        if self.is_terminated:
            obs, infos = self.reset()
            rewards = [np.float32(0.0) for _ in range(self.num_agents)]
            return obs, rewards, list(self._all_false), list(self._all_false), infos

        action_list = self._validate_actions(actions)
        action_list = self._apply_scripted_actions(action_list)

        interaction_counts: list[int] | None = None
        played_partners: np.ndarray | None = None
        if self.interaction_mode == "all_pairs_average":
            rewards = self._pairwise_average_rewards(action_list)
        else:
            played_partners = self._selected_partners.copy()
            rewards, interaction_counts = self._random_partner_with_replacement_rewards(action_list, played_partners)

        self._step += 1
        self._last_actions[:] = np.asarray(action_list, dtype=np.int8) + 1
        self._episode_returns += np.asarray(rewards, dtype=np.float32)

        if self.interaction_mode == "random_partner_with_replacement":
            self._selected_partners = self._sample_all_partners()

        terminated = self._step >= self.max_steps
        self.is_terminated = terminated

        obs = self._build_observations()
        terminations = [terminated for _ in range(self.num_agents)]
        truncations = [terminated for _ in range(self.num_agents)]
        infos = [
            {"true_objective": np.asarray(self._episode_returns[agent_idx], dtype=np.float32)}
            for agent_idx in range(self.num_agents)
        ]

        if self.interaction_mode == "random_partner_with_replacement" and interaction_counts is not None:
            assert played_partners is not None
            for agent_idx in range(self.num_agents):
                infos[agent_idx]["played_partner"] = int(played_partners[agent_idx])
                infos[agent_idx]["selected_partner"] = int(self._selected_partners[agent_idx])
                infos[agent_idx]["interaction_count"] = int(interaction_counts[agent_idx])

        for agent_idx, state in self._scripted_by_agent.items():
            infos[agent_idx]["scripted_strategy"] = state.name
            infos[agent_idx]["scripted_action"] = int(action_list[agent_idx])

        return obs, rewards, terminations, truncations, infos

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
