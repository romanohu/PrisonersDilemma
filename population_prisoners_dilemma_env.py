from __future__ import annotations

from typing import Iterable, Optional, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .core import PairwisePrisonersDilemmaCore


class PopulationPrisonersDilemmaEnv(gym.Env):
    """Population PD with partner selection and sequential directed interactions.

    One round runs in num_agents env steps:
    1) Round-start step (phase=selection before step): partner map is resolved
       and selector i=0 interaction is executed.
    2) Remaining dilemma steps: selector order i=1..N-1, each as i -> partner[i].

    Action for each agent is always a tuple:
    - partner head: Discrete(num_agents - 1)
    - pd head: Discrete(2), where 0=C and 1=D
    """

    metadata = {"render_modes": ["rgb_array"]}

    # Public modes we keep in docs/CLI.
    _SUPPORTED_PARTNER_MODES = {"from_actions", "random"}
    # Backward compatibility aliases; normalized internally.
    _PARTNER_MODE_ALIASES = {
        "random_with_replacement": "random",
        "random_with_replacement_each_step": "random",
    }

    _PHASE_SELECTION = 0
    _PHASE_DILEMMA = 1

    def __init__(
        self,
        *,
        num_agents: int = 8,
        max_steps: int = 150,
        payoff_matrix: Sequence[Sequence[float]] = ((3.0, 0.0), (5.0, 1.0)),
        history_h: int = 1,
        seed: int = 0,
        partner_scheduler: str = "from_actions",
    ):
        self.num_agents = int(num_agents)
        self.max_steps = int(max_steps)
        self.history_h = int(history_h)

        partner_mode = self._PARTNER_MODE_ALIASES.get(str(partner_scheduler), str(partner_scheduler))
        self.partner_mode = partner_mode

        if self.num_agents < 2:
            raise ValueError(f"num_agents must be >= 2, got {num_agents}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if self.history_h <= 0:
            raise ValueError(f"history_h must be positive, got {history_h}")
        if self.partner_mode not in self._SUPPORTED_PARTNER_MODES:
            raise ValueError(
                f"Unsupported partner_scheduler={partner_scheduler}. "
                f"Use one of {sorted(self._SUPPORTED_PARTNER_MODES)}"
            )

        self.core = PairwisePrisonersDilemmaCore(payoff_matrix=payoff_matrix)
        self._rng = np.random.default_rng(int(seed))

        self.action_space = spaces.Tuple((spaces.Discrete(self.num_agents - 1), spaces.Discrete(2)))

        self._obs_hist_dim = (self.num_agents - 1) * (2 * self.history_h)
        obs_dim = self._obs_hist_dim + 1
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(obs_dim,),
                    dtype=np.float32,
                )
            }
        )

        self.render_mode = "rgb_array"
        self.is_multiagent = True

        self._all_false = [False for _ in range(self.num_agents)]

        # Episode-level state
        self._round = 0
        self._phase = self._PHASE_SELECTION
        self._last_actions = np.full((self.num_agents,), -1, dtype=np.int8)  # selector-side actions of last round
        self._action_history = np.full((self.num_agents, self.history_h), -1, dtype=np.int8)
        self._episode_returns = np.zeros((self.num_agents,), dtype=np.float32)
        self._cooperate_counts = np.zeros((self.num_agents,), dtype=np.int32)
        self._defect_counts = np.zeros((self.num_agents,), dtype=np.int32)
        self._selected_counts = np.zeros((self.num_agents,), dtype=np.int32)
        self._env_total_reward = 0.0

        # Round-level state
        self._last_partners = np.full((self.num_agents,), -1, dtype=np.int32)
        self._last_interaction_counts = np.zeros((self.num_agents,), dtype=np.int32)
        self._current_partners: Optional[np.ndarray] = None
        self._round_interaction_counts = np.zeros((self.num_agents,), dtype=np.int32)
        self._round_selector_actions = np.full((self.num_agents,), -1, dtype=np.int8)
        self._dilemma_cursor = 0
        self._active_selector: Optional[int] = None

        # Optional partner override consumed at the next selection boundary.
        self._pending_partners: Optional[np.ndarray] = None

        self.is_terminated = True

    def _sample_random_partners(self) -> np.ndarray:
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
        # One-shot override that is consumed at next selection boundary.
        self._pending_partners = self._validate_partners(partners)

    def should_accept_partner_assignment(self) -> bool:
        # External policy mapping should only push partner assignments at boundaries.
        return bool(self.is_terminated or self._phase == self._PHASE_SELECTION)

    def _decode_relative_partner(self, agent_idx: int, rel_partner: int) -> int:
        if rel_partner < 0 or rel_partner >= self.num_agents - 1:
            raise ValueError(
                f"Invalid relative partner id {rel_partner} for agent {agent_idx}, "
                f"expected [0, {self.num_agents - 2}]"
            )
        return rel_partner if rel_partner < agent_idx else rel_partner + 1

    def _parse_actions(self, actions: Iterable) -> tuple[np.ndarray, np.ndarray]:
        action_list = list(actions)
        if len(action_list) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(action_list)}")

        partners = np.zeros((self.num_agents,), dtype=np.int32)
        pd_actions = np.zeros((self.num_agents,), dtype=np.int8)

        for agent_idx, action_entry in enumerate(action_list):
            if isinstance(action_entry, dict):
                if "partner" not in action_entry or "pd" not in action_entry:
                    raise ValueError(
                        f"Action dict for agent {agent_idx} must contain keys 'partner' and 'pd', got {action_entry}"
                    )
                partner_raw = action_entry["partner"]
                pd_raw = action_entry["pd"]
            elif isinstance(action_entry, (tuple, list, np.ndarray)):
                if len(action_entry) != 2:
                    raise ValueError(
                        f"Action for agent {agent_idx} must have 2 elements: (partner, pd), got {action_entry}"
                    )
                partner_raw = action_entry[0]
                pd_raw = action_entry[1]
            else:
                raise ValueError(
                    f"Unsupported action format for agent {agent_idx}: {type(action_entry)}. "
                    "Expected dict{'partner','pd'} or (partner, pd)."
                )

            pd_action = int(pd_raw)
            if pd_action not in (0, 1):
                raise ValueError(
                    f"Unsupported PD action {pd_action} for agent {agent_idx}. Expected 0(C) or 1(D)."
                )

            partner_rel = int(partner_raw)
            partners[agent_idx] = self._decode_relative_partner(agent_idx, partner_rel)
            pd_actions[agent_idx] = pd_action

        return partners, pd_actions

    def _resolve_partners_for_round(self, partners_from_actions: np.ndarray) -> np.ndarray:
        if self._pending_partners is not None:
            partners = self._pending_partners.copy()
            self._pending_partners = None
            return partners

        if self.partner_mode == "random":
            return self._sample_random_partners()

        return partners_from_actions

    def _encode_action_history(self, history_codes: np.ndarray) -> np.ndarray:
        features = np.zeros((2 * self.history_h,), dtype=np.float32)
        for lag in range(self.history_h):
            code = int(history_codes[lag])
            if code == 0:
                features[2 * lag] = 1.0
            elif code == 1:
                features[2 * lag + 1] = 1.0
        return features

    def _other_slot_offset(self, agent_idx: int, other_idx: int) -> int:
        if other_idx == agent_idx:
            raise ValueError("other_idx must be different from agent_idx")
        rel = other_idx if other_idx < agent_idx else other_idx - 1
        return rel * (2 * self.history_h)

    def _build_observations(self) -> list[dict[str, np.ndarray]]:
        obs: list[dict[str, np.ndarray]] = []
        obs_dim = self._obs_hist_dim + 1

        selector_idx = int(self._active_selector) if self._active_selector is not None else None
        selected_idx = None
        if self._phase == self._PHASE_DILEMMA:
            if self._current_partners is None or selector_idx is None:
                raise RuntimeError("Dilemma phase requires current partners and active selector")
            selected_idx = int(self._current_partners[selector_idx])

        for agent_idx in range(self.num_agents):
            obs_vec = np.zeros((obs_dim,), dtype=np.float32)

            if self._phase == self._PHASE_SELECTION:
                cursor = 0
                for other_idx in range(self.num_agents):
                    if other_idx == agent_idx:
                        continue
                    encoded = self._encode_action_history(self._action_history[other_idx])
                    next_cursor = cursor + encoded.shape[0]
                    obs_vec[cursor:next_cursor] = encoded
                    cursor = next_cursor
                obs_vec[-1] = float(self._PHASE_SELECTION)
            else:
                # In dilemma phase, only the currently interacting pair sees each other.
                opponent_idx: Optional[int] = None
                if agent_idx == selector_idx:
                    opponent_idx = selected_idx
                elif agent_idx == selected_idx:
                    opponent_idx = selector_idx

                if opponent_idx is not None:
                    encoded = self._encode_action_history(self._action_history[opponent_idx])
                    offset = self._other_slot_offset(agent_idx, opponent_idx)
                    obs_vec[offset : offset + encoded.shape[0]] = encoded
                obs_vec[-1] = float(self._PHASE_DILEMMA)

            obs.append({"obs": obs_vec})

        return obs

    def _build_infos(
        self,
        *,
        partners: np.ndarray,
        interaction_counts: np.ndarray,
        pd_actions: np.ndarray,
        partner_last_actions: np.ndarray,
        played_partners: np.ndarray,
        terminated: bool,
    ) -> list[dict]:
        infos: list[dict] = []
        for i in range(self.num_agents):
            total_actions = int(self._cooperate_counts[i] + self._defect_counts[i])
            cooperate_ratio = float(self._cooperate_counts[i] / total_actions) if total_actions > 0 else 0.0
            defect_ratio = float(self._defect_counts[i] / total_actions) if total_actions > 0 else 0.0
            infos.append(
                {
                    "true_objective": np.asarray(self._episode_returns[i], dtype=np.float32),
                    "played_partner": int(played_partners[i]),
                    "selected_partner": int(partners[i]),
                    "interaction_count": int(interaction_counts[i]),
                    "episode_extra_stats": {
                        "last_action": int(pd_actions[i]),
                        "partner_last_action": int(partner_last_actions[i]),
                        "cooperate_count": int(self._cooperate_counts[i]),
                        "defect_count": int(self._defect_counts[i]),
                        "cooperate_ratio": cooperate_ratio,
                        "defect_ratio": defect_ratio,
                        "selected_count_episode": int(self._selected_counts[i]),
                        "env_total_reward": float(self._env_total_reward),
                    },
                    "is_active": not terminated,
                }
            )
        return infos

    def _reset_state(self) -> None:
        self._round = 0
        self._phase = self._PHASE_SELECTION
        self._last_actions[:] = -1
        self._action_history[:, :] = -1
        self._episode_returns[:] = 0.0
        self._cooperate_counts[:] = 0
        self._defect_counts[:] = 0
        self._selected_counts[:] = 0
        self._env_total_reward = 0.0
        self._last_partners[:] = -1
        self._last_interaction_counts[:] = 0
        self._current_partners = None
        self._round_interaction_counts[:] = 0
        self._round_selector_actions[:] = -1
        self._dilemma_cursor = 0
        self._active_selector = None
        self._pending_partners = None
        self.is_terminated = False

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        self._reset_state()

        options = options or {}
        if "partners" in options:
            self._pending_partners = self._validate_partners(options["partners"])

        observations = self._build_observations()
        infos = []
        for i in range(self.num_agents):
            preview_partner = int(self._pending_partners[i]) if self._pending_partners is not None else -1
            infos.append(
                {
                    "selected_partner": preview_partner,
                    "played_partner": -1,
                    "interaction_count": 0,
                    "is_active": True,
                }
            )
        return observations, infos

    def step(self, actions: Iterable):
        # Sample Factory can call step after done; we keep auto-reset behavior.
        if self.is_terminated:
            reset_options = None
            if self._pending_partners is not None:
                reset_options = {"partners": self._pending_partners}
            obs, infos = self.reset(options=reset_options)
            rewards = [np.float32(0.0) for _ in range(self.num_agents)]
            return obs, rewards, list(self._all_false), list(self._all_false), infos

        partners_from_actions, pd_actions = self._parse_actions(actions)

        if self._phase == self._PHASE_SELECTION:
            partners = self._resolve_partners_for_round(partners_from_actions)
            interaction_counts = np.ones((self.num_agents,), dtype=np.int32)
            np.add.at(interaction_counts, partners, 1)

            self._current_partners = partners.copy()
            self._last_partners = partners.copy()
            self._last_interaction_counts = interaction_counts.copy()
            self._round_interaction_counts = interaction_counts.copy()
            self._round_selector_actions[:] = -1
            self._dilemma_cursor = 0
            self._active_selector = 0
            self._phase = self._PHASE_DILEMMA

        if self._current_partners is None or self._active_selector is None:
            raise RuntimeError("Dilemma phase requires partners selected at the current round boundary")

        partners = self._current_partners.copy()
        selector_idx = int(self._active_selector)
        selected_idx = int(partners[selector_idx])
        selector_action = int(pd_actions[selector_idx])
        selected_action = int(pd_actions[selected_idx])

        rewards_array = np.zeros((self.num_agents,), dtype=np.float32)
        rewards_array[selector_idx] += np.float32(self.core.payoff[selector_action, selected_action])
        rewards_array[selected_idx] += np.float32(self.core.payoff[selected_action, selector_action])

        played_partners = np.full((self.num_agents,), -1, dtype=np.int32)
        played_partners[selector_idx] = selected_idx
        played_partners[selected_idx] = selector_idx

        partner_last_actions = np.full((self.num_agents,), -1, dtype=np.int8)
        partner_last_actions[selector_idx] = np.int8(selected_action)
        partner_last_actions[selected_idx] = np.int8(selector_action)

        # We only commit selector-side actions into persistent round history.
        self._round_selector_actions[selector_idx] = np.int8(selector_action)
        self._last_partners = partners.copy()
        self._last_interaction_counts = self._round_interaction_counts.copy()
        self._episode_returns += rewards_array

        if selector_action == 0:
            self._cooperate_counts[selector_idx] += 1
        else:
            self._defect_counts[selector_idx] += 1

        if selected_action == 0:
            self._cooperate_counts[selected_idx] += 1
        else:
            self._defect_counts[selected_idx] += 1

        self._selected_counts[selected_idx] += 1
        self._env_total_reward += float(np.sum(rewards_array))

        round_complete = self._dilemma_cursor + 1 >= self.num_agents
        if round_complete:
            self._last_actions = self._round_selector_actions.copy()
            if self.history_h > 1:
                self._action_history[:, 1:] = self._action_history[:, :-1]
            self._action_history[:, 0] = self._last_actions

            self._round += 1
            terminated = self._round >= self.max_steps
            self.is_terminated = terminated

            self._current_partners = None
            self._active_selector = None
            self._phase = self._PHASE_SELECTION
        else:
            self._dilemma_cursor += 1
            self._active_selector = self._dilemma_cursor
            terminated = False

        observations = self._build_observations()
        rewards = [np.float32(value) for value in rewards_array]
        terminations = [terminated for _ in range(self.num_agents)]
        truncations = [terminated for _ in range(self.num_agents)]
        infos = self._build_infos(
            partners=partners,
            interaction_counts=self._round_interaction_counts,
            pd_actions=self._round_selector_actions.copy(),
            partner_last_actions=partner_last_actions,
            played_partners=played_partners,
            terminated=terminated,
        )
        return observations, rewards, terminations, truncations, infos

    def _action_color(self, action_code: int) -> np.ndarray:
        if action_code == 0:
            return np.asarray([62, 180, 96], dtype=np.uint8)
        if action_code == 1:
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
