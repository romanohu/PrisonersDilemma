from __future__ import annotations

from typing import Sequence

import numpy as np


class PairwisePrisonersDilemmaCore:
    """Pairwise 2x2 Prisoner's Dilemma reward core.

    Each directed interaction (i -> j) yields two rewards:
    - reward for selector i using payoff[a_i, a_j]
    - reward for selected j using payoff[a_j, a_i]
    """

    def __init__(self, payoff_matrix: Sequence[Sequence[float]] = ((3.0, 0.0), (4.0, 1.0))):
        payoff = np.asarray(payoff_matrix, dtype=np.float32)
        if payoff.shape != (2, 2):
            raise ValueError(f"payoff_matrix must be shaped (2, 2), got {tuple(payoff.shape)}")
        self.payoff = payoff

    def compute_round_rewards(self, actions: np.ndarray, partners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute rewards for one round of directed pairwise interactions.

        Args:
            actions: shape (N,), values in {0, 1}.
            partners: shape (N,), partner[i] != i.

        Returns:
            rewards: shape (N,), float32
            interaction_counts: shape (N,), int32
        """
        selector_rewards = self.payoff[actions, actions[partners]]
        selected_rewards = np.zeros_like(selector_rewards, dtype=np.float32)
        np.add.at(selected_rewards, partners, self.payoff[actions[partners], actions])

        rewards = selector_rewards.astype(np.float32) + selected_rewards
        interaction_counts = np.ones((actions.shape[0],), dtype=np.int32)
        np.add.at(interaction_counts, partners, 1)
        return rewards, interaction_counts
