from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class InteractionScheduler(Protocol):
    """Select one partner per agent for the next round."""

    def select_partners(
        self,
        *,
        num_agents: int,
        rng: np.random.Generator,
        action_history: np.ndarray,
        step: int,
    ) -> np.ndarray:
        """Return int32 array of shape (num_agents,), partner[i] != i."""


@dataclass
class RandomPartnerScheduler:
    """Uniform random partner selection with replacement."""

    def select_partners(
        self,
        *,
        num_agents: int,
        rng: np.random.Generator,
        action_history: np.ndarray,
        step: int,
    ) -> np.ndarray:
        del action_history, step
        draws = rng.integers(0, num_agents - 1, size=num_agents, dtype=np.int32)
        agent_ids = np.arange(num_agents, dtype=np.int32)
        draws += (draws >= agent_ids).astype(np.int32)
        return draws.astype(np.int32)
