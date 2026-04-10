import unittest

import numpy as np

from PrisonersDilemma.prisoners_dilemma_env import PrisonersDilemmaEnv


class TestTwoAgentActionCoding(unittest.TestCase):
    def test_internal_history_uses_minus1_0_1_codes(self):
        env = PrisonersDilemmaEnv(num_agents=2, max_steps=3, history_h=2, seed=0)

        env.reset()

        np.testing.assert_array_equal(env._last_actions, np.array([-1, -1], dtype=np.int8))
        np.testing.assert_array_equal(env._action_history, np.full((2, 2), -1, dtype=np.int8))

        env.step([0, 1])

        np.testing.assert_array_equal(env._last_actions, np.array([0, 1], dtype=np.int8))
        np.testing.assert_array_equal(env._action_history[:, 0], np.array([0, 1], dtype=np.int8))


if __name__ == "__main__":
    unittest.main()
