import unittest

import numpy as np

from PrisonersDilemma.population_prisoners_dilemma_env import PopulationPrisonersDilemmaEnv


class TestPopulationActionCoding(unittest.TestCase):
    def test_action_space_has_partner_and_pd_heads(self):
        env = PopulationPrisonersDilemmaEnv(num_agents=4, max_steps=3, history_h=1, seed=0)
        self.assertEqual(env.action_space.__class__.__name__, "Tuple")
        self.assertEqual(env.action_space[0].n, 3)
        self.assertEqual(env.action_space[1].n, 2)

    def test_round_starts_with_selection_and_first_dilemma_interaction(self):
        env = PopulationPrisonersDilemmaEnv(num_agents=3, max_steps=3, history_h=2, seed=0)
        env.reset()

        np.testing.assert_array_equal(env._last_actions, np.array([-1, -1, -1], dtype=np.int8))
        np.testing.assert_array_equal(env._action_history, np.full((3, 2), -1, dtype=np.int8))

        # Round partners: [1,2,1]
        selection_actions = [(0, 0), (1, 1), (1, 0)]
        _, rewards_s, terminations_s, truncations_s, infos_s = env.step(selection_actions)

        # Round-start step resolves partners and executes selector 0 immediately.
        np.testing.assert_array_equal(
            np.asarray(rewards_s, dtype=np.float32),
            np.array([0.0, 5.0, 0.0], dtype=np.float32),
        )
        self.assertEqual(terminations_s, [False, False, False])
        self.assertEqual(truncations_s, [False, False, False])
        self.assertEqual([int(info["selected_partner"]) for info in infos_s], [1, 2, 1])

        # Dilemma substep 2: selector 1 vs selected 2
        # a1=C, a2=D -> [0, 0, 5]
        _, rewards_d2, _, _, _ = env.step([(0, 0), (0, 0), (0, 1)])
        np.testing.assert_array_equal(np.asarray(rewards_d2, dtype=np.float32), np.array([0.0, 0.0, 5.0], dtype=np.float32))

        # Dilemma substep 3: selector 2 vs selected 1
        # a2=D, a1=C -> [0, 0, 5]
        # selected agent 1 uses D against selector 0 and C against selector 2.
        _, rewards_d3, terminations_d3, truncations_d3, infos_d3 = env.step([(0, 0), (0, 0), (0, 1)])
        np.testing.assert_array_equal(np.asarray(rewards_d3, dtype=np.float32), np.array([0.0, 0.0, 5.0], dtype=np.float32))
        self.assertEqual(terminations_d3, [False, False, False])
        self.assertEqual(truncations_d3, [False, False, False])
        self.assertEqual(int(infos_d3[1]["played_partner"]), 2)

        # Selector-side actions are committed after the round ends.
        np.testing.assert_array_equal(env._last_actions, np.array([0, 0, 1], dtype=np.int8))
        np.testing.assert_array_equal(env._action_history[:, 0], np.array([0, 0, 1], dtype=np.int8))

    def test_dilemma_observes_only_currently_interacting_pair(self):
        env = PopulationPrisonersDilemmaEnv(num_agents=3, max_steps=3, history_h=1, seed=0)
        obs, _ = env.reset()
        self.assertEqual(obs[0]["obs"].shape[0], 5)
        np.testing.assert_array_equal(obs[0]["obs"], np.zeros((5,), dtype=np.float32))

        # Round-start: partners [1,2,1], and selector 0 interaction is executed immediately.
        obs_d, _, _, _, _ = env.step([{"partner": 0, "pd": 0}, {"partner": 1, "pd": 1}, {"partner": 1, "pd": 0}])
        self.assertEqual(float(obs_d[0]["obs"][-1]), 1.0)
        self.assertEqual(float(obs_d[1]["obs"][-1]), 1.0)
        self.assertEqual(float(obs_d[2]["obs"][-1]), 1.0)

        # Next active pair is (1,2), so agent 0 is non-participant and sees zeros.
        np.testing.assert_array_equal(obs_d[0]["obs"][:4], np.zeros((4,), dtype=np.float32))

        # Finish round with selector actions [C,D,C].
        env.step([{"partner": 0, "pd": 0}, {"partner": 0, "pd": 1}, {"partner": 0, "pd": 0}])
        obs_s, _, _, _, _ = env.step([{"partner": 0, "pd": 0}, {"partner": 0, "pd": 1}, {"partner": 0, "pd": 0}])

        self.assertEqual(float(obs_s[0]["obs"][-1]), 0.0)
        # Agent0 sees others [1,2] => [D,C]
        np.testing.assert_array_equal(obs_s[0]["obs"][:4], np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32))

    def test_random_matching_ignores_partner_head(self):
        env = PopulationPrisonersDilemmaEnv(num_agents=4, max_steps=2, history_h=1, seed=0, partner_scheduler="random")
        env._sample_random_partners = lambda: np.asarray([1, 0, 1, 1], dtype=np.int32)

        env.reset(seed=7)
        _, _, _, _, infos = env.step([(2, 0), (2, 0), (0, 0), (0, 0)])

        self.assertEqual([int(info["selected_partner"]) for info in infos], [1, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
