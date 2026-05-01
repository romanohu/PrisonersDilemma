import unittest

import numpy as np

from PrisonersDilemma.paired_population_prisoners_dilemma_env import PairedPopulationPrisonersDilemmaEnv


class TestPairedPopulationPrisonersDilemmaEnv(unittest.TestCase):
    def test_action_space_is_always_partner_and_pd(self):
        env = PairedPopulationPrisonersDilemmaEnv(
            num_agents=4,
            pd_horizon=2,
            seed=0,
        )
        self.assertEqual(len(env.action_space.spaces), 2)

    def test_matching_step_is_zero_reward_and_transitions_to_pd(self):
        env = PairedPopulationPrisonersDilemmaEnv(
            num_agents=3,
            pd_horizon=2,
            seed=0,
        )
        obs, infos = env.reset()

        self.assertEqual(infos[0]["phase"], env.PHASE_MATCHING)
        np.testing.assert_array_equal(obs[0]["pd_obs"], np.array([0.0, 0.0], dtype=np.float32))

        # Absolute partner map [1, 2, 0] encoded as relative ids per agent.
        # i=0 -> abs1 -> rel0, i=1 -> abs2 -> rel1, i=2 -> abs0 -> rel0
        actions = [(0, 0), (1, 1), (0, 1)]
        obs_pd, rewards, terminations, truncations, infos_pd = env.step(actions)

        np.testing.assert_array_equal(np.asarray(rewards, dtype=np.float32), np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self.assertEqual(terminations, [False, False, False])
        self.assertEqual(truncations, [False, False, False])
        self.assertEqual([int(info["selected_partner"]) for info in infos_pd], [1, 2, 0])
        self.assertEqual([bool(info["can_act"]) for info in infos_pd], [True, True, False])
        self.assertEqual([bool(info["new_match"]) for info in infos_pd], [True, True, False])
        self.assertEqual([int(info["active_opponent_id"]) for info in infos_pd], [1, 0, -1])
        self.assertEqual([int(info["phase"]) for info in infos_pd], [env.PHASE_PD, env.PHASE_PD, env.PHASE_PD])
        np.testing.assert_array_equal(obs_pd[0]["selection_obs"], np.zeros((2, 2), dtype=np.float32))
        np.testing.assert_array_equal(obs_pd[0]["pd_obs"], np.array([0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(obs_pd[1]["pd_obs"], np.array([0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(obs_pd[2]["pd_obs"], np.array([0.0, 0.0], dtype=np.float32))

    def test_first_pd_observation_is_always_zero_initial_observation(self):
        env = PairedPopulationPrisonersDilemmaEnv(
            num_agents=2,
            pd_horizon=1,
            seed=0,
        )
        env.reset()

        obs_pd, rewards_matching, _, _, infos_pd = env.step([(0, 1), (0, 1)])
        np.testing.assert_array_equal(
            np.asarray(rewards_matching, dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)
        )
        self.assertEqual([int(info["selected_partner"]) for info in infos_pd], [1, 0])
        np.testing.assert_array_equal(obs_pd[0]["pd_obs"], np.array([0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(obs_pd[1]["pd_obs"], np.array([0.0, 0.0], dtype=np.float32))

        _, rewards_pd, terminations_pd, truncations_pd, _ = env.step([(0, 0), (0, 0)])
        np.testing.assert_array_equal(np.asarray(rewards_pd, dtype=np.float32), np.array([3.0, 3.0], dtype=np.float32))
        self.assertEqual(terminations_pd, [False, False])
        self.assertEqual(truncations_pd, [False, False])

    def test_parallel_slot_executes_multiple_matches_in_one_step(self):
        env = PairedPopulationPrisonersDilemmaEnv(
            num_agents=4,
            pd_horizon=1,
            seed=0,
        )
        env.reset()

        # Absolute partners [1,2,3,0], encoded as relative ids [0,1,2,0].
        obs_pd, _, _, _, infos_pd = env.step([(0, 0), (1, 0), (2, 0), (0, 0)])
        self.assertEqual([bool(info["can_act"]) for info in infos_pd], [True, True, True, True])
        self.assertEqual([int(info["active_opponent_id"]) for info in infos_pd], [1, 0, 3, 2])

        _, rewards_pd, terminations_pd, truncations_pd, infos_next = env.step([(0, 0), (0, 0), (0, 0), (0, 0)])
        np.testing.assert_array_equal(np.asarray(rewards_pd, dtype=np.float32), np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float32))
        self.assertEqual(terminations_pd, [False, False, False, False])
        self.assertEqual(truncations_pd, [False, False, False, False])
        # Slot 0 completed, slot 1 starts next and is a new local match for all agents in this topology.
        self.assertEqual([bool(info["new_match"]) for info in infos_next], [True, True, True, True])
        self.assertEqual([int(info["active_opponent_id"]) for info in infos_next], [3, 2, 1, 0])

    def test_reset_population_stats_clears_ema_without_recreating_env(self):
        env = PairedPopulationPrisonersDilemmaEnv(
            num_agents=2,
            pd_horizon=1,
            ema_alpha=1.0,
            own_reward_prior=-1.0,
            partner_reward_prior=-2.0,
            seed=0,
        )
        obs0, _ = env.reset()
        np.testing.assert_array_equal(obs0[0]["selection_obs"], np.array([[-1.0, -2.0]], dtype=np.float32))

        env.step([(0, 0), (0, 0)])  # matching
        env.step([(0, 0), (0, 0)])  # slot0 PD
        _, _, terminations_final, _, infos_final = env.step([(0, 0), (0, 0)])  # slot1 PD -> done
        self.assertEqual(terminations_final, [True, True])
        self.assertAlmostEqual(float(infos_final[0]["true_objective"]), 6.0)
        self.assertAlmostEqual(float(infos_final[1]["true_objective"]), 6.0)

        obs1, _ = env.reset()
        np.testing.assert_array_equal(obs1[0]["selection_obs"], np.array([[6.0, 3.0]], dtype=np.float32))
        np.testing.assert_array_equal(obs1[1]["selection_obs"], np.array([[6.0, 3.0]], dtype=np.float32))

        env.reset_population_stats()
        obs2, _ = env.reset()
        np.testing.assert_array_equal(obs2[0]["selection_obs"], np.array([[-1.0, -2.0]], dtype=np.float32))
        np.testing.assert_array_equal(obs2[1]["selection_obs"], np.array([[-1.0, -2.0]], dtype=np.float32))

    def test_reset_is_required_after_termination(self):
        env = PairedPopulationPrisonersDilemmaEnv(
            num_agents=2,
            pd_horizon=1,
            seed=0,
        )
        env.reset()

        env.step([(0, 0), (0, 0)])  # matching
        env.step([(0, 0), (0, 0)])  # slot0 PD
        _, _, terminations, _, _ = env.step([(0, 0), (0, 0)])  # slot1 PD -> done
        self.assertEqual(terminations, [True, True])

        with self.assertRaises(RuntimeError):
            env.step([(0, 0), (0, 0)])


if __name__ == "__main__":
    unittest.main()
