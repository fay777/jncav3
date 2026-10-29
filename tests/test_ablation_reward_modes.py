import unittest

from solver.learning.gat_seq2seq_ppo.sub_env import SubEnv


class AblationRewardModeTests(unittest.TestCase):

    def make_env(self, *, incremental=False, terminal=False, weight=0.5):
        env = SubEnv.__new__(SubEnv)
        env.normalize_ts = True
        env.incremental_security_shaping = incremental
        env.terminal_only_security_reward = terminal
        env.reward_failure_penalty = 1.0
        env.security_reward_weight = weight
        env.previous_security_score = 0.0
        env.solution = {}
        return env

    @staticmethod
    def solution(*, result=False, place=True, route=True, score=0.0):
        return {
            'v_net_time_revenue': 2.0,
            'v_net_time_cost': 4.0,
            'result': result,
            'place_result': place,
            'route_result': route,
            'ssi': 4.0,
            'score': score,
        }

    def test_terminal_only_ablation_pays_nothing_before_completion(self):
        env = self.make_env(terminal=True)
        env._security_score = lambda solution: solution['score']

        partial_reward = env._calculate_reward(self.solution(score=0.4))
        terminal_reward = env._calculate_reward(self.solution(result=True, score=0.8))

        self.assertEqual(partial_reward, 0.0)
        self.assertAlmostEqual(terminal_reward, 0.65)
        self.assertEqual(env.previous_security_score, 0.0)

    def test_incremental_and_terminal_only_have_same_success_return(self):
        incremental = self.make_env(incremental=True)
        incremental._security_score = lambda solution: solution['score']
        terminal = self.make_env(terminal=True)
        terminal._security_score = lambda solution: solution['score']

        incremental_return = (
            incremental._calculate_reward(self.solution(score=0.4))
            + incremental._calculate_reward(self.solution(result=True, score=0.8))
        )
        terminal_return = (
            terminal._calculate_reward(self.solution(score=0.4))
            + terminal._calculate_reward(self.solution(result=True, score=0.8))
        )

        self.assertAlmostEqual(incremental_return, 0.65)
        self.assertAlmostEqual(terminal_return, 0.65)

    def test_weight_extremes_remove_one_reward_component(self):
        no_security = self.make_env(incremental=True, weight=0.0)
        no_security._security_score = lambda solution: solution['score']
        no_efficiency = self.make_env(incremental=True, weight=1.0)
        no_efficiency._security_score = lambda solution: solution['score']

        no_security_return = (
            no_security._calculate_reward(self.solution(score=0.4))
            + no_security._calculate_reward(self.solution(result=True, score=0.8))
        )
        no_efficiency_return = (
            no_efficiency._calculate_reward(self.solution(score=0.4))
            + no_efficiency._calculate_reward(self.solution(result=True, score=0.8))
        )

        self.assertAlmostEqual(no_security_return, 0.5)
        self.assertAlmostEqual(no_efficiency_return, 0.8)

    def test_terminal_only_failure_is_fixed_penalty(self):
        env = self.make_env(terminal=True)
        env._security_score = lambda solution: solution['score']

        reward = env._calculate_reward(
            self.solution(place=False, route=False, score=0.3)
        )

        self.assertEqual(reward, -1.0)

    def test_incremental_and_terminal_only_have_same_failure_return(self):
        incremental = self.make_env(incremental=True)
        incremental._security_score = lambda solution: solution['score']
        terminal = self.make_env(terminal=True)
        terminal._security_score = lambda solution: solution['score']

        incremental_return = (
            incremental._calculate_reward(self.solution(score=0.4))
            + incremental._calculate_reward(
                self.solution(place=False, route=False, score=0.4)
            )
        )
        terminal_return = (
            terminal._calculate_reward(self.solution(score=0.4))
            + terminal._calculate_reward(
                self.solution(place=False, route=False, score=0.4)
            )
        )

        self.assertAlmostEqual(incremental_return, -1.0)
        self.assertAlmostEqual(terminal_return, -1.0)


if __name__ == '__main__':
    unittest.main()
