import unittest

from base.counter import Counter


class _Network:
    def __init__(self, nodes, links, ssi=0.0):
        self.nodes = nodes
        self.links = links
        self.edges = links
        self.num_nodes = len(nodes)
        self.num_links = len(links)
        self.ssi = ssi


class SecurityMetricTests(unittest.TestCase):
    def setUp(self):
        self.counter = Counter.__new__(Counter)
        self.counter.security_ssi_threshold = 4.0
        self.counter.security_max_level = 6.0
        self.counter.lcb_z_score = 1.645

    def test_wilson_lcb_is_zero_without_history(self):
        self.assertEqual(self.counter._wilson_lower_bound(0, 0), 0.0)

    def test_security_snapshot_uses_static_constraints_and_pre_feedback_lcb(self):
        v_net = _Network(
            {0: {'security': 5}},
            {(0, 0): {'security': 2}},
            ssi=5.0,
        )
        p_net = _Network(
            {10: {
                'security': 5,
                'trust': 0.9,
                'trust_success_count': 1000,
                'trust_attempt_count': 1000,
            }},
            {(10, 10): {'security': 2}},
        )
        solution = {
            'result': True,
            'node_slots': {0: 10},
            'link_paths': {(0, 0): [(10, 10)]},
        }

        result = self.counter.evaluate_security_requirements(v_net, p_net, solution)

        self.assertTrue(result['security_sensitive'])
        self.assertTrue(result['static_security_satisfied'])
        self.assertGreaterEqual(result['min_node_reliability_lcb'], 0.8)
        self.assertTrue(result['lcb_0_7_satisfied'])
        self.assertTrue(result['lcb_0_8_satisfied'])
        self.assertTrue(result['lcb_0_9_satisfied'])
        # Evaluation is a snapshot and must not update feedback counts.
        self.assertEqual(p_net.nodes[10]['trust_attempt_count'], 1000)
        self.assertEqual(p_net.nodes[10]['trust_success_count'], 1000)

    def test_static_violation_fails_all_revised_security_metrics(self):
        v_net = _Network({0: {'security': 6}}, {}, ssi=6.0)
        p_net = _Network({10: {
            'security': 5,
            'trust': 1.0,
            'trust_success_count': 100,
            'trust_attempt_count': 100,
        }}, {})
        solution = {'result': True, 'node_slots': {0: 10}, 'link_paths': {}}

        result = self.counter.evaluate_security_requirements(v_net, p_net, solution)

        self.assertFalse(result['static_security_satisfied'])
        self.assertFalse(result['lcb_0_7_satisfied'])
        self.assertFalse(result['lcb_0_8_satisfied'])
        self.assertFalse(result['lcb_0_9_satisfied'])

    def test_summary_reports_conditional_and_arrival_based_security_rates(self):
        record = {
            'event_type': 1,
            'success_count': 1,
            'v_net_count': 1,
            'early_rejection': False,
            'place_result': True,
            'route_result': True,
            'security_sensitive': True,
            'result': True,
            'security_requirements_satisfied': False,
            'static_security_satisfied': True,
            'lcb_0_7_satisfied': True,
            'lcb_0_8_satisfied': False,
            'lcb_0_9_satisfied': False,
            'min_node_reliability_lcb': 0.75,
            'total_revenue': 10.0,
            'total_cost': 5.0,
            'total_time_revenue': 100.0,
            'total_time_cost': 50.0,
            'v_net_arrival_time': 10.0,
            'p_net_available_resource': 100.0,
            'p_net_node_available_resource': 60.0,
            'p_net_link_available_resource': 40.0,
            'inservice_count': 1,
        }

        summary = self.counter.summary_records([record])

        self.assertEqual(summary['security_sensitive_acceptance_rate'], 1.0)
        self.assertEqual(summary['static_security_satisfaction_rate'], 1.0)
        self.assertEqual(summary['lcb_0_7_rate'], 1.0)
        self.assertEqual(summary['lcb_0_8_rate'], 0.0)
        self.assertEqual(summary['lcb_0_9_rate'], 0.0)
        self.assertEqual(summary['mean_min_node_reliability_lcb'], 0.75)
        self.assertEqual(summary['reliable_secure_deployment_rate_lcb_0_7'], 1.0)
        self.assertEqual(summary['reliable_secure_deployment_rate_lcb_0_8'], 0.0)
        self.assertEqual(summary['reliable_secure_deployment_rate_lcb_0_9'], 0.0)


if __name__ == '__main__':
    unittest.main()
