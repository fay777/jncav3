"""Fast invariants for the paper-consistent trust/SRSR accounting module.

Run from the project root with ``python verify_security_accounting.py``.
It does not train a model or modify datasets.
"""

from types import SimpleNamespace

import pandas as pd

from base.counter import Counter
from base.solution import Solution
from data.physical_network import PhysicalNetwork


def make_solution(v_net):
    solution = Solution(v_net)
    solution.result = True
    solution.node_slots[0] = 0
    solution.node_slots[1] = 1
    solution.link_paths[(0, 1)] = [(0, 1)]
    return solution


def main():
    p_net = PhysicalNetwork()
    p_net.add_node(0, security=6)
    p_net.add_node(1, security=4)
    p_net.add_edge(0, 1, security=3)
    p_net.init_network_trust(0.6, 0.4, 6.0)

    # At cold start: trust(0) = .6 * 1 + .4 * .5 = .8.
    assert abs(p_net.nodes[0]['trust'] - 0.8) < 1e-12
    assert all(0.0 <= p_net.nodes[n]['trust'] <= 1.0 for n in p_net.nodes)

    v_net = SimpleNamespace(
        id=0,
        lifetime=1,
        arrival_time=0,
        num_nodes=2,
        num_links=1,
        ssi=4.5,
        nodes={0: {'security': 5}, 1: {'security': 4}},
        links={(0, 1): {'security': 3}},
    )
    solution = make_solution(v_net)
    counter = Counter(security_ssi_threshold=4.0, security_max_level=6.0)

    snapshot = counter.evaluate_security_requirements(v_net, p_net, solution)
    # Node 0 needs 5/6 trust but begins at .8, so SRSR is non-trivial.
    assert snapshot['security_sensitive']
    assert not snapshot['node_security_satisfied']
    assert snapshot['link_security_satisfied']

    # The post-request feedback changes future trust only; it must not change
    # the already stored SRSR snapshot for this request.
    solution.update(**snapshot)
    p_net.record_trust_attempt(solution.node_slots.values(), succeeded=True)
    assert p_net.nodes[0]['trust'] > 0.8
    assert solution.security_requirements_satisfied == snapshot['security_requirements_satisfied']

    # A failed selected-node attempt lowers feedback without division by zero.
    trust_before_failure = p_net.nodes[1]['trust']
    p_net.record_trust_attempt([1], succeeded=False)
    assert p_net.nodes[1]['trust'] < trust_before_failure
    assert all(0.0 <= p_net.nodes[n]['trust'] <= 1.0 for n in p_net.nodes)

    # SRSR denominator is deployed sensitive requests, not all sensitive
    # arrivals: the second sensitive request below was rejected.
    records = pd.DataFrame([
        {
            'event_type': 1, 'early_rejection': False, 'place_result': True,
            'route_result': True, 'security_sensitive': True, 'result': True,
            'security_requirements_satisfied': False, 'success_count': 1,
            'v_net_count': 2, 'total_revenue': 10, 'total_cost': 5,
            'total_time_revenue': 10, 'total_time_cost': 5,
            'v_net_arrival_time': 1, 'p_net_available_resource': 10,
            'p_net_node_available_resource': 5,
            'p_net_link_available_resource': 5, 'inservice_count': 1,
        },
        {
            'event_type': 1, 'early_rejection': False, 'place_result': False,
            'route_result': True, 'security_sensitive': True, 'result': False,
            'security_requirements_satisfied': False, 'success_count': 1,
            'v_net_count': 2, 'total_revenue': 10, 'total_cost': 5,
            'total_time_revenue': 10, 'total_time_cost': 5,
            'v_net_arrival_time': 2, 'p_net_available_resource': 10,
            'p_net_node_available_resource': 5,
            'p_net_link_available_resource': 5, 'inservice_count': 1,
        },
    ])
    summary = counter.summary_records(records)
    assert summary['security_sensitive_request_count'] == 1
    assert summary['security_sensitive_success_count'] == 0
    assert summary['srsr'] == 0.0
    print('Security-accounting invariants passed.')


if __name__ == '__main__':
    main()
