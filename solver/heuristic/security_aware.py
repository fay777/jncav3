"""Security-aware heuristic baselines adapted to the common JNCA setting.

The two cited works model capabilities that are deliberately absent from this
repository: multi-version security functions, multi-vendor selection, and (in
APPF-DSI) disjoint primary/backup protection.  These implementations therefore
retain only their *deployment principles* under the shared fixed-chain model.
They are not claimed to be verbatim reproductions of either published method.
"""

from base import Solution, SolutionStepEnvironment
from solver import registry
from ..solver import Solver


class _SecurityAwareSequentialSolver(Solver):
    """Shared feasible sequential mapper for security-aware adapted baselines."""

    def _resource_pressure(self, p_net, p_node_id, v_net, v_node_id):
        pressures = []
        for attr in self.controller.node_resource_attrs:
            available = float(p_net.nodes[p_node_id][attr.name])
            demand = float(v_net.nodes[v_node_id][attr.name])
            pressures.append(demand / max(available, 1.0))
        return sum(pressures) / len(pressures) if pressures else 0.0

    def _candidate_score(self, p_net, p_node_id, v_net, v_node_id):
        raise NotImplementedError

    def solve(self, instance):
        v_net, p_net = instance['v_net'], instance['p_net']
        solution = Solution(v_net)

        # The service order is preserved because the request topology is a
        # path.  Static CPU, memory, node-security, bandwidth and link-security
        # feasibility is always enforced by the common Controller.
        for v_node_id in list(v_net.nodes):
            selected = list(solution['node_slots'].values())
            candidates = self.controller.find_candidate_nodes(
                v_net, p_net, v_node_id, filter=selected
            )
            if not candidates:
                solution['place_result'] = False
                return solution

            candidates.sort(
                key=lambda p_node_id: self._candidate_score(
                    p_net, p_node_id, v_net, v_node_id
                )
            )

            deployed = False
            for p_node_id in candidates:
                ok, _ = self.controller.place_and_route(
                    v_net,
                    p_net,
                    v_node_id,
                    p_node_id,
                    solution,
                    shortest_method=self.shortest_method,
                    k=self.k_shortest,
                )
                if ok:
                    deployed = True
                    break

                # ``place_and_route`` tentatively reserves node/link resources
                # before it discovers that a route is infeasible.  A failed
                # candidate must therefore be completely removed before trying
                # the next one.  Besides preventing phantom resource
                # consumption, this also clears the failure flags left by the
                # tentative route attempt.
                if v_node_id in solution['node_slots']:
                    self.controller.undo_place_and_route(
                        v_net, p_net, v_node_id, p_node_id, solution
                    )
                solution['place_result'] = True
                solution['route_result'] = True

            if not deployed:
                solution['route_result'] = False
                return solution

        # A request is successful only when every VNF has been placed and all
        # links have been routed.  Explicitly restore every status field so a
        # transient failure from a discarded candidate cannot pollute summary
        # metrics or security accounting.
        solution['result'] = True
        solution['place_result'] = True
        solution['route_result'] = True
        return solution


@registry.register(
    solver_name='appf_dsi_adapted',
    env_cls=SolutionStepEnvironment,
    solver_type='heuristic',
)
class APPFDSIAdaptedSolver(_SecurityAwareSequentialSolver):
    """Wang et al. (2024)-inspired security-capability greedy baseline.

    Adaptation scope: the original APPF-DSI solves multi-version SFC embedding
    with asymmetric dedicated protection.  Here each VNF has one fixed type and
    there is no backup-chain representation.  The retained principle is to
    favour a security-capable, lightly loaded primary placement before routing
    with the common feasible-link controller.
    """

    def _candidate_score(self, p_net, p_node_id, v_net, v_node_id):
        offered = float(p_net.nodes[p_node_id]['security'])
        max_level = max(
            float(attr.high) for attr in self.controller.node_security_attrs
        )
        pressure = self._resource_pressure(p_net, p_node_id, v_net, v_node_id)
        # APPF-DSI-inspired primary placement is capability-first: among
        # already feasible nodes, preserve a preference for higher security
        # capability and use resource pressure only as a secondary criterion.
        # The common mapper sorts ascending, hence the negative capability.
        capability = offered / max(max_level, 1.0)
        return (-capability, pressure, p_node_id)


@registry.register(
    solver_name='ssd_ose_adapted',
    env_cls=SolutionStepEnvironment,
    solver_type='heuristic',
)
class SSDOSEAdaptedSolver(_SecurityAwareSequentialSolver):
    """Wang et al. (2025)-inspired security-cost-balance baseline.

    Adaptation scope: the original SSD-OSE selects among multi-vendor,
    multi-version security functions and estimates chain-level security.  The
    common simulator fixes VNF types and has no vendor/function-version state.
    This baseline therefore implements its security-cost-balance and
    overflowing-security-elimination principles as a node-ranking rule: choose
    the least over-provisioned feasible security level, then minimise resource
    pressure.  It is reported as ``SSD-OSE-adapted``, never as SSD-OSE itself.
    """

    def _candidate_score(self, p_net, p_node_id, v_net, v_node_id):
        offered = float(p_net.nodes[p_node_id]['security'])
        required = float(v_net.nodes[v_node_id]['security'])
        max_level = max(
            float(attr.high) for attr in self.controller.node_security_attrs
        )
        overflow = (offered - required) / max(max_level, 1.0)
        pressure = self._resource_pressure(p_net, p_node_id, v_net, v_node_id)
        # SSD-OSE-inspired placement explicitly eliminates excess security
        # capability before considering residual-resource pressure.
        return (overflow, pressure, p_node_id)
