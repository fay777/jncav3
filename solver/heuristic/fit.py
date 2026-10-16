import numpy as np
import networkx as nx

from base import SolutionStepEnvironment, Solution
from solver import registry
from ..solver import Solver

from ..learning.gat_seq2seq_ppo.sub_env import SubEnv
class FitSolver(Solver):

    name = 'first_fit'

    def __init__(self, controller, recorder, counter, **kwargs):
        super().__init__(controller, recorder, counter, **kwargs)

@registry.register(
    solver_name='first_fit',
    env_cls=SolutionStepEnvironment,
    solver_type='heuristic')
class FirstFitSolver(FitSolver):

    def __init__(self, controller, recorder, counter, **kwargs):
        super(FirstFitSolver, self).__init__(controller, recorder, counter, **kwargs)

    def solve(self, instance):
        # 处理一条SFC请求
        v_net, p_net  = instance['v_net'], instance['p_net']
        solution = Solution(v_net)
        for v_node_id in list(v_net.nodes):
            selected_p_net_nodes = list(solution['node_slots'].values())
            candidate_p_net_nodes = self.controller.find_candidate_nodes(v_net, p_net, v_node_id,
                                                                         filter=selected_p_net_nodes)
            if len(candidate_p_net_nodes) == 0:
                # Failure
                solution['place_result'] = False
                return solution
            p_node_id = candidate_p_net_nodes[0]
            place_and_route_result, place_and_route_info = self.controller.place_and_route(v_net, p_net, v_node_id,
                                                                                           p_node_id, solution,
                                                                                           shortest_method=self.shortest_method,
                                                                                           k=1)
            if not place_and_route_result:
                # Failure
                solution['route_result'] = False
                return solution
        # Success
        solution['result'] = True
        return solution

    def select_action(self, env, v_node_id):
        v_node_requests_dict = env.v_net.nodes[v_node_id]
        filter = env.selected_nodes if self.reusable == False else None
        candidate_nodes = env.find_candidate_nodes(v_node_requests_dict, filter=filter, rtype='id')
        if len(candidate_nodes) == 0:
            return 0
        else:
            return candidate_nodes[0]


    # def select_action(self, env, v_node_id=None):
    #     if v_node_id is None:
    #         v_node_id = env.curr_v_node_id
    #     v_node_requests_dict = env.v_net.nodes[v_node_id]
    #     filter = env.selected_nodes if self.reusable == False else None
    #     candidate_nodes = env.find_candidate_nodes(v_node_requests_dict, filter=filter, rtype='id')
    #     if len(candidate_nodes) == 0:
    #         return 0
    #     else:
    #         return candidate_nodes[0]


class RandomFitSolver(Solver):

    name = 'random_fit'

    def __init__(self, controller, recorder, counter, **kwargs):
        super(RandomFitSolver, self).__init__(controller, recorder, counter, **kwargs)

    def select_action(self, env, v_node_id=None):
        if v_node_id is None:
            v_node_id = env.curr_v_node_id
        v_node_requests_dict = env.v_net.nodes[v_node_id]
        filter = env.selected_nodes if self.reusable == False else None
        candidate_nodes = env.find_candidate_nodes(v_node_requests_dict, filter=filter, rtype='id')
        if len(candidate_nodes) == 0:
            return 0
        else:
            random_id = np.random.choice(len(candidate_nodes), 1)[0]
            p_node_id = candidate_nodes[random_id]
            return p_node_id


class NearestFitSolver(Solver):

    name = 'nearest_fit'

    def __init__(self, controller, recorder, counter, **kwargs):
        super(NearestFitSolver, self).__init__(controller, recorder, counter, **kwargs)

    def select_action(self, env, v_node_id=None):
        if v_node_id is None:
            v_node_id = env.curr_v_node_id
        v_node_requests_dict = env.v_net.nodes[v_node_id]
        filter = env.selected_nodes if self.reusable == False else None
        candidate_nodes = env.find_candidate_nodes(v_node_requests_dict, filter=filter, rtype='id')
        
        # first v_node
        if v_node_id == 0:
            if len(candidate_nodes) == 0:
                # FAILURE
                return env.p_net.num_nodes
            random_id = np.random.choice(len(candidate_nodes), 1)[0]
            p_node_id = candidate_nodes[random_id]
            return p_node_id

        # !!!
        temp_graph = self.p_net
        # look for the shortest path
        last_pid = env.selected_nodes[-1]
        try:
            path_length = nx.shortest_path_length(temp_graph, source=last_pid)
            sorted_nodes = [node for node, _ in sorted(path_length.items(), key=lambda item: item[1])]
            for node in sorted_nodes:
                if node in candidate_nodes:
                    return node
                else:
                    continue
            # FAILURE
            return env.p_net.num_nodes
        except:
            # FAILURE
            return env.p_net.num_nodes


if __name__ == '__main__':
    pass
