"""A node-placement stage followed by common constrained link mapping."""

import numpy as np

from ..sub_rl_environment import PlaceStepSubRLEnv


class TwoStageSubEnv(PlaceStepSubRLEnv):
    """Two-stage environment used by the Chen et al.-inspired baseline.

    Stage 1 selects all physical hosts using the GCN-A2C policy.  Stage 2 is
    performed exactly once by the shared Controller after node placement, so
    every method uses the same bandwidth and static link-security checks.
    """

    def compute_reward(self, solution):
        if solution['result']:
            reward = solution['v_net_r2c_ratio']
        elif solution['place_result'] and solution['route_result']:
            reward = 0.0
        else:
            reward = -1.0 / self.v_net.num_nodes
        self.solution['v_net_reward'] += reward
        return reward

    def get_observation(self):
        p_net_obs = self._get_p_net_obs()
        v_net_obs = self._get_v_net_obs()
        return {
            'p_net_x': p_net_obs['x'],
            'p_net_edge_index': p_net_obs['edge_index'],
            'v_net_x': v_net_obs['x'],
        }

    def _get_p_net_obs(self):
        node_data = self.obs_handler.get_node_attrs_obs(
            self.p_net,
            node_attr_types=['resource', 'extrema', 'security'],
            node_attr_benchmarks=self.node_attr_benchmarks,
        )
        edge_aggr_data = self.obs_handler.get_link_aggr_attrs_obs(
            self.p_net,
            link_attr_types=['resource', 'extrema', 'security'],
            aggr='sum',
            link_sum_attr_benchmarks=self.link_sum_attr_benchmarks,
        )
        selected = np.zeros(self.p_net.num_nodes, dtype=np.float32)
        selected[self.selected_p_net_nodes] = 1.0
        x = np.concatenate((node_data, edge_aggr_data, np.expand_dims(selected, -1)), axis=-1)
        return {'x': x, 'edge_index': self.obs_handler.get_link_index_obs(self.p_net)}

    def _get_v_net_obs(self):
        if self.curr_v_node_id >= self.v_net.num_nodes:
            return {'x': np.empty((0, 0), dtype=np.float32)}
        node_data = self.obs_handler.get_node_attrs_obs(
            self.v_net,
            node_attr_types=['resource', 'security'],
            node_attr_benchmarks=self.node_attr_benchmarks,
        )
        edge_aggr_data = self.obs_handler.get_link_aggr_attrs_obs(
            self.v_net,
            link_attr_types=['resource', 'security'],
            aggr='sum',
            link_sum_attr_benchmarks=self.link_sum_attr_benchmarks,
        )
        return {'x': np.concatenate((node_data, edge_aggr_data), axis=-1)}
