import numpy as np
import networkx as nx
from ..spaces import spaces
from ..sub_rl_environment import JointPRStepSubRLEnv, PlaceStepSubRLEnv


class SubEnv(JointPRStepSubRLEnv):

    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        super(SubEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)

        num_p_net_node_attrs = len(self.p_net.get_node_attrs(['resource', 'extrema','security']))
        num_p_net_link_attrs = len(self.p_net.get_link_attrs(['resource', 'extrema','security']))

    def compute_reward(self, solution):
        """ 根据采取行动的结果计算应得的奖励 """
        reward = self._calculate_reward(solution)
        return reward

    def _calculate_reward(self, solution):
        revenue = solution['v_net_time_revenue']  # 总收益
        cost = solution['v_net_time_cost']  if solution['v_net_time_cost'] != 0 else 1  # 总成本
        ts = solution['TS']  # 安全性能
        success = solution['result']
        # ====================== 动态权重计算 ======================
        ssi = solution['ssi']
        K = 2
        SSI_THRESHOLD = 4 # ssi期望为3.5
        omega = 1 / (1 + np.exp(-K * (ssi - SSI_THRESHOLD)))  # 式17
        phi = 1 - omega
        REWARD_COEFF = 0.9
        # ====================== 最终奖励计算 ======================
        if solution['result']:
            # 成功奖励（式16上半部分）
            reward = omega * ts + phi * (revenue / cost)
        elif solution['place_result'] and solution['route_result']:
            reward = omega * ts + phi * (revenue / cost)
        else:
            # 失败惩罚（式16下半部分）
            penalty = omega * ts + phi * (revenue / cost)
            reward = -REWARD_COEFF * penalty
        self.solution['v_net_reward'] = reward
        return reward

    def get_observation(self):
        p_net_obs = self._get_p_net_obs()
        v_net_obs = self._get_v_net_obs()
        # sub - attr
        return {'p_net_x': p_net_obs['x'],  # x 即连接处理整合后的100个节点数据 [100, 5+3+1]
                'p_net_edge_index': p_net_obs['edge_index'],
                'p_net_edge_attr_data' : p_net_obs['edge_attr_data'],
                'v_net_x': v_net_obs['x']}

    def _get_p_net_obs(self):
        """
            节点资源，平均距离，节点度数，物理节点状态，虚拟节点特征
        """
        # node data : [100, 5]  cpu max_cpu mem max_mem security   得到节点归一化数据
        node_data = self.obs_handler.get_node_attrs_obs(self.p_net, node_attr_types=['resource', 'extrema', 'security'], node_attr_benchmarks=self.node_attr_benchmarks)
        # edge_aggr_data : [100, 3]  bw max_bw security
        # edge_aggr_data = self.obs_handler.get_link_aggr_attrs_obs(self.p_net, link_attr_types=['resource', 'extrema', 'security'], aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)

        selected_nodes = np.zeros(self.p_net.num_nodes, dtype=np.float32)
        selected_nodes[self.selected_p_net_nodes] = 1.  # 标记已经有VNF部署的物理节点
        #                                                             np.expand_dims(selected_nodes, axis=-1).shape : [100,1]
        # node_data = np.concatenate((node_data, edge_aggr_data, np.expand_dims(selected_nodes, axis=-1)), axis=-1)
        node_data = np.concatenate((node_data, np.expand_dims(selected_nodes, axis=-1)), axis=-1)
        # edge_index
        edge_index = self.obs_handler.get_link_index_obs(self.p_net)
        edge_aggr_data = self.obs_handler.get_edge_aggr_attrs_obs(self.p_net,
                                                                  edge_index = edge_index,
                                                                  link_attr_types=['resource', 'extrema', 'security'],
                                                                  aggr='max',
                                                                  link_attr_benchmarks=self.link_attr_benchmarks)
        # data
        p_net_obs = {
            'x': node_data,
            'edge_index': edge_index,
            'edge_attr_data': edge_aggr_data,
        }
        return p_net_obs

    def _get_v_net_obs(self):
        if self.curr_v_node_id  >= self.v_net.num_nodes:
            return []
        # self.link_sum_attr_benchmarks : 物理网络链路属性的最大值
        # node data : [num, 3]  cpu mem security   得到节点归一化数据
        node_data = self.obs_handler.get_node_attrs_obs(self.v_net, node_attr_types=['resource','security'], node_attr_benchmarks=self.node_attr_benchmarks)
        # edge_aggr_data : [num, 2]  bw security
        edge_aggr_data = self.obs_handler.get_link_aggr_attrs_obs(self.v_net, link_attr_types=['resource','security'], aggr='sum', link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
        node_data = np.concatenate((node_data, edge_aggr_data), axis=-1)
        return {'x': node_data}
