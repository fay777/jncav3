import numpy as np
import networkx as nx
from ..spaces import spaces
from ..sub_rl_environment import JointPRStepSubRLEnv, PlaceStepSubRLEnv


class SubEnv(JointPRStepSubRLEnv):

    def __init__(self, p_net, v_net, controller, recorder, counter, **kwargs):
        super(SubEnv, self).__init__(p_net, v_net, controller, recorder, counter, **kwargs)

        # Kept switchable so the legacy experiment remains reproducible.  The
        # normalized formulation is used only when --normalize_ts true is
        # supplied to a new training run.
        self.normalize_ts = kwargs.get('normalize_ts', False)
        self.incremental_security_shaping = kwargs.get('incremental_security_shaping', False)
        # Optional ablation: expose the current, pre-feedback per-node trust
        # value to the policy.  The default keeps the reward-only trust design
        # and therefore preserves the former observation space exactly.
        self.include_trust_in_state = kwargs.get('include_trust_in_state', False)
        self.reward_failure_penalty = float(kwargs.get('reward_failure_penalty', 1.0))
        # A non-negative value freezes the security/efficiency trade-off;
        # -1 keeps the SSI-adaptive weighting used by legacy experiments.
        self.security_reward_weight = float(kwargs.get('security_reward_weight', -1.0))
        # q_0 = 0 for every SFC.  It is advanced only after a successful
        # partial placement, so terminal failure can cancel all prior shaping.
        self.previous_security_score = 0.0

        num_p_net_node_attrs = len(self.p_net.get_node_attrs(['resource', 'extrema','security']))
        num_p_net_link_attrs = len(self.p_net.get_link_attrs(['resource', 'extrema','security']))

    def compute_reward(self, solution):
        """ 根据采取行动的结果计算应得的奖励 """
        reward = self._calculate_reward(solution)
        return reward

    def _calculate_reward(self, solution):
        revenue = solution['v_net_time_revenue']  # 总收益
        cost = solution['v_net_time_cost']  if solution['v_net_time_cost'] != 0 else 1  # 总成本
        ts = self._security_score(solution)
        success = solution['result']
        # ====================== 动态权重计算 ======================
        ssi = solution['ssi']
        K = 2
        SSI_THRESHOLD = 4 # ssi期望为3.5
        omega = 1 / (1 + np.exp(-K * (ssi - SSI_THRESHOLD)))  # 式17
        phi = 1 - omega
        if self.security_reward_weight >= 0.0:
            omega = self.security_reward_weight
            phi = 1.0 - omega
        REWARD_COEFF = 0.9
        if self.incremental_security_shaping:
            if not self.normalize_ts:
                raise ValueError('incremental_security_shaping requires normalize_ts=true')

            # A completed solution is the only point at which the efficiency
            # term is paid.  For intermediate successes, pay merely the
            # security increment.  Hence a successful SFC receives omega*q_T
            # exactly once in total, rather than once per VNF.
            if success:
                reward = omega * (ts - self.previous_security_score) + phi * (revenue / cost)
                self.previous_security_score = ts
            elif solution['place_result'] and solution['route_result']:
                reward = omega * (ts - self.previous_security_score)
                self.previous_security_score = ts
            else:
                # Cancel previously accrued security shaping on a failed SFC.
                reward = -self.reward_failure_penalty - omega * self.previous_security_score
            self.solution['v_net_reward'] = reward
            return reward

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

    def _security_score(self, solution):
        """Return the legacy TS or a bounded mean security score.

        Trust is already in [0, 1].  A physical-link security level is divided
        by its configured maximum before aggregation.  Averaging over mapped
        VNFs and physical-link hops prevents longer chains from receiving a
        larger reward solely because they contain more components.
        """
        if not self.normalize_ts:
            return solution['TS']

        node_ids = list(solution['node_slots'].values())
        link_hops = [
            p_link
            for p_links in solution['link_paths'].values()
            for p_link in p_links
        ]
        component_count = len(node_ids) + len(link_hops)
        if component_count == 0:
            return 0.0

        link_max_security = max(
            (float(attr.high) for attr in self.controller.link_security_attrs),
            default=1.0,
        )
        node_score = sum(float(self.p_net.nodes[node_id].get('trust', 0.0))
                         for node_id in node_ids)
        link_score = sum(
            float(self.p_net.edges[p_link].get('security', 0.0)) / link_max_security
            for p_link in link_hops
        )
        return (node_score + link_score) / component_count

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
        # ``trust`` is already bounded in [0, 1] by PhysicalNetwork.  It is
        # appended only when requested so the reward-only baseline retains
        # precisely its original feature layout and checkpoint compatibility.
        state_features = [node_data]
        if self.include_trust_in_state:
            trust = np.asarray(
                [self.p_net.nodes[node_id].get('trust', 0.0)
                 for node_id in self.p_net.nodes],
                dtype=np.float32,
            ).reshape(-1, 1)
            trust = np.clip(trust, 0.0, 1.0)
            state_features.append(trust)
        state_features.append(np.expand_dims(selected_nodes, axis=-1))
        node_data = np.concatenate(state_features, axis=-1)
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
