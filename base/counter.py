from typing import *
import numpy as np
import pandas as pd

from data import PhysicalNetwork, VirtualNetwork, Network
from data.attribute import create_attrs_from_setting
from .solution import Solution


class Counter(object):

    def __init__(self, node_attrs_setting: list = [], link_attrs_setting: list = [], **kwargs):
        self.node_resource_unit_price = kwargs.get('node_resource_unit_price', 1.) 
        self.link_resource_unit_price = kwargs.get('link_resource_unit_price', 1.)

        self.revenue_service_time_weight = kwargs.get('revenue_service_time_weight', 0.001) 
        self.revenue_start_price_weight = kwargs.get('revenue_start_price_weight', 1.)

        self.all_node_attrs = list(create_attrs_from_setting(node_attrs_setting).values())
        self.all_link_attrs = list(create_attrs_from_setting(link_attrs_setting).values())

        self.node_resource_attrs = [n_attr for n_attr in self.all_node_attrs if n_attr.type == 'resource']
        self.link_resource_attrs = [l_attr for l_attr in self.all_link_attrs if l_attr.type == 'resource']
        self.security_ssi_threshold = kwargs.get('security_ssi_threshold', 4.0)
        self.security_max_level = kwargs.get('security_max_level', 6.0)
        self.lcb_z_score = kwargs.get('lcb_z_score', 1.645)

    def _wilson_lower_bound(self, successes: int, attempts: int) -> float:
        """Return a one-sided Wilson lower bound for a Bernoulli success rate.

        A node with no completed historical attempts has no empirical
        reliability evidence and receives an LCB of zero.  This statistic is
        reporting-only: it never changes feasibility, reward, or observation.
        """
        attempts = int(attempts)
        successes = int(successes)
        if attempts <= 0:
            return 0.0
        successes = min(max(successes, 0), attempts)
        rate = successes / attempts
        z2 = self.lcb_z_score ** 2
        denominator = 1.0 + z2 / attempts
        centre = rate + z2 / (2.0 * attempts)
        margin = self.lcb_z_score * np.sqrt(
            rate * (1.0 - rate) / attempts + z2 / (4.0 * attempts ** 2)
        )
        return float(max(0.0, min(1.0, (centre - margin) / denominator)))

    def evaluate_security_requirements(
            self, v_net: VirtualNetwork, p_net: PhysicalNetwork, solution: Solution
        ) -> dict:
        """Evaluate the SRSR conditions using the pre-feedback trust snapshot."""
        security_sensitive = float(getattr(v_net, 'ssi', 0.0)) >= self.security_ssi_threshold
        complete_node_mapping = len(solution['node_slots']) == v_net.num_nodes
        complete_link_mapping = len(solution['link_paths']) == v_net.num_links

        node_static_security_satisfied = complete_node_mapping
        node_security_satisfied = complete_node_mapping
        node_lcbs = []
        if complete_node_mapping:
            for v_node_id, p_node_id in solution['node_slots'].items():
                required_security = v_net.nodes[v_node_id].get('security', 0.0)
                if p_net.nodes[p_node_id].get('security', 0.0) < required_security:
                    node_static_security_satisfied = False
                required_trust = v_net.nodes[v_node_id].get('security', 0.0) / self.security_max_level
                if p_net.nodes[p_node_id].get('trust', 0.0) < required_trust:
                    node_security_satisfied = False
                node_lcbs.append(self._wilson_lower_bound(
                    p_net.nodes[p_node_id].get('trust_success_count', 0),
                    p_net.nodes[p_node_id].get('trust_attempt_count', 0),
                ))

        link_security_satisfied = complete_link_mapping
        if link_security_satisfied:
            for v_link, p_links in solution['link_paths'].items():
                required_security = v_net.links[v_link].get('security', 0.0)
                for p_link in p_links:
                    if p_net.edges[p_link].get('security', 0.0) < required_security:
                        link_security_satisfied = False
                        break
                if not link_security_satisfied:
                    break

        min_node_reliability_lcb = min(node_lcbs) if node_lcbs else 0.0
        static_security_satisfied = bool(
            solution['result'] and node_static_security_satisfied and link_security_satisfied
        )

        return {
            'security_sensitive': security_sensitive,
            'node_static_security_satisfied': node_static_security_satisfied,
            'node_security_satisfied': node_security_satisfied,
            'link_security_satisfied': link_security_satisfied,
            'static_security_satisfied': static_security_satisfied,
            'min_node_reliability_lcb': min_node_reliability_lcb,
            'lcb_0_7_satisfied': bool(static_security_satisfied and min_node_reliability_lcb >= 0.7),
            'lcb_0_8_satisfied': bool(static_security_satisfied and min_node_reliability_lcb >= 0.8),
            'lcb_0_9_satisfied': bool(static_security_satisfied and min_node_reliability_lcb >= 0.9),
            'security_requirements_satisfied': bool(
                solution['result'] and node_security_satisfied and link_security_satisfied
            ),
        }

    def count_partial_solution(self, v_net: VirtualNetwork, solution: Solution) -> dict:
        """
        Count the revenue and cost of a partial solution

        Args:
            v_net (VirtualNetwork): Virtual network
            solution (Solution): Partial solution

        Returns:
            dict: The information of partial solution with revenue and cost
        """
        """
        计算 局部解决方案 的收入和成本
        
        Args：
            v_net（虚拟网络）：虚拟网络
            solution（solution）：部分解决方案
            
        Returns：
            dict：具有收益和成本的部分解决方案的信息
        """
        # node revenue
        v_net_node_revenue = 0
        for nid in solution['node_slots'].keys():
            v_net_node_revenue += sum([v_net.nodes[nid][n_attr.name] for n_attr in self.node_resource_attrs])
        v_net_link_revenue = 0
        v_net_link_cost = 0
        # link revenue
        for v_link, p_links in solution['link_paths'].items():
            one_revenue = sum([v_net.links[v_link][l_attr.name] for l_attr in self.link_resource_attrs])
            v_net_link_revenue += one_revenue
            for p_link in p_links:
                one_cost = sum([solution['link_paths_info'][(v_link, p_link)][l_attr.name] for l_attr in self.link_resource_attrs])
                v_net_link_cost += one_cost

        solution['v_net_node_revenue'] = v_net_node_revenue
        solution['v_net_link_revenue'] = v_net_link_revenue

        solution['v_net_revenue'] = v_net_node_revenue + v_net_link_revenue
        solution['v_net_link_cost'] = v_net_link_cost
        solution['v_net_path_cost'] = v_net_link_cost - v_net_link_revenue
        solution['v_net_node_cost'] = v_net_node_revenue
        solution['v_net_cost'] = solution['v_net_node_cost'] + solution['v_net_link_cost']
        solution['v_net_r2c_ratio'] = solution['v_net_revenue'] / solution['v_net_cost'] if solution['v_net_cost'] != 0 else 0
        return solution.to_dict()

    def count_security_solution(self, v_net: VirtualNetwork, p_net:PhysicalNetwork, solution: Solution) -> dict:
        """
        Count the revenue and cost of a partial solution

        Args:
            p_net (PhysicalNetwork): PhysicalNetwork
            v_net (VirtualNetwork): Virtual network
            solution (Solution): Partial solution

        Returns:
            dict: The information of partial solution with revenue and cost
        """
        """
        计算 局部解决方案 的收入和成本

        Args：
            v_net（虚拟网络）：虚拟网络
            p_net（物理物理）：物理网络
            solution（solution）：部分解决方案

        Returns：
            dict：具有收益和成本的部分解决方案的信息
        """
        # Preserve the pre-feedback snapshot prepared by the environment.  The
        # fallback makes this counter safe when called from an isolated solver.
        if not getattr(solution, 'security_snapshot_taken', False):
            solution.update(**self.evaluate_security_requirements(v_net, p_net, solution))
            solution['security_snapshot_taken'] = True

        TS = 0  # SFC安全性能要求
        # node revenue cost
        v_net_node_revenue = 0
        v_net_node_cost = 0

        for nid in solution['node_slots'].keys():
            Cg = 0
            for n_attr in self.node_resource_attrs:
                Cg += v_net.nodes[nid][n_attr.name]
            v_net_node_revenue += Cg * v_net.nodes[nid]['security']

            node_id = solution['node_slots'][nid]
            node = p_net.nodes[node_id]
            v_net_node_cost += Cg * node['trust']

            TS += node['trust']


        if solution['route_result'] == False:
            v_net_node_revenue -= Cg * v_net.nodes[nid]['security']
            v_net_node_cost -= Cg * node['trust']
            TS -= node['trust']

        # link revenue cost
        v_net_link_revenue = 0
        v_net_link_cost = 0
        for v_link, p_links in solution['link_paths'].items():
            one_revenue = sum([v_net.links[v_link][l_attr.name] * v_net.links[v_link]['security'] for l_attr in self.link_resource_attrs])
            v_net_link_revenue += one_revenue
            # v_link, p_links  ： (3, 2) ， [(57, 17), (17, 8), (8, 69), (69, 93)]
            for p_link in p_links:
                one_cost = sum(
                    [solution['link_paths_info'][(v_link, p_link)][l_attr.name] * p_net.edges[p_link]['security'] for l_attr in self.link_resource_attrs])
                v_net_link_cost += one_cost

                TS += p_net.edges[p_link]['security']

        # 链路花费一般大于收益


        solution['TS'] = TS
        solution['ssi'] = v_net.ssi

        solution['v_net_node_revenue'] = v_net_node_revenue
        solution['v_net_link_revenue'] = v_net_link_revenue
        solution['v_net_revenue'] = v_net_node_revenue + v_net_link_revenue

        solution['v_net_node_cost'] = v_net_node_cost
        solution['v_net_link_cost'] = v_net_link_cost
        solution['v_net_cost'] = solution['v_net_node_cost'] + solution['v_net_link_cost']

        solution['v_net_r2c_ratio'] = solution['v_net_revenue'] / solution['v_net_cost'] if solution['v_net_cost'] != 0 else 0

        solution['v_net_time_revenue'] = solution['v_net_revenue'] * v_net.lifetime
        solution['v_net_time_cost'] = solution['v_net_cost'] * v_net.lifetime
        solution['v_net_time_rc_ratio'] = solution['v_net_r2c_ratio'] * v_net.lifetime

        return solution.to_dict()

    def count_solution(self, v_net: VirtualNetwork, solution: Solution) -> dict:
        """
        Count the revenue and cost of a solution

        Args:
            v_net (VirtualNetwork): Virtual network
            solution (Solution): Solution

        Returns:
            dict: The information of partial solution with revenue and cost
        """
        """
        计算解决方案的收入和成本
        
        Args：
            v_net（虚拟网络）：虚拟网络
            solution（solution）：解决方案
        
        Returns：
            dict：具有收益和成本的部分解决方案的信息  
        """
        solution['num_placed_nodes'] = len(solution.node_slots)
        solution['num_routed_links'] = len(solution.link_paths) 
        solution['v_net_node_demand'] = self.calculate_sum_node_resource(v_net)
        solution['v_net_link_demand'] = self.calculate_sum_link_resource(v_net)
        solution['v_net_demand'] = solution['v_net_node_demand'] + solution['v_net_demand']
        # Success
        if solution['result']:
            solution['place_result'] = True
            solution['route_result'] = True
            solution['early_rejection'] = False
            solution['v_net_node_revenue'] = solution['v_net_node_demand']
            solution['v_net_link_revenue'] = solution['v_net_link_demand']
            solution['v_net_node_cost'] = solution['v_net_node_revenue']
            solution['v_net_link_cost'] = self.calculate_v_net_link_cost(v_net, solution)
            solution['v_net_path_cost'] = solution['v_net_link_cost'] - solution['v_net_link_revenue']
            solution['v_net_revenue'] = solution['v_net_node_revenue'] + solution['v_net_link_revenue']
            solution['v_net_cost'] = solution['v_net_revenue'] + solution['v_net_path_cost']
            solution['v_net_r2c_ratio'] = solution['v_net_revenue'] / solution['v_net_cost'] if solution['v_net_cost'] != 0 else 0
        # Faliure
        else:
            solution['v_net_node_revenue'] = 0
            solution['v_net_link_revenue'] = 0
            solution['v_net_revenue'] = 0
            solution['v_net_path_cost'] = 0
            solution['v_net_cost'] = 0
            solution['v_net_r2c_ratio'] = 0
            # solution['node_slots'] = {}
            # solution['link_paths'] = {}
        solution['v_net_time_revenue'] = solution['v_net_revenue'] * v_net.lifetime
        solution['v_net_time_cost'] = solution['v_net_cost'] * v_net.lifetime
        solution['v_net_time_rc_ratio'] = solution['v_net_r2c_ratio'] * v_net.lifetime
        return solution.to_dict()

    def calculate_sum_network_resource(self, network: Network, node: bool = True, link: bool = True):
        """
        Calculate the sum of network resource.

        Args:
            network (Network): Network
            node (bool, optional): Whether to calculate the sum of node resource. Defaults to True.
            link (bool, optional): Whether to calculate the sum of link resource. Defaults to True.

        Returns:
            float: The sum of network resource
        """
        n = np.array(network.get_node_attrs_data(self.node_resource_attrs)).sum() if node else 0
        e = np.array(network.get_link_attrs_data(self.link_resource_attrs)).sum() if link else 0
        return n + e

    def calculate_sum_node_resource(self, network: Network):
        """
        Calculate the sum of node resource.
        """
        n = np.array(network.get_node_attrs_data(self.node_resource_attrs)).sum()
        return n

    def calculate_sum_link_resource(self, network: Network):
        """
        Calculate the sum of link resource.
        """
        e = np.array(network.get_link_attrs_data(self.link_resource_attrs)).sum()
        return e

    def calculate_v_net_cost(self, v_net: VirtualNetwork, solution: Solution):
        v_net_node_cost = self.calculate_sum_node_resource(v_net)
        v_net_link_cost = self.calculate_v_net_link_cost(v_net, solution)
        return v_net_node_cost + v_net_link_cost

    def calculate_v_net_revenue(self, v_net: VirtualNetwork, solution: Solution = None):
        """
        Calculate the deployment cost of current v_net according to `link paths`.
        """
        return self.calculate_sum_network_resource(v_net)

    def calculate_v_net_link_cost(self, v_net: VirtualNetwork, solution: Solution):
        """
        Calculate the deployment cost of current v_net according to `link paths`.
        """
        sum_link_cost = 0
        for v_link, p_links in solution['link_paths'].items():
            for p_link in p_links:
                for l_attr in self.link_resource_attrs:
                    sum_link_cost += solution['link_paths_info'][(v_link, p_link)][l_attr.name]
        return sum_link_cost

    def summary_records(self, records: Union[list, pd.DataFrame]):
        """
        Summarize the records.

        Args:
            records (Union[list, pd.DataFrame]): The records to be summarized.

        Returns:
            dict: The summary information.
        """
        if isinstance(records, list):
            records = pd.DataFrame(records)
        elif isinstance(records, pd.DataFrame):
            pass
        else:
            raise TypeError
        summary_info = {}
        # ac rate
        summary_info['acceptance_rate'] = records.iloc[-1]['success_count'] / records.iloc[-1]['v_net_count']
        summary_info['success_count'] = records.iloc[-1]['success_count']
        summary_info['early_rejection_count'] = ((records['event_type']==1) & (records['early_rejection']==True)).sum()
        summary_info['place_failure_count'] = ((records['event_type']==1) & (records['place_result']==False)).sum()
        summary_info['route_failure_count'] = ((records['event_type']==1) & (records['route_result']==False)).sum()
        arrival_records = records[records['event_type'] == 1]
        security_sensitive = arrival_records.get(
            'security_sensitive', pd.Series(False, index=arrival_records.index, dtype=bool)
        ).fillna(False).astype(bool)
        deployed = arrival_records.get(
            'result', pd.Series(False, index=arrival_records.index, dtype=bool)
        ).fillna(False).astype(bool)
        security_satisfied = arrival_records.get(
            'security_requirements_satisfied',
            pd.Series(False, index=arrival_records.index, dtype=bool),
        ).fillna(False).astype(bool)
        deployed_sensitive = security_sensitive & deployed
        summary_info['security_sensitive_arrival_count'] = int(security_sensitive.sum())
        summary_info['security_sensitive_deployed_count'] = int(deployed_sensitive.sum())
        summary_info['security_sensitive_acceptance_rate'] = (
            summary_info['security_sensitive_deployed_count'] / summary_info['security_sensitive_arrival_count']
            if summary_info['security_sensitive_arrival_count'] else 0.0
        )
        # Backward-compatible name: historically this counted successfully
        # deployed sensitive requests, not all sensitive arrivals.
        summary_info['security_sensitive_request_count'] = summary_info['security_sensitive_deployed_count']
        summary_info['security_sensitive_success_count'] = int(
            (deployed_sensitive & security_satisfied).sum()
        )
        summary_info['srsr'] = (
            summary_info['security_sensitive_success_count'] / summary_info['security_sensitive_request_count']
            if summary_info['security_sensitive_request_count'] else 0.0
        )
        # ``srsr`` is retained only for compatibility with old result files.
        # Revised reporting separates hard static compliance from empirical
        # reliability and also reports an arrival-based metric that cannot be
        # inflated merely by rejecting difficult requests.
        static_security = arrival_records.get(
            'static_security_satisfied',
            pd.Series(False, index=arrival_records.index, dtype=bool),
        ).fillna(False).astype(bool)
        lcb_0_7 = arrival_records.get(
            'lcb_0_7_satisfied',
            pd.Series(False, index=arrival_records.index, dtype=bool),
        ).fillna(False).astype(bool)
        lcb_0_8 = arrival_records.get(
            'lcb_0_8_satisfied',
            pd.Series(False, index=arrival_records.index, dtype=bool),
        ).fillna(False).astype(bool)
        lcb_0_9 = arrival_records.get(
            'lcb_0_9_satisfied',
            pd.Series(False, index=arrival_records.index, dtype=bool),
        ).fillna(False).astype(bool)
        min_node_lcb = pd.to_numeric(
            arrival_records.get(
                'min_node_reliability_lcb',
                pd.Series(0.0, index=arrival_records.index, dtype=float),
            ),
            errors='coerce',
        ).fillna(0.0)
        deployed_denominator = summary_info['security_sensitive_deployed_count']
        arrival_denominator = summary_info['security_sensitive_arrival_count']
        summary_info['static_security_satisfied_count'] = int(
            (deployed_sensitive & static_security).sum()
        )
        summary_info['static_security_satisfaction_rate'] = (
            summary_info['static_security_satisfied_count'] / deployed_denominator
            if deployed_denominator else 0.0
        )
        summary_info['lcb_0_7_satisfied_count'] = int((deployed_sensitive & lcb_0_7).sum())
        summary_info['lcb_0_7_rate'] = (
            summary_info['lcb_0_7_satisfied_count'] / deployed_denominator
            if deployed_denominator else 0.0
        )
        summary_info['lcb_0_8_satisfied_count'] = int((deployed_sensitive & lcb_0_8).sum())
        summary_info['lcb_0_8_rate'] = (
            summary_info['lcb_0_8_satisfied_count'] / deployed_denominator
            if deployed_denominator else 0.0
        )
        summary_info['lcb_0_9_satisfied_count'] = int((deployed_sensitive & lcb_0_9).sum())
        summary_info['lcb_0_9_rate'] = (
            summary_info['lcb_0_9_satisfied_count'] / deployed_denominator
            if deployed_denominator else 0.0
        )
        summary_info['mean_min_node_reliability_lcb'] = (
            float(min_node_lcb[deployed_sensitive].mean())
            if deployed_denominator else 0.0
        )
        summary_info['reliable_secure_deployment_rate_lcb_0_7'] = (
            summary_info['lcb_0_7_satisfied_count'] / arrival_denominator
            if arrival_denominator else 0.0
        )
        summary_info['reliable_secure_deployment_rate_lcb_0_8'] = (
            summary_info['lcb_0_8_satisfied_count'] / arrival_denominator
            if arrival_denominator else 0.0
        )
        summary_info['reliable_secure_deployment_rate_lcb_0_9'] = (
            summary_info['lcb_0_9_satisfied_count'] / arrival_denominator
            if arrival_denominator else 0.0
        )
        # rc ratio
        summary_info['r2c_ratio'] = records.iloc[-1]['total_revenue'] / records.iloc[-1]['total_cost'] if records.iloc[-1]['total_cost'] else 0
        summary_info['time_r2c_ratio'] = records.iloc[-1]['total_time_revenue'] / records.iloc[-1]['total_time_cost']  if records.iloc[-1]['total_time_cost'] else 0
        summary_info['total_revenue'] = records.iloc[-1]['total_revenue']
        summary_info['total_cost'] = records.iloc[-1]['total_cost']
        summary_info['total_time_revenue'] = records.iloc[-1]['total_time_revenue']
        summary_info['total_time_cost'] = records.iloc[-1]['total_time_cost']
        # revenue / cost
        summary_info['total_simulation_time'] = records.iloc[-1]['v_net_arrival_time']
        summary_info['long_term_aver_revenue'] = summary_info['total_revenue'] / summary_info['total_simulation_time']
        summary_info['long_term_aver_cost'] = summary_info['total_cost'] / summary_info['total_simulation_time']
        summary_info['long_term_aver_r2c_ratio'] = summary_info['long_term_aver_revenue'] / summary_info['long_term_aver_cost']
        summary_info['long_term_aver_time_revenue'] = summary_info['total_time_revenue'] / summary_info['total_simulation_time']
        summary_info['long_term_aver_time_cost'] = summary_info['total_time_cost'] / summary_info['total_simulation_time']
        # summary_info['long_term_weighted_aver_time_revenue'] = self.revenue_service_time_weight * summary_info['long_term_aver_time_revenue'] + self.revenue_start_price_weight * summary_info['long_term_aver_revenue']
        # summary_info['total_simulation_time'] = records[records['event_type']==1].iloc[-1]['arrival_time']
        # state
        summary_info['min_p_net_available_resource'] = records.loc[:, 'p_net_available_resource'].min()
        summary_info['min_p_net_node_available_resource'] = records.loc[:, 'p_net_node_available_resource'].min()
        summary_info['min_p_net_link_available_resource'] = records.loc[:, 'p_net_link_available_resource'].min()
        summary_info['max_inservice_count'] = records.loc[:, 'inservice_count'].max()
        # rl reward
        # if 'cumulative_reward' in records.columns:
        #     cumulative_rewards = records.loc[:, 'cumulative_reward'].dropna()
        #     summary_info['cumulative_reward'] = cumulative_rewards.iloc[-1]
        # else:
        #     summary_info['cumulative_reward'] = 0
        return summary_info

    @classmethod
    def summary_csv(cls, fpath: str):
        """
        Summary the records in csv file.

        Args:
            fpath (str): The path of csv file.

        Returns:
            dict: The summary information.
        """
        records = pd.read_csv(fpath, header=0)
        summary_info = cls.summary_records(records)
        return summary_info
