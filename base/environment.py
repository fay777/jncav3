import os
import copy
import time
import numpy as np
from pprint import pprint

from utils import get_p_net_dataset_dir_from_setting, get_v_nets_dataset_dir_from_setting
from .solution import Solution


def clone_network(network):
    """Return an isolated custom-network copy compatible with NetworkX 3.x.

    ``copy.deepcopy`` no longer works reliably for the custom ``Network``
    subclasses because newer NetworkX graph views do not expose the legacy
    private adjacency fields expected by pickle.  Graph.copy() preserves the
    topology; the explicit deep copies below isolate resource attributes for
    an episode rollback/reset.  Rebuilding attribute objects is required
    because NetworkX creates a subclass through its no-argument constructor.
    """
    cloned = network.copy(as_view=False)
    cloned.graph = copy.deepcopy(network.graph)
    for node, attrs in network.nodes(data=True):
        cloned.nodes[node].clear()
        cloned.nodes[node].update(copy.deepcopy(attrs))
    for source, target, attrs in network.edges(data=True):
        cloned.edges[source, target].clear()
        cloned.edges[source, target].update(copy.deepcopy(attrs))
    cloned.create_attrs_from_setting()
    # Preserve cached normalization benchmarks (for PhysicalNetwork) and any
    # other domain-specific state outside NetworkX's graph/node/edge stores.
    networkx_state = {'graph', '_node', '_adj', 'node_attrs', 'link_attrs'}
    for name, value in network.__dict__.items():
        if name not in networkx_state:
            try:
                setattr(cloned, name, copy.deepcopy(value))
            except AttributeError:
                # ``nodes`` and ``links`` are read-only NetworkX views and
                # are reconstructed from the copied topology.
                continue
    return cloned


class Environment:
    """A general environment for various solvers based on heuristics and RL"""
    """ 基于启发式和 RL 的各种求解器的通用环境 """
    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, **kwargs):
        self.p_net = p_net
        self.init_p_net = clone_network(p_net)
        self.v_net_simulator = v_net_simulator
        self.num_v_nets = self.v_net_simulator.num_v_nets
        self.num_events = self.num_v_nets * 2
        self.controller = controller
        self.recorder = recorder
        self.counter = counter
        self.verbose = kwargs.get('verbose', 1)
        self.p_net_dataset_dir = kwargs.get('p_net_dataset_dir', 'unknown_p_net_dataset_dir')
        self.v_nets_dataset_dir = kwargs.get('v_nets_dataset_dir', 'unknown_v_nets_dataset_dir')
        self.renew_v_net_simulator = kwargs.get('renew_v_net_simulator', False)

        self.solver_name = kwargs.get('solver_name', 'unknown_slover')
        self.run_id = kwargs.get('run_id', 'unknown_device-unknown_run_time')
        self.seed = kwargs.get('seed', None)
        self.if_save_records = kwargs.get('if_save_records', True)
        self.summary_file_name = kwargs.get('summary_file_name', 'global_summary.csv')

        # ranking strategy
        self.node_ranking_method = kwargs.get('node_ranking_method', 'order')
        self.link_ranking_method = kwargs.get('link_ranking_method', 'order')
        # node mapping
        self.matching_mathod = kwargs.get('matching_mathod', 'greedy')
        # link mapping
        self.shortest_method = kwargs.get('shortest_method', 'k_shortest')
        self.k_shortest = kwargs.get('k_shortest', 10)

        self.extra_summary_info = {}
        self.extra_record_info = {}

    def ready(self, event_id: int = 0):
        """
        Prepare for the given event.

        Args:
            event_id: the id of the event to be processed.
        """
        """
            为给定事件做准备。

            参数：
                event_id：要处理的事件的 ID。
        """
        self.curr_event = self.v_net_simulator.events[event_id]
        self.num_processed_v_nets += 1
        self.v_net = self.v_net_simulator.v_nets[int(self.curr_event['v_net_id'])]
        self.solution = Solution(self.v_net)

        self.p_net_backup = clone_network(self.p_net) if self.curr_event['type'] == 1 else None  #备份处理event前的物理网络拓情况，前一次失败方便回滚

        self.recorder.update_state({
            'event_id': self.curr_event['id'],
            'event_type': self.curr_event['type'],
            'event_time': self.curr_event['time'],
        })
        if self.verbose >= 2:
            print(f"\nEvent: id={event_id}, type={self.curr_event['type']}")
            print(f"{'-' * 30}")

    def reset(self, seed: int = None):
        """
        Reset the environment.
        每一次epoch重载一次环境
        Args:
            seed: the seed for the random number generator. If None, use the seed in the config.
        """
        seed = seed if seed is not None else self.seed

        self.p_net = clone_network(self.init_p_net)
        self.recorder.reset()
        self.recorder.count_init_p_net_info(self.p_net)
        if self.recorder.if_temp_save_records and self.verbose >= 1:
            print(f'temp save record in {self.recorder.temp_save_path}\n')  # 临时保存当前epoch的处理进度

        self.v_nets_dataset_dir = get_v_nets_dataset_dir_from_setting(self.v_net_simulator.v_sim_setting)
        if self.renew_v_net_simulator:
            self.v_net_simulator.renew(v_nets=True, events=True, seed=seed)
            print(f'Generate virtual networks with seed {seed}') if self.verbose >= 1 else None
        elif os.path.exists(self.v_nets_dataset_dir):
            self.v_net_simulator = self.v_net_simulator.load_dataset(self.v_nets_dataset_dir)
            print(f'Load virtual networks from {self.v_nets_dataset_dir}') if self.verbose >= 1 else None
        else: 
            self.v_net_simulator.renew(v_nets=True, events=True, seed=seed)
            print(f'\n*** Generate virtual networks with seed {seed}')
        self.cumulative_reward = 0
        self.num_processed_v_nets = 0
        self.start_run_time = time.time()
        self.ready(event_id=0)
        return self.get_observation()

    def step(self, action):
        """
        Take an action and return the next observation, reward, done, and info.

        Args:
            action: the action to be taken.

        Returns:
            observation: the observation after taking the action.
            reward: the reward after taking the action.
            done: whether the episode is done.
            info: the extra information.
        """
        """
            执行一个操作并返回下一个observation, reward, done, and info.

            Args：
                action：要采取的行动。
    
            Returns：
                observation：执行操作后的观察结果。
                reward：执行操作后的奖励。
                done：事件是否完成。
                info：额外信息。
        """
        return NotImplementedError

    def compute_reward(self) -> float:
        """Compute the reward for the current Virtual Network."""
        return NotImplementedError

    def get_observation(self) -> dict:
        """Get the observation for the current Virtual Network."""
        return NotImplementedError

    def render(self, mode="human") -> None:
        """
        Render the environment.

        Args:
            mode: the mode to render the environment.
        """
        """
            渲染环境

            Args：
                mode：渲染环境的模式。
        """
        return NotImplementedError

    def release(self) -> dict:
        """
        Release the current Virtual Network when it leaves the system.
        """
        solution = self.recorder.get_record(v_net_id=self.v_net['id'])
        self.controller.release(self.v_net, self.p_net, solution)
        self.solution['description'] = 'Leave Event'
        record = self.count_and_add_record()  # 这里会写一次temp.csv
        return record

    def get_failure_reason(self, solution: Solution) -> str:
        """
        Get the reason of failure, which is used to rollback the state of the physical network.

        Args:
            solution (Solution): the solution of the current Virtual Network.

        Returns:
            reason (str): the reason of failure.
        """
        """
            获取故障原因，用于回滚物理网络的状态。

            Args：
                solution （Solution）：当前虚拟网络的解决方案。
    
            Returns:
                reason (str)：失败原因。
        """
        if solution['early_rejection']:
            return 'reject'
        if not solution['place_result']:
            return 'place'
        elif not solution['route_result']:
            return 'route'
        else:
            return 'unknown'

    def rollback_for_failure(self, reason='place') -> None:
        """
        Rollback the state of the physical network for the failure of the current Virtual Network.

        Args:
            reason (str): the reason of failure.
        """
        """
            回滚当前虚拟网络故障时的物理网络状态

            Args：
                reason (str)：故障原因
        """
        # self.solution.reset()
        self.p_net = clone_network(self.p_net_backup)
        if reason in ['unknown', -1]:
            self.solution['description'] = 'Unknown Reason'
        if reason in ['reject', 0]:
            self.solution['description'] = 'Early Rejection'
            self.solution['early_rejection'] = True
        elif reason in ['place', 1]:
            self.solution['description'] = 'Place Failure'
            self.solution['place_result'] = False
        elif reason in ['route', 2]:
            self.solution['description'] = 'Route Failure'
            self.solution['route_result'] = False
        else:
            return NotImplementedError

    def transit_obs(self) -> bool:
        """
        Automatically Transit the observation to the next event until the next enter event comes or episode has done.

        Returns:
            done (bool): whether the episode is done.
        """
        """
            过渡  缓冲  处理生存周期已经结束的SFC请求
            
            自动将观测转到下一个event，直到下一个enter event到来 或epoch结束。

            Returns:
                done（bool）：episode是否结束。
        """

        # Leave events transition
        while True:
            next_event_id = int(self.curr_event['id'] + 1)
            # episode finished
            if next_event_id > self.num_events - 1:  # 一共2000个event ： 1000个SFC请求进入 1000个SFC请求离开
                return True
            # 如果当前epoch没结束，则为下一个SFC处理做准备  下一个event可能是 离开事件也可能是进入事件
            self.ready(next_event_id)
            # 如果当期处理的event是离开事件，则释放当期event的资源，并跳到下一个event 判断是否需要ready() 直至跳到 进入事件
            if self.curr_event['type'] == 0: # 以SFC为单位    type为 1：进入 ，type为 0：离开
                record = self.release()  # 释放持续时间结束的SFC请求
            else:
                return False

    @property
    def selected_p_net_nodes(self) -> list:
        """Get the already selected physical nodes for the current Virtual Network."""
        """
            为当前虚拟网络获取 已选择的物理节点
        """
        return list(self.solution['node_slots'].values())

    @property
    def placed_v_net_nodes(self) -> list:
        """Get the already placed virtual nodes for the current Virtual Network."""
        """
            获取 当前虚拟网络中已放置的虚拟节点
        """
        return list(self.solution['node_slots'].keys())

    @property
    def num_placed_v_net_nodes(self) -> int:
        """Get the number of already placed virtual nodes for the current Virtual Network."""
        """
            获取 当前虚拟网络中已放置的虚拟节点的数量
        """
        return len(self.solution['node_slots'].keys())

    ### recorder ###
    def add_record(self, record: dict, extra_info: dict = {}) -> dict:
        """
        Add extra information to the record and add the record to the recorder.

        Args:
            record (dict): the record to be added.
            extra_info (dict): the extra information to be added.

        Returns:
            record (dict): the record with extra information.
        """
        """
            为记录添加额外信息，并将记录添加到记录器中。

            Args：
                record （dict）：要添加的记录。
                extra_info （dict）：要添加的额外信息。
    
            Returns:
                record （dict）：添加了额外信息的记录。
        """
        self.extra_record_info.update(extra_info)
        record = self.recorder.add_record(record, self.extra_record_info)
        if self.verbose >= 2:
            self.display_record(record, extra_items=list(extra_info.keys()))
        return record

    def count_and_add_record(self, extra_info: dict = {}) -> dict:
        """
        Count the record and add the record to the recorder.

        Args:
            extra_info (dict): the extra information to be added.
        """
        """
            计算记录并将记录添加到记录器中。

            Args：
                extra_info（dict）：要添加的额外信息。
        """
        record = self.recorder.count(self.v_net, self.p_net, self.solution)
        record = self.add_record(record, extra_info)   # 写一次临时 temp.csv
        return record

    def display_record(
            self, 
            record: dict, 
            display_items: list = ['result', 'v_net_id', 'v_net_cost', 'v_net_revenue', 
                        'p_net_available_resource', 'total_revenue', 'total_cost', 'description'], 
            extra_items: list = []
        ) -> None:
        """
        Display the record, including the default display items and extra display items.

        Args:
            record (dict): the record to be displayed.
            display_items (list): the default display items.
            extra_items (list): the extra display items.
        """
        """
            显示记录，包括默认显示项和额外显示项。

            Args：
                record （dict）：要显示的记录。
                display_items （列表）：默认显示项目。
                extra_items（列表）：额外显示项。
        """
        display_items = display_items + extra_items
        print(''.join([f'{k}: {v}\n' for k, v in record.items() if k in display_items]))

    def summary_records(self, extra_summary_info={}, summary_file_name=None, record_file_name=None) -> None:
        """
        Summarize the records and save the summary information and records to the file.

        Args:
            extra_summary_info (dict): the extra summary information to be added.
            summary_file_name (str): the name of the summary file.
            record_file_name (str): the name of the record file.
        """
        """
            汇总记录并将汇总信息和记录保存到文件中。

            Args：
                extra_summary_info（dict）：要添加的额外摘要信息。
                summary_file_name （str）：摘要文件的名称。
                record_file_name （str）：记录文件的名称。
        """
        start_run_time = time.strftime('%Y%m%dT%H%M%S', time.localtime(self.start_run_time))
        if summary_file_name is None:
            summary_file_name = self.summary_file_name
        if record_file_name is None:
            record_file_name = f'{self.solver_name}-{self.run_id}-{start_run_time}.csv'  # drl-sfcp-main\save\a3c_gcn_seq2seq\LAPTOP-M43DF0FU-20250514T194148\records
        summary_info = self.recorder.summary_records(self.recorder.memory)
        end_run_time = time.time()
        clock_running_time = end_run_time - self.start_run_time  # 时钟运行时间
        run_info_dict = {
            'solver_name': self.solver_name,
            'seed': self.seed,
            'p_net_dataset_dir': self.p_net_dataset_dir,
            'v_nets_dataset_dir': self.v_nets_dataset_dir,
            'run_id': self.run_id,
            'start_run_time': start_run_time, 
            'clock_running_time': clock_running_time
        }
        for k, v in extra_summary_info.items():
            run_info_dict[k] = v
        info = {**summary_info, **run_info_dict}  # 保存到 global_summary.csv

        if self.if_save_records:
            record_path = self.recorder.save_records(record_file_name)
            summary_path = self.recorder.save_summary(info, summary_file_name)
                
        if self.verbose >= 1:
            pprint(info)
            if self.if_save_records:
                print(f'save records to {record_path}')
                print(f'save summary to {summary_path}')
        return info


class SolutionStepEnvironment(Environment):
    
    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, **kwargs):
        super(SolutionStepEnvironment, self).__init__(p_net, v_net_simulator, controller, recorder, counter, **kwargs)

    def step(self, solution: Solution):
        """
        Step the environment with the solution.

        Args:
            solution (Solution): the solution to be deployed.

        Returns:
            observation (dict): the observation of the environment.
            reward (float): the reward of the step.
            done (bool): whether the episode is done.
            info (dict): the information of the step.
        """
        # Enter 
        self.solution = solution
        if self.solution['total_violation'] > 0: solution['result'] = False
        # Snapshot SRSR before this request contributes a success/failure
        # outcome to dynamic trust.  The controller and its action mask remain
        # untouched; this is accounting only.
        if self.recorder.state['event_type'] == 1:
            self.solution.update(**self.counter.evaluate_security_requirements(
                self.v_net, self.p_net, self.solution
            ))
            self.solution['security_snapshot_taken'] = True
        # self.counter.count_solution(self.v_net, self.solution)  # 计算解决方案的收入和成本
        # 666
        # self.counter.count_security_solution(self.v_net, self.p_net, self.solution)
        # Success
        if solution['result']:
            assert len(solution['node_slots']) == self.v_net.num_nodes
            assert len(solution['link_paths']) == self.v_net.num_links
            self.solution['description'] = 'Success'
            total_p_resource_2 = self.counter.calculate_sum_network_resource(self.p_net)
            self.controller.deploy(self.v_net, self.p_net, self.solution, self.recorder.state['success_count'])
            total_p_resource_1_n = self.counter.calculate_sum_node_resource(self.p_net)
            total_p_resource_1_e = self.counter.calculate_sum_link_resource(self.p_net)
            total_p_resource_0_n = self.counter.calculate_sum_node_resource(self.p_net_backup)
            total_p_resource_0_e = self.counter.calculate_sum_link_resource(self.p_net_backup)
            total_p_resource_1 = self.counter.calculate_sum_network_resource(self.p_net)
            total_p_resource_0 = self.counter.calculate_sum_network_resource(self.p_net_backup)
            # if total_p_resource_0 - total_p_resource_1 != solution['v_net_cost']:
            #     print('Unsafely Update!')
            assert total_p_resource_2 == total_p_resource_0
            # assert total_p_resource_0_n - total_p_resource_1_n == solution['v_net_node_cost'], f"{total_p_resource_0_n - total_p_resource_1_n}, {solution['v_net_node_cost']}"
            # assert total_p_resource_0_e - total_p_resource_1_e == solution['v_net_link_cost'], f"{total_p_resource_0_e - total_p_resource_1_e}, {solution['v_net_link_cost']}"
            # assert total_p_resource_0 - total_p_resource_1 == solution['v_net_cost'], f"{total_p_resource_0 - total_p_resource_1}, {solution['v_net_cost']}"
        # Failure
        else:
            failure_reason = self.get_failure_reason(self.solution)
            self.rollback_for_failure(reason=failure_reason)
        # 666
        record = self.count_and_add_record()  # 计算并得到当前SFC请求的处理情况信息    同时写一次temp.csv
        # Feedback is incorporated only after the record has captured the
        # pre-feedback trust state.  A failed partial mapping contributes an
        # attempt (but not a success) for every selected physical node.
        if self.recorder.state['event_type'] == 1:
            self.p_net.record_trust_attempt(
                self.solution['node_slots'].values(), self.solution['result']
            )
        # 释放 "离开event" 对应的vnf的资源
        done = self.transit_obs()  # 返回epoch是否结束  结束返回True
        return self.get_observation(), self.compute_reward(), done, self.get_info(record)

    def get_info(self, record={}):
        info = copy.deepcopy(record)
        return info

    def get_observation(self):
        return {'v_net': clone_network(self.v_net), 'p_net': clone_network(self.p_net)}

    def compute_reward(self):
        return 0

    def generate_action_mask(self):
        return np.arange(self.p_net.num_nodes)

class JointPRStepEnvironment(Environment):

    def __init__(self, p_net, v_net_simulator, controller, recorder, counter, **kwargs):
        super(JointPRStepEnvironment, self).__init__(p_net, v_net_simulator, controller, recorder, counter, **kwargs)


    @property
    def last_placed_v_node_id(self):
        """Get the last placed virtual node id."""
        if self.num_placed_v_net_nodes == 0:
            return None
        return list(self.solution['node_slots'].keys())[-1]

    @property
    def curr_v_node_id(self):
        """Get the current virtual node id to be placed."""
        if self.num_placed_v_net_nodes == self.v_net.num_nodes:
            return 0
        return self.v_net.ranked_nodes[self.num_placed_v_net_nodes]


    def step(self, p_node_id: int):
        """
        Step the environment with the solution.

        Args:
            p_node_id (int): the physical node id to be deployed.

        Returns:
            observation (dict): the observation of the environment.
            reward (float): the reward of the step.
            done (bool): whether the episode is done.
            info (dict): the information of the step.
        """
        place_and_route_result, place_and_route_info = self.controller.place_and_route(self.v_net, 
                                                                                        self.p_net, 
                                                                                        self.curr_v_node_id, 
                                                                                        p_node_id, 
                                                                                        self.solution, 
                                                                                        shortest_method=self.shortest_method, 
                                                                                        k=self.k_shortest)
        # Step Failure
        if not place_and_route_result:
            failure_reason = self.get_failure_reason(self.solution)
            self.rollback_for_failure(failure_reason)
        # Step Success
        else:
            # VN Success ?
            if self.num_placed_v_net_nodes == self.v_net.num_nodes:
                self.solution['result'] = True
            else:
                record = self.solution.to_dict()
                return self.get_observation(), self.compute_reward(), False, self.get_info(record)

        record = self.count_and_add_record(self.v_net, self.p_net, self.solution)

        # obs transition
        if not place_and_route_result or self.solution['result']:
            done = self.transit_obs()
            if done:
                summary_info = self.summary_records()
                pprint(summary_info)
        else:
            done = False
        return self.get_observation(), self.compute_reward(), done, self.get_info(record)

    def get_observation(self):
        """Get the observation of the environment."""
        return {'v_node_id': self.v_net[self.curr_v_node_id], 'p_net': self.p_net}

    def compute_reward(self):
        """Compute the reward of the step."""
        return 0

if __name__ == '__main__':
    pass
