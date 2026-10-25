
import copy
import pprint
from collections import OrderedDict


class ClassDict(object):

    def __init__(self):
        super(ClassDict, self).__init__()

    def update(self, *args, **kwargs):
        """Update the ClassDict object with key-value pairs from a dictionary or another ClassDict object."""
        for k, v in kwargs.items():
            self[k] = v

    @classmethod
    def from_dict(cls, dict):
        """Create a new ClassDict object from a dictionary."""
        cls.__dict__ = dict
        return cls

    def to_dict(self):
        """Return a dictionary containing the contents of the ClassDict object."""
        solution_dict = copy.deepcopy(self.__dict__)
        for order_dict_key in ['node_slots', 'link_paths', 'node_slots_info', 'link_paths_info']:
            solution_dict[order_dict_key] = dict(solution_dict[order_dict_key])
        return solution_dict

    def __getitem__(self, key):
        if isinstance(key, str):
            return getattr(self, key, None)
        elif isinstance(key, int):
            return super().__getitem__(key)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)


class Solution(ClassDict):
    """
    A class representing a solution to a virtual network mapping problem.

    Attributes:
        v_net_id: The ID of the virtual network being mapped.
        v_net_lifetime: The lifetime of the virtual network being mapped.
        v_net_arrival_time: The arrival time of the virtual network being mapped.
        v_net_num_nodes: The number of nodes in the virtual network being mapped.
        v_net_num_egdes: The number of edges in the virtual network being mapped.
        result: A boolean indicating whether the virtual network has been successfully mapped.
        node_slots: A dictionary mapping node IDs to the IDs of the slots they are mapped to.
        link_paths: A dictionary mapping link IDs to the IDs of the paths they are mapped to.
        node_slots_info: A dictionary mapping node IDs to the information of the slots they are mapped to.
        link_paths_info: A dictionary mapping link IDs to the information of the paths they are mapped to.
        v_net_cost: The total cost of the virtual network being mapped.
        v_net_revenue: The total revenue of the virtual network being mapped.
        v_net_demand: The total demand of the virtual network being mapped.
        v_net_node_demand: The total demand of the nodes in the virtual network being mapped.
        v_net_link_demand: The total demand of the links in the virtual network being mapped.
        v_net_node_revenue: The total revenue of the nodes in the virtual network being mapped.
        v_net_link_revenue: The total revenue of the links in the virtual network being mapped.
        v_net_node_cost: The total cost of the nodes in the virtual network being mapped.
        v_net_link_cost: The total cost of the links in the virtual network being mapped.
        v_net_path_cost: The total cost of the paths in the virtual network being mapped.
        v_net_r2c_ratio: The revenue-to-cost ratio of the virtual network being mapped.
        v_net_time_cost: The total time cost of the virtual network being mapped.
        v_net_time_revenue: The total time revenue of the virtual network being mapped.
        v_net_time_rc_ratio: The time revenue-to-cost ratio of the virtual network being mapped.
        description: A string describing the solution.
        total_violation: The total violation of the solution.
        current_violation: The current violation of the solution.
        total_place_violation: The total placement violation of the solution.
        total_route_violation: The total routing violation of the solution.
        place_result: A boolean indicating whether the placement of the virtual network has been successfully mapped.
        route_result: A boolean indicating whether the routing of the virtual network has been successfully mapped.
        early_rejection: A boolean indicating whether the virtual network has been rejected before the mapping process.
        revoke_times: The number of times the virtual network has been revoked.
        selected_actions: A list of actions selected by the agent.
    """
    """
    代表虚拟网络映射问题解决方案的类。

    Args：
        v_net_id： 正在映射的虚拟网络的 ID。
        v_net_lifetime： 正在映射的虚拟网络的生命周期。
        v_net_arrival_time（到达时间）： 正在映射的虚拟网络的到达时间。
        v_net_num_nodes（节点数）： 正在映射的虚拟网络中的节点数。
        v_net_num_edges： 正在映射的虚拟网络中的边的数量。
        result：结果： 布尔值，表示虚拟网络是否已成功映射。
        
        node_slots：节点插槽： 将节点 ID 映射到所映射插槽 ID 的字典。
        link_paths： 将链接 ID 映射到所映射路径 ID 的字典。
        node_slots_info： 将节点 ID 与它们所映射的插槽信息进行映射的字典。
        link_paths_info： 一个字典，用于将链接 ID 映射到其所映射的路径信息。
        
        v_net_cost（网络成本）： 被映射虚拟网络的总成本。
        v_net_revenue: 虚拟网络的总收入： 被映射虚拟网络的总收入。
        v_net_demand: 虚拟网络的总需求： 映射的虚拟网络的总需求。
        v_net_node_demand（节点需求）： 映射的虚拟网络中节点的总需求。
        v_net_link_demand： 正在映射的虚拟网络中链路的总需求。
        v_net_node_revenue（节点收入）： 映射的虚拟网络中节点的总收入。
        v_net_link_revenue: 虚拟网络中链路的总收入： 正在映射的虚拟网络中链路的总收入。
        v_net_node_cost（节点成本）： 被映射虚拟网络中节点的总成本。
        v_net_link_cost： 正在映射的虚拟网络中链路的总成本。
        v_net_path_cost： 正在映射的虚拟网络中路径的总成本。
        v_net_r2c_ratio： 正在映射的虚拟网络的收入成本比。
        v_net_time_cost（网络时间成本）： 正在映射的虚拟网络的总时间成本。
        v_net_time_revenue（网络时间收入）： 正在映射的虚拟网络的总时间收入。
        v_net_time_rc_ratio： 正在映射的虚拟网络的时间收入与成本比率。
        
        description：描述解决方案的字符串： 描述解决方案的字符串。
        total_violation： 解决方案的总违规次数。
        current_violation： 解决方案的当前违规情况。
        total_place_violation： 解决方案的总违规位置。
        total_route_violation： 解决方案的总路由违规次数。
        place_result（放置结果）： 一个布尔值，表示虚拟网络的位置是否已成功映射。
        route_result：路由结果： 表示虚拟网络的路由是否已成功映射的布尔值。
        early_rejection： 一个布尔值，表示虚拟网络是否在映射过程之前被拒绝。
        revoke_times（撤销次数）： 虚拟网络被撤销的次数。
        selected_actions： 代理选择的操作列表。
    """
    def __init__(self, v_net):
        """
        Creates a new Solution object.

        Args:
            v_net: The virtual network being mapped.

        Returns:
            A new Solution object.
        """
        super(Solution, self).__init__()
        self.v_net_id = v_net.id
        self.v_net_lifetime = v_net.lifetime
        self.v_net_arrival_time = v_net.arrival_time
        self.v_net_num_nodes = v_net.num_nodes
        self.v_net_num_egdes = v_net.num_links
        self.reset()

    def reset(self):
        """Resets all attributes of the object to their initial state."""
        self.result = False
        self.node_slots = OrderedDict()
        self.link_paths = OrderedDict()
        self.node_slots_info = OrderedDict()
        self.link_paths_info = OrderedDict()

        self.v_net_cost = 0
        self.v_net_revenue = 0
        self.v_net_demand = 0
        self.v_net_node_demand = 0
        self.v_net_link_demand = 0
        self.v_net_node_revenue = 0
        self.v_net_link_revenue = 0
        self.v_net_node_cost = 0
        self.v_net_link_cost = 0
        # self.v_net_path_cost = 0
        self.v_net_r2c_ratio = 0
        self.v_net_time_cost = 0
        self.v_net_time_revenue = 0
        self.v_net_time_rc_ratio = 0

        self.description = ''
        self.total_violation = 0.
        self.current_violation = 0.
        self.total_place_violation = 0.
        self.total_route_violation = 0.
        self.place_result = True
        self.route_result = True
        self.early_rejection = False
        self.revoke_times = 0
        self.selected_actions = []
        # Snapshot fields are written before the current request updates node
        # trust, then persisted unchanged in the request record.
        self.security_snapshot_taken = False
        self.security_sensitive = False
        self.node_static_security_satisfied = False
        self.node_security_satisfied = False
        self.link_security_satisfied = False
        self.static_security_satisfied = False
        self.min_node_reliability_lcb = 0.0
        self.lcb_0_7_satisfied = False
        self.lcb_0_8_satisfied = False
        self.security_requirements_satisfied = False

    def is_feasible(self):
        """
        Checks if the solution is feasible.

        Returns:
            True if the solution is feasible, False otherwise.
        """
        """
        检查解决方案是否可行
        
        Returns：
            如果解决方案可行，则为True，否则为False。
        """
        return self.result and self.total_violation <= 0

    def display(self):
        """Pretty print the solution object's attributes using pprint module."""
        pprint.pprint(self.__dict__)


    def __repr__(self):
        pprint.pprint(self.__dict__)
        return super().__repr__()
