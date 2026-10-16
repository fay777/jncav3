
from functools import cached_property
import os
import copy
import random
import numpy as np
import networkx as nx

from .network import Network
from .attribute import NodeInfoAttribute, LinkInfoAttribute


class PhysicalNetwork(Network):
    """
    PhysicalNetwork class is a subclass of Network class. It represents a physical network.

    Attributes:
        degree_benchmark (dict): The degree benchmark for the network.
        node_attr_benchmarks (dict): The node attribute benchmarks for the network.
        link_attr_benchmarks (dict): The link attribute benchmarks for the network.

    Methods:
        from_topology_zoo_setting(topology_zoo_setting: Dict[str, Any], seed: Optional[int] = None) -> PhysicalNetwork:
            Returns a PhysicalNetwork object generated from the Topology Zoo data, with optional seed for reproducibility.

        save_dataset(self, dataset_dir: str) -> None:
            Saves the dataset as a .gml file in the specified directory.

        load_dataset(dataset_dir: str) -> PhysicalNetwork:
            Loads the dataset from the specified directory as a PhysicalNetwork object, 
            and calculates the benchmarks for normalization.
    """
    """
        PhysicalNetwork类是Network类的一个子类。它代表了一个物理网络。
        
        Attributes：
            degree_benchmark（dict）：网络的度基准。
            node_attr_benchmarks（dict）：网络的节点属性基准。
            link_attr_benchmarks（dict）：网络的链接属性基准。
        
        Methods：
            from_topology_zoo_setting（topology_zoo_setting:Dict[str，Any]，种子：可选[int]=无）->物理网络：
                返回从Topology Zoo数据生成的PhysicalNetwork对象，并带有可选的种子以实现可重复性。
            save_daset（self，dataset_dir:str）->无：
                将数据集另存为指定目录中的.gml文件。
            load_dataset（dataset_dir:str）->物理网络：
                从指定目录加载数据集作为PhysicalNetwork对象，并计算归一化的基准。
    """
    def __init__(self, incoming_graph_data: nx.Graph = None, node_attrs_setting: list = [], link_attrs_setting: list = [], **kwargs) -> None:
        """
        Initialize a PhysicalNetwork object.

        Args:
            incoming_graph_data (nx.Graph): An existing graph object (optional).
            node_attrs_setting (list): Node attribute settings (default []).
            link_attrs_setting (list): Link attribute settings (default []).
            **kwargs: Additional keyword arguments.
        """
        super(PhysicalNetwork, self).__init__(incoming_graph_data, node_attrs_setting, link_attrs_setting, **kwargs)

    def generate_topology(self, num_nodes: int, type: str = 'waxman', **kwargs):
        """
        Generate a topology for the network.

        Args:
            num_nodes (int): The number of nodes in the network.
            type (str, optional): The type of network to generate. Defaults to 'waxman'.
            **kwargs: Keyword arguments to pass to the network generator.
        """
        super().generate_topology(num_nodes, type, **kwargs)
        self.degree_benchmark = self.get_degree_benchmark()

    def generate_attrs_data(self, node: bool = True, link: bool = True) -> None:
        """
        Generate attribute data for the network.

        Args:
            node (bool, optional): Whether or not to generate node attribute data. Defaults to True.
            link (bool, optional): Whether or not to generate link attribute data. Defaults to True.
        """
        """
        为网络生成对应属性的值
        Args：
        node（bool，可选）：是否生成节点属性数据。默认为True。
        link（bool，可选）：是否生成链接属性数据。默认为True。
        """
        super().generate_attrs_data(node, link) # 给节点属性赋值
        if node: 
            self.node_attr_benchmarks = self.get_node_attr_benchmarks()
        if link:
            self.link_attr_benchmarks = self.get_link_attr_benchmarks()
            self.link_sum_attr_benchmarks = self.get_link_sum_attr_benchmarks()

    @staticmethod
    def from_setting(setting: dict, seed: int = None) -> 'PhysicalNetwork':
        """
        Create a PhysicalNetwork object from the given setting.

        Args:
            setting (dict): The network settings.
            seed (int): The random seed for network generation.

        Returns:
            PhysicalNetwork: A PhysicalNetwork object.
        """
        setting = copy.deepcopy(setting)
        node_attrs_setting = setting.pop('node_attrs_setting')
        link_attrs_setting = setting.pop('link_attrs_setting')
        # load topology from file 否则重新生成一个
        try:
            if 'file_path' not in setting['topology']:
                raise Exception
            file_path = setting['topology'].get('file_path')
            net = PhysicalNetwork(node_attrs_setting=node_attrs_setting, link_attrs_setting=link_attrs_setting, **setting)
            G = nx.read_gml(file_path, label='id')
            net.__dict__['graph'].update(G.__dict__['graph'])
            net.__dict__['_node'] = G.__dict__['_node']
            net.__dict__['_adj'] = G.__dict__['_adj']
            n_attr_names = net.nodes[list(net.nodes)[0]].keys()
            for n_attr_name in n_attr_names:
                if n_attr_name not in net.node_attrs.keys():
                    net.node_attrs[n_attr_name] = NodeInfoAttribute(n_attr_name)
            l_attr_names = net.links[list(net.links)[0]].keys()
            for l_attr_name in l_attr_names:
                if l_attr_name not in net.link_attrs.keys():
                    net.link_attrs[l_attr_name] = LinkInfoAttribute(l_attr_name)
            print(f'Loaded the topology from {file_path}')
        except Exception as e:
            num_nodes = setting.pop('num_nodes')
            net = PhysicalNetwork(node_attrs_setting=node_attrs_setting, link_attrs_setting=link_attrs_setting, **setting)
            topology_setting = setting.pop('topology')
            # topology_type = topology_setting.pop('type')
            net.generate_topology(num_nodes, **topology_setting)
        if seed is None:
            seed = setting.get('seed')
        random.seed(seed)
        np.random.seed(seed)

        net.generate_attrs_data()
        return net

    def init_network_trust(self, trust_security_weight=0.6,
                           trust_feedback_weight=0.4, security_max_level=6.0):
        """Reset the per-run dynamic trust state.

        Trust combines normalized static security with a Laplace-smoothed
        per-node SFC success rate.  Counters are intentionally reset here so
        independent seeds and environment resets never share feedback.
        """
        if trust_security_weight < 0 or trust_feedback_weight < 0:
            raise ValueError('Trust weights must be non-negative.')
        if abs(trust_security_weight + trust_feedback_weight - 1.0) > 1e-8:
            raise ValueError('Trust weights must sum to one.')
        if security_max_level <= 0:
            raise ValueError('security_max_level must be positive.')

        self.trust_security_weight = trust_security_weight
        self.trust_feedback_weight = trust_feedback_weight
        self.security_max_level = security_max_level
        for node_id in self.nodes:
            node = self.nodes[node_id]
            node['trust_attempt_count'] = 0
            node['trust_success_count'] = 0
            self._refresh_trust(node_id)

    def _refresh_trust(self, node_id):
        node = self.nodes[node_id]
        attempts = node.get('trust_attempt_count', 0)
        successes = node.get('trust_success_count', 0)
        static_security = node.get('security', 0.0) / self.security_max_level
        static_security = min(1.0, max(0.0, static_security))
        # Laplace smoothing gives untried nodes a neutral feedback value 0.5.
        feedback = (successes + 1.0) / (attempts + 2.0)
        node['trust'] = (
            self.trust_security_weight * static_security
            + self.trust_feedback_weight * feedback
        )
        return node['trust']

    def update_get_trust(self, node_id, success_count=None):
        """Compatibility refresh used by legacy controller deployment code.

        This method deliberately does not mutate feedback counters.  The
        environment records a complete SFC attempt after the SRSR snapshot,
        which prevents a request from contributing its own success outcome to
        the trust value used to evaluate that request.
        """
        return self._refresh_trust(node_id)

    def record_trust_attempt(self, node_ids, succeeded):
        """Record one finished SFC attempt for each distinct selected node."""
        for node_id in set(node_ids):
            node = self.nodes[node_id]
            node['trust_attempt_count'] = node.get('trust_attempt_count', 0) + 1
            node['trust_success_count'] = (
                node.get('trust_success_count', 0) + int(bool(succeeded))
            )
            self._refresh_trust(node_id)


    @staticmethod
    def from_topology_zoo_setting(topology_zoo_setting: dict, seed: int = None) -> 'PhysicalNetwork':
        """
        Create a PhysicalNetwork object from a topology zoo setting.

        Args:
            topology_zoo_setting (dict): A dictionary containing the setting for the physical network.
            seed (int): An optional integer value to seed the random number generators.

        Returns:
            net (PhysicalNetwork): A PhysicalNetwork object representing the physical network.
        """
        setting = copy.deepcopy(topology_zoo_setting)
        node_attrs_setting = setting.pop('node_attrs_setting')
        link_attrs_setting = setting.pop('link_attrs_setting')
        file_path = setting.pop('file_path')

        net = PhysicalNetwork(node_attrs_setting=node_attrs_setting, link_attrs_setting=link_attrs_setting, **setting)
        G = nx.read_gml(file_path, label='id')
        net.__dict__['graph'].update(G.__dict__['graph'])
        net.__dict__['_node'] = G.__dict__['_node']
        net.__dict__['_adj'] = G.__dict__['_adj']
        n_attr_names = net.nodes[list(net.nodes)[0]].keys()
        for n_attr_name in n_attr_names:
            if n_attr_name not in net.node_attrs.keys():
                net.node_attrs[n_attr_name] = NodeInfoAttribute(n_attr_name)
        l_attr_names = net.links[list(net.links)[0]].keys()
        for l_attr_name in l_attr_names:
            if l_attr_name not in net.link_attrs.keys():
                net.link_attrs[l_attr_name] = LinkInfoAttribute(l_attr_name)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        net.generate_attrs_data()
        return net

    def save_dataset(self, dataset_dir: str) -> None:
        """
        Save the physical network dataset to a directory.

        Args:
            dataset_dir (str): The path to the directory where the physical network dataset is to be saved.
            dataset_dir（str）：物理网络数据集保存目录的路径。
        """
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)
        file_path = os.path.join(dataset_dir, 'p_net.gml')
        self.to_gml(file_path)

    @staticmethod
    def load_dataset(dataset_dir: str) -> 'PhysicalNetwork':
        """
        Load the physical network dataset from a directory.

        Args:
            dataset_dir (str): The path to the directory where the physical network dataset is stored.
        """
        if not os.path.exists(dataset_dir):
            raise ValueError(f'Find no dataset in {dataset_dir}.\nPlease firstly generating it.')
        file_path = os.path.join(dataset_dir, 'p_net.gml')
        p_net = PhysicalNetwork.from_gml(file_path)
        # get benchmark for normalization
        p_net.degree_benchmark = p_net.get_degree_benchmark()
        p_net.node_attr_benchmarks = p_net.get_node_attr_benchmarks()
        p_net.link_attr_benchmarks = p_net.get_link_attr_benchmarks()
        p_net.link_sum_attr_benchmarks = p_net.get_link_sum_attr_benchmarks()
        return p_net
