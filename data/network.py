import copy
import numpy as np
import networkx as nx
import random
from typing import *
from functools import cached_property, lru_cache

from networkx.classes.reportviews import DegreeView, EdgeView, NodeView
from networkx.classes.filters import no_filter

from utils import write_setting
from .attribute import Attribute


class Network(nx.Graph):
    """
    Network class inherited from networkx.Graph.

    Attributes:
        node_attrs: Node attributes.
        link_attrs: Link attributes.
        graph_attrs: Graph attributes.
    
    Methods:
        create_attrs_from_setting: Create node and link attribute dictionaries from their respective settings.
        init_graph_attrs: Initialize graph attributes.
        set_graph_attrs_data: Set graph attributes data.
        get_graph_attrs_data: Get graph attributes data.
        get_node_attrs: Get node attributes.
        get_link_attrs: Get link attributes.
        get_node_attrs_data: Get node attributes data.
        get_link_attrs_data: Get link attributes data.
        get_node_attr_data: Get node attribute data.
        get_link_attr_data: Get link attribute data.
        set_node_attrs_data: Set node attributes data.
        set_link_attrs_data: Set link attributes data.
        set_node_attr_data: Set node attribute data.
        set_link_attr_data: Set link attribute data.
        generate_topology: Generate the network topology.
        check_attrs_existence: Check if all defined attributes exist in the graph.
        write_setting: Write network setting to file.
    """
    """
        继承自 networkx.Graph.Network 的网络类。

        属性：
            node_attrs： 节点属性。
            link_attrs： 链接属性。
            graph_attrs： 图形属性。
        
        方法：
            create_attrs_from_setting： 根据各自的设置创建节点和链接属性字典。
            init_graph_attrs： 初始化图形属性。
            set_graph_attrs_data：设置图形属性数据： 设置图形属性数据。
            get_graph_attrs_data：获取图形属性数据： 获取图形属性数据。
            get_node_attrs： 获取节点属性。
            get_link_attrs： 获取链接属性。
            get_node_attrs_data: 获取节点属性数据： 获取节点属性数据。
            get_link_attrs_data： 获取链接属性数据。
            get_node_attr_data：获取节点属性数据： 获取节点属性数据。
            get_link_attr_data：获取节点属性数据： 获取链接属性数据。
            set_node_attrs_data: 设置节点属性数据： 设置节点属性数据。
            set_link_attrs_data： 设置链接属性数据。
            set_node_attr_data: 设置节点属性数据： 设置节点属性数据。
            set_link_attr_data: 设置链接属性数据： 设置链接属性数据。
            generate_topology： 生成网络拓扑结构。
            check_attrs_existence（检查属性是否存在 检查图中是否存在所有已定义的属性。
            write_setting（写入设置 将网络设置写入文件。
    """
    def __init__(self, incoming_graph_data: Optional[nx.Graph] =None, node_attrs_setting: list = [], link_attrs_setting: list = [], **kwargs):
        """
        Initializes a new Network instance.

        Args:
            incoming_graph_data (optional, Graph): Graph instance to convert to Network or data to initialize graph. Default: None.
            node_attrs_setting (list): List of dictionaries containing node attribute settings. Default: [].
            link_attrs_setting (list): List of dictionaries containing link attribute settings. Default: [].
            **kwargs: Additional keyword arguments to set graph attributes.
        """
        """
            初始化一个新的网络实例。

            参数：
                incoming_graph_data （可选，Graph）： 要转换为网络的图形实例或初始化图形的数据。默认值： 无。
                node_attrs_setting（列表）： 包含节点属性设置的字典列表。默认值：[]。
                link_attrs_setting （列表）： 包含链接属性设置的字典列表： 包含链接属性设置的字典列表。默认值： [].
                **kwargs： 用于设置图形属性的附加关键字参数。
        """
        super(Network, self).__init__(incoming_graph_data)
        self.init_graph_attrs()
        # set graph attributes
        self.graph['node_attrs_setting'] += node_attrs_setting
        self.graph['link_attrs_setting'] += link_attrs_setting
        self.create_attrs_from_setting()
        # Read extra kwargs
        # 赋予每个SFC之前产生的 id arrival_time lifetime值
        self.set_graph_attrs_data(kwargs)

    def create_attrs_from_setting(self):
        """Create node and link attribute dictionaries from their respective settings."""
        """根据各自的设置创建节点和链接属性字典"""
        self.node_attrs = {n_attr_dict['name']: Attribute.from_dict(n_attr_dict) for n_attr_dict in self.graph['node_attrs_setting']}
        self.link_attrs = {e_attr_dict['name']: Attribute.from_dict(e_attr_dict) for e_attr_dict in self.graph['link_attrs_setting']}

    def check_attrs_existence(self):
        """Check if all defined attributes exist in the graph."""
        # check node attrs
        for n_attr_name in self.node_attrs.keys():
            assert n_attr_name in self.nodes[list(self.nodes)[0]].keys(), f'{n_attr_name}'
        # check link attrs
        for l_attr_name in self.link_attrs.keys():
            assert l_attr_name in self.links[list(self.links)[0]].keys(), f'{l_attr_name}'

    ### Generate ###
    def generate_topology(self, num_nodes, type='path', **kwargs):
        """
        Generate the network topology.

        Args:
            num_nodes: the number of nodes in the generated graph
            type: the type of graph to generate, currently supported types are: 'path', 'star', 'waxman', and 'random'
            **kwargs: additional keyword arguments required for certain graph types
        """
        """
        生成网络拓扑
        Args：
        num_nodes：生成的图中的节点数
        type：要生成的图形类型，目前支持的类型有：“path”、“star”、“waxman”和“random”
        **kwargs：某些图类型需要额外的关键字参数
        """
        assert num_nodes >= 1
        assert type in ['path', 'star', 'waxman', 'random'], ValueError('Unsupported graph type!')
        self.set_graph_attrs_data({'num_nodes': num_nodes, 'type': type})

        if type == 'path':  # 虚拟链路的type类型
            G = nx.path_graph(num_nodes)

        elif type == 'star':
            G = nx.star_graph(num_nodes)
        elif type == 'grid_2d':
            wm_alpha = kwargs.get('m')
            wm_beta = kwargs.get('n')
            G = nx.grid_2d_graph(num_nodes, periodic=False)


        elif type == 'waxman':
            wm_alpha = kwargs.get('wm_alpha', 0.5)
            wm_beta = kwargs.get('wm_beta', 0.2)
            num_edges = kwargs.get('num_edges')
            not_connected = True
            while not_connected:
                G = nx.waxman_graph(num_nodes, wm_alpha, wm_beta)
                not_connected = not nx.is_connected(G)

            # 获取当前边数
            current_edges = G.number_of_edges()
            # 调整边数
            if num_edges is not None:
                if current_edges < num_edges:
                        # 需要添加边
                        non_edges = list(nx.non_edges(G))
                        random.shuffle(non_edges)
                        for u, v in non_edges[:num_edges - current_edges]:
                            G.add_edge(u, v)
                elif current_edges > num_edges:
                    # 需要删除边，但要确保图保持连通
                    edges = list(G.edges())
                    random.shuffle(edges)
                    edges_to_remove = current_edges - num_edges
                    removed = 0

                    for u, v in edges:
                        if removed >= edges_to_remove:
                            break
                        # 检查删除这条边后图是否仍然连通
                        G.remove_edge(u, v)
                        if nx.is_connected(G):
                            removed += 1
                        else:
                            # 如果不连通，恢复这条边
                            G.add_edge(u, v)
            self.set_graph_attrs_data({'wm_alpha': wm_alpha, 'wm_beta': wm_beta})


        elif type == 'random':
            random_prob = kwargs.get('random_prob', 0.5)
            self.set_graph_attrs_data({'random_prob': random_prob})
            G = nx.erdos_renyi_graph(num_nodes, random_prob, directed=False)
            not_connected = True
            while not_connected:
                G = nx.erdos_renyi_graph(num_nodes, random_prob, directed=False)
                not_connected = not nx.is_connected(G)
        else:
            raise NotImplementedError
        self.__dict__['_node'] = G.__dict__['_node']
        self.__dict__['_adj'] = G.__dict__['_adj']

    def generate_attrs_data(self, node=True, link=True):
        """Generate the data of network attributes based on attributes."""
        """" 根据属性生成网络属性数据 """
        if node:
            for n_attr in self.node_attrs.values():
                # 一轮处理SFC中所有VNF的一个属性
                if n_attr.generative or n_attr.type == 'extrema':
                    # 对于虚拟网络，为每个SFC赋值
                    attribute_data = n_attr.generate_data(self)
                    n_attr.set_data(self, attribute_data)
        if link:
            for l_attr in self.link_attrs.values():
                if l_attr.generative or l_attr.type == 'extrema':
                    attribute_data = l_attr.generate_data(self)
                    l_attr.set_data(self, attribute_data)

    ### Number ###
    @cached_property
    def num_nodes(self):
        """Get the number of nodes."""
        return self.number_of_nodes()
    
    @cached_property
    def num_links(self):
        """Get the number of links."""
        return self.number_of_edges()

    @cached_property
    def links(self):
        """Get the number of links."""
        return EdgeView(self)

    @property
    def adjacency_matrix(self):
        """Get the adjacency matrix of Network."""
        return nx.to_scipy_sparse_matrix(self, format='csr')

    ### Get Attribute ###
    def get_graph_attrs(self, names):
        """
        Get the attributes of the network.

        Args:
            names (str): The names of the attributes to retrieve. If None, return all attributes.

        Returns:
            dict: A dictionary of network attributes.
        """
        if names is None: return self.graph
        return {attr: self.graph[attr] for attr in names}

    def get_node_attrs(self, types=None, names=None):
        """
        Get the node attributes of the network.

        Args:
            types (str): The types of the node attributes to retrieve. If None, return all attributes.
            names (str): The names of the node attributes to retrieve. If None, return all attributes.

        Returns:
            list: A list of node attributes.
        """
        if types is None and names is None:
            return list(self.node_attrs.values())
        elif types is not None:
            selected_node_attrs = []
            for n_attr in self.node_attrs.values():
                selected_node_attrs.append(n_attr) if n_attr.type in types else None
        elif names is not None:
            selected_node_attrs = []
            for n_attr in self.node_attrs.values():
                selected_node_attrs.append(n_attr) if n_attr.name in names else None
        return selected_node_attrs

    def get_link_attrs(self, types=None, names=None):
        """Get the link attributes of the network.

        Args:
            types (str): The types of the link attributes to retrieve. If None, return all attributes.
            names (str): The names of the link attributes to retrieve. If None, return all attributes.

        Returns:
            list: A list of link attributes.
        """
        if types is None and names is None: 
            return list(self.link_attrs.values())
        elif types is not None:  
            selected_link_attrs = []
            for l_attr in self.link_attrs.values():
                selected_link_attrs.append(l_attr) if l_attr.type in types else None
        elif names is not None:  
            selected_link_attrs = []
            for l_attr in self.link_attrs.values():
                selected_link_attrs.append(l_attr) if l_attr.name in names else None
        return selected_link_attrs

    ### Set Data ### 
    def init_graph_attrs(self):
        """Initialize the graph attributes."""
        for key in ['node_attrs_setting', 'link_attrs_setting']:
            if key not in self.graph:
                self.graph[key] = []
        for key, value in self.graph.items():
            if key not in ['num_nodes']:
                self[key] = value


    def set_graph_attribute(self, name, value):
        """Set a graph attribute `attr` to `value`."""
        if name in ['num_nodes']:
            self.graph[name] = value
            return
        self.graph[name] = value
        self[name] = value

    def set_graph_attrs_data(self, attributes_data):
        """Set the data of graph attributes."""
        for key, value in attributes_data.items():
            self.set_graph_attribute(key, value)

    def set_node_attrs_data(self, node_attributes_data):
        """Set the data of node attributes."""
        for n_attr, data in node_attributes_data.items():
            n_attr.set_data(self, data)

    def set_link_attrs_data(self, link_attributes_data):
        """Set the data of link attributes."""
        for l_attr, data in link_attributes_data.items():
            l_attr.set_data(self, data)

    ### Get Data ###
    def get_node_attrs_data(self, node_attrs):
        """Get the data of node attributes."""
        if isinstance(node_attrs[0], str):
            node_attrs_data = [list(nx.get_node_attributes(self, n_attr_name).values()) for n_attr_name in node_attrs]
        else:
            node_attrs_data = [n_attr.get_data(self) for n_attr in node_attrs]
        return node_attrs_data

    def get_link_attrs_data(self, link_attrs):
        """Get the data of link attributes."""
        if isinstance(link_attrs[0], str):
            link_attrs_data = [list(nx.get_edge_attributes(self, l_attr_name).values()) for l_attr_name in link_attrs]
        else:
            link_attrs_data = [l_attr.get_data(self) for l_attr in link_attrs]
        return link_attrs_data

    def get_adjacency_attrs_data(self, link_attrs, normalized=False):
        """Get the data of adjacency attributes."""
        adjacency_data = [l_attr.get_adjacency_data(self, normalized) for l_attr in link_attrs]
        return adjacency_data

    def get_aggregation_attrs_data(self, link_attrs, aggr='sum', normalized=False):
        aggregation_data = [l_attr.get_aggregation_data(self, aggr, normalized) for l_attr in link_attrs]
        return aggregation_data

    ### other ###
    def subgraph(self, nodes):
        subnet = super().subgraph(nodes)
        subnet.node_attrs = self.node_attrs
        subnet.link_attrs = self.link_attrs
        return subnet

    def subnetwork(self, nodes):
        return self.subgraph(nodes)

    def get_subgraph_view(self, filter_node=no_filter, filter_edge=no_filter):
        view = nx.subgraph_view(self, filter_node=filter_node, filter_edge=filter_edge)
        view.node_attrs = self.node_attrs
        view.link_attrs = self.link_attrs
        return view

    def get_subnetwork_view(self, filter_node=no_filter, filter_edge=no_filter):
        return self.get_subgraph_view(filter_node, filter_edge)

    ### Update ###
    def update_node_resources(self, node_id, v_net_node, method='+'):
        """Update (increase) the value of node atributes."""
        for n_attr in self.node_attrs.keys():
            if n_attr.type != 'resource':
                continue
            n_attr.update(self.nodes[node_id], v_net_node, method)

    def update_link_resources(self, link_pair, v_net_link, method='+'):
        """Update (increase) the value of link atributes."""
        for l_attr in self.link_attrs:
            if l_attr.type != 'resource':
                continue
            l_attr.update(self.links[link_pair], v_net_link, method)

    def update_path_resources(self, path, v_net_link, method='+'):
        """Update (increase) the value of links atributes of path with the same increments."""
        assert len(path) >= 1
        for l_attr in self.link_attrs:
            l_attr.update_path(self, path, v_net_link, method)

    ### Benchmark ###
    def _get_attr_benchmarks(self, attr_types: list, attrs_list: list, attr_data: np.ndarray) -> dict:
        """Get attributes benchmark for normlization

        Args:
            attr_types: An list of specified types of attributes.
            attrs_list: An list of specified attributes.
            attr_data: The data of these attributes.

        Returns:
            attr_benchmarks: A dict like {attr_name: attr_benchmark}
        """
        """
            获取用于规范化的属性基准    从attr_data中取出最大的数  以此标准化（attr_data例如[3, 100] ）
            Args：
            attr_types：指定属性类型的列表。
            attrs_list：指定属性的列表。
            attr_data：这些属性的数据。
            Returns：
            attr_benchmarks：一个类似于{attr_name: attr_benchmark}的字典
        """
        attr_benchmarks = {}
        if 'extrema' in attr_types:
            for attr, attr_data in zip(attrs_list, attr_data):
                # for resource attributes, the maximum of extrema attributes are used as its benchmark
                # 对于resource属性，极值属性的最大值被用作其基准
                if attr.type == 'resource':
                    continue
                elif attr.type == 'extrema':
                    attr_benchmarks[attr.originator] = attr_data.max()
                else:
                    attr_benchmarks[attr.name] = attr_data.max()
        else:
            for attr, attr_data in zip(attrs_list, attr_data):
                attr_benchmarks[attr.name] = attr_data.max()
        return attr_benchmarks

    @lru_cache
    def get_degree_benchmark(self):
        return max(list(dict(self.degree).values()))

    @lru_cache
    def get_node_attr_benchmarks(self, node_attr_types: list = ['resource', 'extrema','security']):
        if node_attr_types is None:
            n_attrs = self.get_node_attrs()
            node_attr_types = [n_attr.type for n_attr in n_attrs]
        else:
            n_attrs = self.get_node_attrs(node_attr_types)
        n_attrs = self.get_node_attrs(node_attr_types)
        node_data = np.array(self.get_node_attrs_data(n_attrs), dtype=np.float32)
        node_attr_benchmarks = self._get_attr_benchmarks(node_attr_types, n_attrs, node_data)
        return node_attr_benchmarks

    @lru_cache
    def get_link_attr_benchmarks(self, link_attr_types=['resource', 'extrema','security']):
        if link_attr_types is None:
            l_attrs = self.get_link_attrs()
            link_attr_types = [l_attr.type for l_attr in l_attrs]
        else:
            l_attrs = self.get_link_attrs(link_attr_types)
        link_data = np.array(self.get_link_attrs_data(l_attrs), dtype=np.float32)
        link_data = np.concatenate([link_data, link_data], axis=1)
        link_attr_benchmarks = self._get_attr_benchmarks(link_attr_types, l_attrs, link_data)
        return link_attr_benchmarks

    @lru_cache
    def get_link_sum_attr_benchmarks(self, link_attr_types=['resource', 'extrema','security']):
        if link_attr_types is None:
            l_attrs = self.get_link_attrs()
            link_attr_types = [l_attr.type for l_attr in l_attrs]
        else:
            l_attrs = self.get_link_attrs(link_attr_types)
        # link_sum_attrs_data : [3, 100]  bw  max_bw  security
        link_sum_attrs_data = np.array(self.get_aggregation_attrs_data(l_attrs, aggr='sum'), dtype=np.float32)

        link_sum_attr_benchmarks = self._get_attr_benchmarks(link_attr_types, l_attrs, link_sum_attrs_data)

        return link_sum_attr_benchmarks


    ### Internal ###
    def __getitem__(self, key):
        """Gets the data of the attribute key."""
        if isinstance(key, int):
            return super().__getitem__(key)
        elif isinstance(key, str):
            return getattr(self, key, None)
        else:
            return TypeError

    # def __repr__(self):
    #     info = [f"{key}={self._size_repr(item)}" for key, item in self]
    #     return f"{self.__class__.__name__}({', '.join(info)})"

    def __setitem__(self, key: str, value):
        """Sets the attribute key to value."""
        setattr(self, key, value)

    def clone(self):
        return self.__class__.from_dict({
            k: copy.deepcopy(v)
            for k, v in self.__dict__.items()
        })

    def to_gml(self, fpath):
        nx.write_gml(self, fpath)

    @classmethod
    def from_gml(cls, fpath):
        gml_net = nx.read_gml(fpath, destringizer=int)
        net = cls(incoming_graph_data=gml_net)
        net.check_attrs_existence()
        return net

    def save_attrs_dict(self, fpath):
        attrs_dict = {
            'graph_attrs_dict': self.get_graph_attrs(),
            'node_attrs': [n_attr.to_dict() for n_attr in self.node_attrs.values()],
            'link_attrs': [l_attr.to_dict() for l_attr in self.link_attrs.values()]
        }
        write_setting(attrs_dict, fpath)


if __name__ == '__main__':
    pass
