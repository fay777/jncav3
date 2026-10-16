import numpy as np

from .network import Network


class VirtualNetwork(Network):
    """
    VirtualNetwork class is a subclass of Network class. It represents a virtual network.

    Methods:
        generate_topology: Generate a virtual network topology.
        total_node_resource_demand: Get the total resource demand of all nodes in the virtual network.
        total_link_resource_demand: Get the total resource demand of all links in the virtual network.
        total_resource_demand: Get the total resource demand of all nodes and links in the virtual network.
    """
    """
    VirtualNetwork类是Network类的一个子类。它代表了一个虚拟网络。
    Methods：
    generate_topology：生成虚拟网络拓扑。
    total_node_resource_demand：获取虚拟网络中所有节点的总资源需求。
    total_link_resource_demand：获取虚拟网络中所有链路的总资源需求。
    total_resource_demand：获取虚拟网络中所有节点和链路的总资源需求。
    """
    def __init__(self, incoming_graph_data=None, node_attrs_setting=[], link_attrs_setting=[], **kwargs):
        super(VirtualNetwork, self).__init__(incoming_graph_data, node_attrs_setting, link_attrs_setting, **kwargs)


    def generate_topology(self, num_nodes, type='random', **kwargs):
        return super().generate_topology(num_nodes, type=type, **kwargs)

    def set_ssi(self):
        v_net_sdl = self.get_node_attrs_data(self.get_node_attrs('security'))
        ssi = 0
        for i in range(len(v_net_sdl[0])):
            ssi += v_net_sdl[0][i]
        ssi /= len(v_net_sdl[0])
        self.ssi = ssi   # ssi 最大为6



    @property
    def total_node_resource_demand(self):
        n = np.array(self.get_node_attrs_data(self.get_node_attrs('resource'))).sum()
        return n

    @property
    def total_link_resource_demand(self):
        e = np.array(self.get_link_attrs_data(self.get_link_attrs('resource'))).sum()
        return e

    @property
    def total_resource_demand(self):
        n = np.array(self.get_node_attrs_data(self.get_node_attrs('resource'))).sum()
        e = np.array(self.get_link_attrs_data(self.get_link_attrs('resource'))).sum()
        return n + e

if __name__ == '__main__':
    pass