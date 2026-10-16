import random
import numpy as np

from .physical_network import PhysicalNetwork
from .virtual_network_request_simulator import VirtualNetworkRequestSimulator
from utils import get_p_net_dataset_dir_from_setting, get_v_nets_dataset_dir_from_setting


class Generator:

    # 定义类中的静态方法,本质上是普通函数，只是逻辑上属于某个类的命名空间
    @staticmethod
    def generate_dataset(config, p_net=True, v_nets=True, save=False):
        """
        Generate a dataset consisting of a physical network and a virtual network request simulator.

        Args:
            config (dict or object): Configuration object containing the settings for the generator.
            p_net (bool): Whether or not to generate a physical network dataset.
            v_nets (bool): Whether or not to generate a virtual network request simulator dataset.
            save (bool): Whether or not to save the generated datasets.

        Returns:
            Tuple: A tuple consisting of the generated physical network and virtual network request simulator.
        """


        physical_network = Generator.generate_p_net_dataset_from_config(config, save=save) if p_net else None
        v_net_simulator = Generator.generate_v_nets_dataset_from_config(config, save=save) if v_nets else None

        return physical_network, v_net_simulator

    @staticmethod
    def generate_p_net_dataset_from_config(config, save=False):
        """
        Generate a physical network dataset based on the given configuration.

        Args:
            config (dict or object): Configuration object containing the settings for the generator.
            save (bool): Whether or not to save the generated dataset.

        Returns:
            PhysicalNetwork: A PhysicalNetwork object representing the generated dataset.
        """
        if not isinstance(config, dict):
            config = vars(config)
        p_net_setting = config['p_net_setting']
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        
        p_net = PhysicalNetwork.from_setting(p_net_setting)

        if save:
            p_net_dataset_dir = get_p_net_dataset_dir_from_setting(p_net_setting)
            p_net.save_dataset(p_net_dataset_dir)
            if config.get('verbose', 1):
                print(f'save p_net dataset in {p_net_dataset_dir}')

        # new_p_net = PhysicalNetwork.load_dataset(p_net_dataset_dir)
        return p_net

    @staticmethod
    def generate_v_nets_dataset_from_config(config, save=False):
        """
        Generate a virtual network request simulator dataset based on the given configuration.

        Args:
            config (dict or object): Configuration object containing the settings for the generator.
            save (bool): Whether or not to save the generated dataset.

        Returns:
            VirtualNetworkRequestSimulator: A VirtualNetworkRequestSimulator object representing the generated dataset.
        """
        """
        基于给定的配置生成虚拟网络请求模拟器数据集。
        
        Args：
            config（dict或object）：包含生成器设置的配置对象。
            save（bool）：是否保存生成的数据集。
        Returns：
            VirtualNetworkRequestSimulator：表示生成的数据集的VirtualNetworkRequest模拟器对象。
        """
        # isinstance：判断一个对象是否是一个已知的类型，或者这个对象是否是一个类的实例
        if not isinstance(config, dict):
            config = vars(config)
        v_sim_setting = config['v_sim_setting']
        random.seed(config['seed'])
        np.random.seed(config['seed'])

        v_net_simulator = VirtualNetworkRequestSimulator.from_setting(v_sim_setting)
        v_net_simulator.renew()

        if save:
            v_nets_dataset_dir = get_v_nets_dataset_dir_from_setting(v_sim_setting)
            v_net_simulator.save_dataset(v_nets_dataset_dir)
            if config.get('verbose', 1):
                print(f'save v_net dataset in {v_nets_dataset_dir}')

        # new_v_net_simulator = VirtualNetworkRequestSimulator.load_dataset(v_nets_dataset_dir)
        return v_net_simulator
        