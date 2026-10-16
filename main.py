import os
import random
import numpy as np
import torch
from config import get_config, show_config, save_config, load_config
from data import Generator
from base import BasicScenario
from solver import REGISTRY


def run(config):
    print(f"\n{'-' * 30}    Start     {'-' * 30}\n")
    # Load solver info: environment and solver class
    # 负载求解器信息：环境和求解器类
    solver_info = REGISTRY.get(config.solver_name)    # {'solver': <class 'solver.learning.a3c_gcn_seq2seq.solver.A3CGcnSeq2SeqSolver'>, 'env': <class 'base.environment.SolutionStepEnvironment'>, 'type': 'r_learning'}
    Env, Solver = solver_info['env'], solver_info['solver']
    print(f'Use {config.solver_name} Solver (Type = {solver_info["type"]})...\n')
    print(Env, Solver)
    scenario = BasicScenario.from_config(Env, Solver, config)
    scenario.run()

    print(f"\n{'-' * 30}   Complete   {'-' * 30}\n")


def set_global_seed(seed):
    """Seed Python, NumPy, and PyTorch before constructing data or models."""
    if seed is None:
        raise ValueError('A non-null --seed is required for reproducible experiments.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # Please refer to `base.loader` to obtain all available solvers

    # 1. Get Config
    # The key settings are controlled with config.py
    # while other advanced settings are listed in settings/*.yaml
    # 按键设置由config.py控制
    # 而其他高级设置列在settings/*.yaml中
    config = get_config()
    set_global_seed(config.seed)
    # The solver and checkpoint path are supplied through the command line.
    # A training run should use --pretrained_model_path "".
    # 2. Generate Dataset
    # Although we do not generate a static dataset,
    # the environment will automatically produce a random dataset.
    # 虽然我们不生成静态数据集，
    # 环境将自动生成随机数据集。
    # p_net, v_net_simulator = Generator.generate_dataset(
    #     config,
    #     p_net=False,
    #     v_nets=False,
    #     save=False) # Here, no dataset will be generated and saved.

    # 3. Start to Run
    # A scenario with an environment and a solver will be create following provided config.
    # The interaction between the environment and the solver will happen in this scenario.
    # 将根据提供的配置创建具有环境和求解器的场景。
    # 在这种情况下，环境和求解器之间的交互将发生。
    run(config)
