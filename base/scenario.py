import os
import tqdm
import pprint

from .controller import Controller
from .recorder import Recorder
from .counter import Counter
from .solution import Solution
from config import show_config, save_config
from data.physical_network import PhysicalNetwork
from data.virtual_network_request_simulator import VirtualNetworkRequestSimulator
from utils import get_p_net_dataset_dir_from_setting
from .environment import clone_network


class Scenario:

    def __init__(self, env, solver, config):
        self.env = env
        self.solver = solver
        self.config = config
        self.verbose = config.verbose

    @classmethod
    def from_config(cls, Env, Solver, config):
        # Create basic class: controller, recorder, counter, recorder
        # 创建基本类：控制器、记录器、计数器、记录器
        counter = Counter(config.v_sim_setting['node_attrs_setting'], 
                          config.v_sim_setting['link_attrs_setting'], 
                          **vars(config))
        controller = Controller(config.v_sim_setting['node_attrs_setting'], 
                                config.v_sim_setting['link_attrs_setting'], 
                                **vars(config))
        recorder = Recorder(counter, **vars(config))


        # Create p_net and v_net simulator
        if config.p_net_dataset_path and not os.path.exists(config.p_net_dataset_path):
            raise FileNotFoundError(
                f'Frozen physical-network dataset not found: {config.p_net_dataset_path}'
            )
        config.p_net_dataset_dir = (config.p_net_dataset_path or
                                    get_p_net_dataset_dir_from_setting(config.p_net_setting))
        if os.path.exists(config.p_net_dataset_dir):
            p_net = PhysicalNetwork.load_dataset(config.p_net_dataset_dir)
            p_net.init_network_trust(
                trust_security_weight=config.trust_security_weight,
                trust_feedback_weight=config.trust_feedback_weight,
                security_max_level=config.security_max_level,
            )

            print(f'*** Load Physical Network from {config.p_net_dataset_dir} *** ') if config.verbose >= 1 else None
        else:
            p_net = PhysicalNetwork.from_setting(config.p_net_setting)
            p_net.init_network_trust(
                trust_security_weight=config.trust_security_weight,
                trust_feedback_weight=config.trust_feedback_weight,
                security_max_level=config.security_max_level,
            )

            print(f'*** Generate Physical Network from setting *** ')
        v_net_simulator = VirtualNetworkRequestSimulator.from_setting(config.v_sim_setting)
        print(f'*** Create VNR Simulator from setting *** ') if config.verbose >= 1 else None


        # create env and solver
        env = Env(p_net, v_net_simulator, controller, recorder, counter, **vars(config))
        solver = Solver(controller, recorder, counter, **vars(config))


        # Create scenario
        # 实例化cls代表的类 即 Scenario
        scenario = cls(env, solver, config)
        # Validation has its own frozen request trajectory and isolated
        # recorder/counter.  It shares the immutable physical topology but is
        # never allowed to update the policy.
        scenario.validation_env = None
        if config.validation_v_nets_dataset_path:
            validation_kwargs = vars(config).copy()
            validation_kwargs['v_nets_dataset_path'] = config.validation_v_nets_dataset_path
            validation_kwargs['renew_v_net_simulator'] = False
            validation_kwargs['if_save_records'] = False
            validation_kwargs['if_temp_save_records'] = False
            validation_output_dir = os.path.join(config.save_dir, 'validation')
            validation_kwargs['save_dir'] = validation_output_dir
            validation_kwargs['summary_dir'] = validation_output_dir
            validation_counter = Counter(config.v_sim_setting['node_attrs_setting'],
                                         config.v_sim_setting['link_attrs_setting'],
                                         **validation_kwargs)
            validation_recorder = Recorder(validation_counter, **validation_kwargs)
            validation_simulator = VirtualNetworkRequestSimulator.from_setting(config.v_sim_setting)
            scenario.validation_env = Env(clone_network(p_net), validation_simulator,
                                          controller, validation_recorder,
                                          validation_counter, **validation_kwargs)
        if config.verbose >= 2: show_config(config)
        if config.if_save_config: save_config(config)  # print(Save config in save\gat_seq2seq_ppo\LAPTOP-M43DF0FU\config.yaml)

        return scenario

    def reset(self):
        pass

    def ready(self):
        # load pretrained model
        if hasattr(self.solver, 'load_model') and self.config.pretrained_model_path not in ['None', '']:
            if os.path.exists(self.config.pretrained_model_path):
                self.solver.load_model(self.config.pretrained_model_path)
            else:
                print(f'Load pretrained model failed: Path does not exist {self.config.pretrained_model_path}')
        # execute pretrain
        elif hasattr(self.solver, 'learn') and self.config.num_train_epochs > 0:
            print(f"\n{'-' * 30}  Pretrain  {'-' * 30}\n")
            self.solver.learn(self.env, num_epochs=self.config.num_train_epochs,
                              validation_env=self.validation_env,
                              validation_interval=self.config.validation_interval)
            # Testing after training must evaluate the held-out-validation
            # winner, never merely the final epoch.
            best_checkpoint = os.path.join(getattr(self.solver, 'model_dir', ''), 'best_model.pkl')
            if self.validation_env is not None and os.path.exists(best_checkpoint):
                self.solver.load_model(best_checkpoint)
                print(f'Loaded validation-selected checkpoint: {best_checkpoint}')
            print(f"\n{'-' * 30}    Done    {'-' * 30}\n")
        # set eval mode
        if hasattr(self.solver, 'eval'):
            self.solver.eval()

class BasicScenario(Scenario):

    def __init__(self, env, solver, config):
        super(BasicScenario, self).__init__(env, solver, config)

    def run(self):
        self.ready()
        print("ready 结束")
        if self.config.train_only:
            print('Training-only run complete; post-training test loop skipped.')
            return

        metrics = {
            'acceptance_rate': [],
            'long_term_aver_time_revenue': [],
            'long_term_aver_r2c_ratio': [],
        }

        for epoch_id in range(self.config.start_epoch, self.config.start_epoch + self.config.num_epochs):
            print(f'\nEpoch {epoch_id}') if self.verbose >= 2 else None
            instance = self.env.reset()

            pbar = tqdm.tqdm(desc=f'Running with {self.config.solver_name} in epoch {epoch_id}', total=self.env.num_v_nets) if self.verbose <= 1 else None
            v_nets_i = 0
            while True:
                solution = self.solver.solve(instance)
                next_instance, _, done, info = self.env.step(solution)
                if pbar is not None: 
                    pbar.update(1)
                    pbar.set_postfix({
                        'ac': f'{info["success_count"] / info["v_net_count"]:1.2f}',
                        'r2c': f'{info["total_r2c"]:1.2f}',
                        'a_r': f'{info["long_term_aver_time_revenue"]:1.2f}',
                        'inservice': f'{info["inservice_count"]:05d}',
                    })
                # 保存epoch指标
                metrics['acceptance_rate'].append(info["success_count"] / info["v_net_count"])
                metrics['long_term_aver_time_revenue'].append(info['long_term_aver_time_revenue'])
                metrics['long_term_aver_r2c_ratio'].append(info['long_term_aver_r2c_ratio'])
                self.env.recorder.save_metrics(metrics, v_nets_i, 'run')  # 保存 metrics
                v_nets_i += 1
                if done:
                    break
                instance = next_instance

            if pbar is not None: pbar.close()
            summary_info = self.env.summary_records()


            if self.verbose == 0:
                pprint.pprint(summary_info)
