import os
import abc

from base import Controller, Recorder, Counter, Solution


class Solver:

    def __init__(self, controller: Controller, recorder: Recorder, counter: Counter, **kwargs):
        """
        Initialize a Solver object.

        Args:
            controller (Controller): The controller object.
            recorder (Recorder): The recorder object.
            counter (Counter): The counter object.
            **kwargs (dict): A dictionary containing the optional arguments:
                reusable (bool): Whether to reuse resources (default: False).
                verbose (int): Verbosity level (default: 1).
                save_dir (str): Directory to save output files (default: 'save').
                solver_name (str): Name of the solver (default: 'unknown_solver').
                run_id (str): ID of the run (default: 'unknown_host-unknown_time').
                node_ranking_method (str): Method used for ranking nodes (default: 'order').
                link_ranking_method (str): Method used for ranking links (default: 'order').
                matching_mathod (str): Method used for node mapping (default: 'greedy').
                shortest_method (str): Method used for link mapping (default: 'k_shortest').
                k_shortest (int): Number of shortest paths to consider (default: 10).
                allow_rejection (bool): Whether to allow v-nets to be rejected (default: False).
                allow_revocable (bool): Whether to allow v-nets to be revoked (default: False).
        """
        __metaclass__ = abc.ABCMeta
        self.controller = controller
        self.recorder = recorder
        self.counter = counter
        self.reusable = kwargs.get('reusable', False)
        self.verbose = kwargs.get('verbose', 1)
        self.num_arrived_v_nets = 0
        save_dir = kwargs.get('save_dir', 'save')
        solver_name = kwargs.get('solver_name', 'unknown_solver')
        run_id = kwargs.get('run_id', 'unknown_host-unknown_time')
        self.save_dir = os.path.join(save_dir, solver_name, run_id)
        # ranking strategy
        self.node_rank = None
        self.link_rank = None
        self.node_ranking_method = kwargs.get('node_ranking_method', 'order')
        self.link_ranking_method = kwargs.get('link_ranking_method', 'order')
        # node mapping
        self.matching_mathod = kwargs.get('matching_mathod', 'greedy')
        # link mapping
        self.shortest_method = kwargs.get('shortest_method', 'k_shortest')
        self.k_shortest = kwargs.get('k_shortest', 10)
        # action
        self.allow_rejection = kwargs.get('allow_rejection', False)
        self.allow_revocable = kwargs.get('allow_revocable', False)
        # Environment-level reward/state controls.  They must be retained in
        # ``basic_config`` because every per-request SubEnv is constructed
        # from that dictionary rather than directly from the top-level CLI
        # configuration.
        self.normalize_ts = kwargs.get('normalize_ts', False)
        self.incremental_security_shaping = kwargs.get('incremental_security_shaping', False)
        self.include_trust_in_state = kwargs.get('include_trust_in_state', False)
        self.reward_failure_penalty = kwargs.get('reward_failure_penalty', 1.0)
        self.security_reward_weight = kwargs.get('security_reward_weight', -1.0)
        self.basic_config = {
            'reusable': self.reusable,
            'node_ranking_method': self.node_ranking_method,
            'link_ranking_method': self.link_ranking_method,
            'matching_mathod': self.matching_mathod,
            'shortest_method': self.shortest_method,
            'k_shortest': self.k_shortest,
            'allow_revocable': self.allow_revocable,
            'allow_rejection': self.allow_rejection,
            'normalize_ts': self.normalize_ts,
            'incremental_security_shaping': self.incremental_security_shaping,
            'include_trust_in_state': self.include_trust_in_state,
            'reward_failure_penalty': self.reward_failure_penalty,
            'security_reward_weight': self.security_reward_weight,
        }
        # self.env_info = dict(
        #     num_p_net_node_attrs = len(self.p_net.get_node_attrs(['resource'])), # , 'extrema'
        #     num_p_net_link_attrs = len(self.p_net.get_link_attrs(['resource'])), # , 'extrema'
        #     num_v_net_node_attrs = len(self.v_net_simulator.v_sim_setting['node_attrs_setting']),
        #     num_v_net_link_attrs = len(self.v_net_simulator.v_sim_setting['link_attrs_setting'])
        # )

    def solve(instance: dict) -> Solution:
        """
        Solves the problem instance, and returns the solution.

        Args:
            instance (dict): The problem instance to solve.

        Returns:
            Solution (Solution): The solution to the problem instance.
        """
        return NotImplementedError
