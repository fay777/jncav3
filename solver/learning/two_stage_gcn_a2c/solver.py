"""Two-stage GCN-A2C baseline adapted from Chen et al. (2023)."""

from base import SolutionStepEnvironment
from solver import registry
from ..a2c_gcn_seq2seq.solver import A2CGcnSeq2SeqSolver
from .sub_env import TwoStageSubEnv


@registry.register(
    solver_name='two_stage_gcn_a2c_adapted',
    env_cls=SolutionStepEnvironment,
    solver_type='r_learning',
)
class TwoStageGCNA2CAdaptedSolver(A2CGcnSeq2SeqSolver):
    """Chen et al. (2023)-inspired two-stage GCN--DRL baseline.

    The published framework targets multi-datacenter networks.  In this common
    single-substrate setting, stage 1 is a GCN-A2C node-placement policy and
    stage 2 maps all virtual links through the shared feasible-path Controller.
    This preserves the two-stage placement/routing decomposition while avoiding
    unmodelled datacenter-level decisions; it is therefore an adapted baseline,
    not a claimed direct reproduction.
    """

    def __init__(self, controller, recorder, counter, **kwargs):
        super().__init__(controller, recorder, counter, **kwargs)
        self.SubEnv = TwoStageSubEnv
