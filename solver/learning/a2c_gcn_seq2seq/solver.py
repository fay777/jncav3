"""Accurately named alias for the legacy ``a3c_gcn_seq2seq`` implementation."""

from base import SolutionStepEnvironment
from solver import registry
from ..a3c_gcn_seq2seq.solver import A3CGcnSeq2SeqSolver


@registry.register(
    solver_name='a2c_gcn_seq2seq',
    env_cls=SolutionStepEnvironment,
    solver_type='r_learning',
)
class A2CGcnSeq2SeqSolver(A3CGcnSeq2SeqSolver):
    """GCN--Seq2Seq Advantage Actor--Critic baseline.

    The legacy implementation inherits :class:`A2CSolver` and performs one
    synchronous actor--critic update.  It does not launch asynchronous workers
    or maintain shared worker parameters, so ``A2C-GCN`` is the accurate name.
    ``a3c_gcn_seq2seq`` remains registered only as a backwards-compatible
    legacy alias and must not be used as the paper label.
    """

    pass
