"""Minimal action-space compatibility used by the local RL environments.

The project only requires a discrete action count and does not use the Gym API.
Keeping this local avoids an unnecessary runtime dependency.
"""


class Discrete:
    def __init__(self, n):
        if int(n) <= 0:
            raise ValueError('Discrete action-space size must be positive.')
        self.n = int(n)

    def contains(self, value):
        return isinstance(value, int) and 0 <= value < self.n


class _Spaces:
    Discrete = Discrete


spaces = _Spaces()
