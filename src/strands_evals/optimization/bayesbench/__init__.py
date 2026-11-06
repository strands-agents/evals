"""The Bayesian estimation behind the default score model (`BayesBenchModel`), adapted from BayesBench.

- `FactorSpace` and `Encoder`: categorical factors, dummy-coded into an intercept, main effects, and two-way
  interactions.
- `Posterior`: the conjugate Gaussian posterior of a linear model with known per-observation noise.

numpy and scipy only (the `optimization` extra), and nothing else from `strands_evals`. Imported only by
`score_model/bayesbench.py`, lazily, so `strands_evals.optimization` still imports without the extra. BayesBench's
choice of what to run next (expected information gain, seed designs, stopping rules) is left out: the optimizer's
proposers choose the candidates.
"""

from .encoding import Encoder
from .posterior import Posterior
from .space import FactorSpace

__all__ = [
    "Encoder",
    "FactorSpace",
    "Posterior",
]
