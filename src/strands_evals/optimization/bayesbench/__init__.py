"""The Bayesian estimation behind the default score model (`BayesBenchModel`), adapted from BayesBench.

- `FactorSpace` and `Encoder`: categorical factors, dummy-coded into an intercept, main effects, and two-way
  interactions.
- `Posterior`: the conjugate Gaussian posterior of a linear model with known per-observation noise.
- `log_evidence`, `prior_scale_weights`, and `Mixture`: prior scales weighed by the marginal likelihood they give the
  observations (empirical Bayes), and the posteriors under them averaged by those weights.
- `linear_contrast`: one linear combination of the effects, e.g. a level's effect, with its sd and 90% interval.
- `HeldOutDiagnostics`: each new observation scored against the posterior from before it, and a flag when the model
  is more wrong than its uncertainty admits.

numpy and scipy only (the `optimization` extra), and nothing else from `strands_evals`. Imported only by
`score_model/bayesbench.py`, lazily, so `strands_evals.optimization` still imports without the extra. BayesBench's
choice of what to run next (expected information gain, seed designs, stopping rules) is left out: the optimizer's
proposers choose the candidates.
"""

from .averaging import Mixture, ScaleWeight, log_evidence, prior_scale_weights
from .contrast import Contrast, linear_contrast
from .diagnostics import Diagnostics, HeldOutDiagnostics
from .encoding import Encoder
from .posterior import Posterior
from .space import FactorSpace

__all__ = [
    "Contrast",
    "Diagnostics",
    "Encoder",
    "FactorSpace",
    "HeldOutDiagnostics",
    "Mixture",
    "Posterior",
    "ScaleWeight",
    "linear_contrast",
    "log_evidence",
    "prior_scale_weights",
]
