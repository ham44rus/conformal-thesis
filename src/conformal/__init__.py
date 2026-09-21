"""等角予測（conformal prediction）の実装。卒業研究 2026年度後期。"""

from .split import conformal_index, conformal_quantile, conformal_pvalue, interval_from_quantile
from .scores import absolute_residual, normalized_residual, cqr_score, cqr_interval
from .metrics import (
    coverage,
    mean_width,
    clopper_pearson,
    bin_edges_from_quantiles,
    feature_stratified_coverage,
    size_stratified_coverage,
)
from .adaptive import (
    out_of_fold_predict,
    fit_sigma,
    SigmaEstimator,
    fit_quantile_pair,
    QuantilePair,
)

__all__ = [
    "conformal_quantile",
    "conformal_pvalue",
    "interval_from_quantile",
    "absolute_residual",
    "normalized_residual",
    "cqr_score",
    "cqr_interval",
    "coverage",
    "mean_width",
    "clopper_pearson",
    "bin_edges_from_quantiles",
    "feature_stratified_coverage",
    "size_stratified_coverage",
    "out_of_fold_predict",
    "fit_sigma",
    "SigmaEstimator",
    "fit_quantile_pair",
    "QuantilePair",
]
