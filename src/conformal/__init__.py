"""等角予測（conformal prediction）の実装。卒業研究 2026年度後期。"""

from .split import conformal_quantile, conformal_pvalue, interval_from_quantile
from .scores import absolute_residual, normalized_residual, cqr_score, cqr_interval
from .metrics import (
    coverage,
    mean_width,
    clopper_pearson,
    feature_stratified_coverage,
    size_stratified_coverage,
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
    "feature_stratified_coverage",
    "size_stratified_coverage",
]
