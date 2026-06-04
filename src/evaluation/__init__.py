from .plots import loss_acc_plot, plot_roc_ovr, plot_roc_ovo
from .crossval_score import (
    get_all_runs,
    filter_by_loss_discrepancy,
    group_by_arch_and_config,
    score_models,
    select_best_configs
)

"""
Możemy też:

from src.evaluation.plot import loss_acc_plot
"""