"""Public K-SPADA API."""

try:
    from importlib.metadata import PackageNotFoundError, version
except ModuleNotFoundError:
    from pkg_resources import DistributionNotFound as PackageNotFoundError
    from pkg_resources import get_distribution

    version = lambda package_name: get_distribution(package_name).version

from pathlib import Path
import os

from .config import BEST_LIVER_CANCER_TRAINING, BEST_LIVER_CANCER_TUTORIAL, LiverCancerTutorialConfig, TrainingConfig
from .createCGraph import (
    add_path_scores,
    build_communication_graph,
    create_train_valid_cgraph,
    prepare_query_cgraph,
    validate_cgraph_fields,
)
from .data_processing import (
    align_adata_to_genes,
    align_adatas_to_union_genes,
    bin_adata,
    bin_expression_matrix,
    bin_expression_row,
    normalize_total_log1p,
    prepare_data_workflow,
    read_adata,
    split_by_batch,
    standardize_obs_columns,
    use_count_layer,
    write_adata,
)
from .pre import (
    add_prediction_match,
    latest_model_weight,
    prediect,
    predict_model,
    prediction_accuracy,
    prediction_accuracy_by_label,
)
from .train import fit_model, train_model
from .visualization import build_label_colors, plot_prediction_spatial

name = "KSPADA"
try:
    __version__ = version("kspada")
except PackageNotFoundError:
    __version__ = "0.2.0"


def train(adata, gmt_path, project=None, pre_weights='', label_name='Celltype', max_g=300, max_gs=300, mask_ratio=0.015, n_unannotated=1, batch_size=8, embed_dim=48, depth=2, num_heads=4, lr=0.001, epochs=10, lrf=0.01, use_gcn=False, data_batch=None, **kwargs):
    """Compatibility entrypoint matching the original TOSICA.train interface."""
    return fit_model(
        adata,
        gmt_path,
        project=project,
        pre_weights=pre_weights,
        label_name=label_name,
        max_g=max_g,
        max_gs=max_gs,
        mask_ratio=mask_ratio,
        n_unannotated=n_unannotated,
        batch_size=batch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        lr=lr,
        epochs=epochs,
        lrf=lrf,
        use_gcn=use_gcn,
        data_batch=[] if data_batch is None else data_batch,
        **kwargs,
    )


def pre(adata, model_weight_path, project, laten=False, save_att='X_att', save_lantent='X_lat', n_step=10000, cutoff=0.1, n_unannotated=1, batch_size=50, embed_dim=48, depth=2, num_heads=4, use_gcn=False, data_batch=None):
    """Compatibility entrypoint matching the original TOSICA.pre interface."""
    mask_path = Path.cwd() / project / 'mask.npy'
    result_adata, _latent = prediect(
        adata,
        model_weight_path,
        project=project,
        mask_path=str(mask_path),
        laten=laten,
        save_att=save_att,
        save_lantent=save_lantent,
        n_step=n_step,
        cutoff=cutoff,
        n_unannotated=n_unannotated,
        batch_size=batch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        use_gcn=use_gcn,
        data_batch=[] if data_batch is None else data_batch,
    )
    return result_adata


__all__ = [
    "BEST_LIVER_CANCER_TRAINING",
    "BEST_LIVER_CANCER_TUTORIAL",
    "LiverCancerTutorialConfig",
    "TrainingConfig",
    "add_path_scores",
    "add_prediction_match",
    "align_adata_to_genes",
    "align_adatas_to_union_genes",
    "bin_adata",
    "bin_expression_matrix",
    "bin_expression_row",
    "build_communication_graph",
    "build_label_colors",
    "create_train_valid_cgraph",
    "latest_model_weight",
    "normalize_total_log1p",
    "plot_prediction_spatial",
    "pre",
    "predict_model",
    "prediction_accuracy",
    "prediction_accuracy_by_label",
    "prepare_data_workflow",
    "prepare_query_cgraph",
    "read_adata",
    "split_by_batch",
    "standardize_obs_columns",
    "train",
    "train_model",
    "use_count_layer",
    "validate_cgraph_fields",
    "write_adata",
]
