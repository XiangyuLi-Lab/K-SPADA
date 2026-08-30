"""Spatial prediction visualizations matching the original handoff scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd


AnnDataLike = Union[str, Path, ad.AnnData]


def _read_adata(adata_input: AnnDataLike, copy: bool = True) -> ad.AnnData:
    """Read a local h5ad file or copy an in-memory AnnData object."""
    if isinstance(adata_input, ad.AnnData):
        return adata_input.copy() if copy else adata_input
    input_path = Path(adata_input).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    return ad.read_h5ad(input_path)


def build_label_colors(labels: Iterable[str]) -> Dict[str, object]:
    """Create the original ``tab20c`` label mapping used by the handoff code."""
    unique_labels = list(dict.fromkeys(str(label) for label in labels))
    color_map = plt.colormaps.get_cmap("tab20c").resampled(max(len(unique_labels), 1))
    return {label: color_map(index) for index, label in enumerate(unique_labels)}


def plot_prediction_spatial(
    adata_input: AnnDataLike,
    output_prefix: Optional[Union[str, Path]] = None,
    reference_adata: Optional[AnnDataLike] = None,
    point_size: float = 50,
):
    """Plot ground truth and prediction using the original training-script logic."""
    prediction_adata = _read_adata(adata_input, copy=True)
    required_columns = {"x", "y", "Celltype", "Prediction"}
    missing_columns = required_columns.difference(prediction_adata.obs.columns)
    if missing_columns:
        raise KeyError(f"Missing required observation columns: {sorted(missing_columns)}")

    if reference_adata is None:
        label_order = prediction_adata.obs["Celltype"].astype(str).unique().tolist()
    else:
        reference_object = _read_adata(reference_adata, copy=False)
        label_order = reference_object.obs["Celltype"].astype(str).unique().tolist()
    predicted_labels = prediction_adata.obs["Prediction"].astype(str).unique().tolist()
    label_order.extend(label for label in predicted_labels if label not in label_order)
    colors = build_label_colors(label_order)

    x_coordinates = prediction_adata.obs["x"]
    y_coordinates = -prediction_adata.obs["y"]
    celltypes = prediction_adata.obs["Celltype"].astype(str)
    predictions = prediction_adata.obs["Prediction"].astype(str)
    figure, axes = plt.subplots(1, 2, figsize=(20, 10))

    for celltype in celltypes.unique():
        coordinates = pd.concat([x_coordinates[celltypes == celltype], y_coordinates[celltypes == celltype]], axis=1)
        axes[0].scatter(coordinates["x"], coordinates["y"], color=colors[celltype], label=celltype, s=point_size)
    axes[0].legend()
    axes[0].set_title("Celltype")

    for prediction in predictions.unique():
        coordinates = pd.concat([x_coordinates[predictions == prediction], y_coordinates[predictions == prediction]], axis=1)
        axes[1].scatter(coordinates["x"], coordinates["y"], color=colors[prediction], label=prediction, s=point_size)
    axes[1].legend()
    axes[1].set_title("Prediction")
    plt.tight_layout()

    if output_prefix is not None:
        resolved_prefix = Path(output_prefix).expanduser().resolve()
        resolved_prefix.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(resolved_prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
        figure.savefig(resolved_prefix.with_suffix(".pdf"), bbox_inches="tight")
        figure.savefig(resolved_prefix.with_suffix(".svg"), bbox_inches="tight")
    return figure, axes, colors
