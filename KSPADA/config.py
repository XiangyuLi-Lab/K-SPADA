"""Central configuration for reproducible K-SPADA tutorials and experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class TrainingConfig:
    """Store the complete model and optimization hyperparameter set."""

    pathway_name: str = "human_gobp"
    label_name: str = "Celltype"
    max_genes_per_pathway: int = 300
    max_pathway_tokens: int = 300
    mask_ratio: float = 0.015
    unannotated_tokens: int = 1
    embedding_dim: int = 48
    transformer_depth: int = 2
    attention_heads: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 5e-5
    epochs: int = 10
    final_learning_rate_ratio: float = 0.01
    graph_batch_size: int = 128
    spatial_neighbors: int = 5
    communication_neighbors: int = 2
    random_seed: int = 1

    def to_training_kwargs(self) -> Dict[str, object]:
        """Return model and optimizer arguments accepted by ``KSPADA.train_model``."""
        return {
            "gmt_path": self.pathway_name,
            "label_name": self.label_name,
            "max_g": self.max_genes_per_pathway,
            "max_gs": self.max_pathway_tokens,
            "mask_ratio": self.mask_ratio,
            "n_unannotated": self.unannotated_tokens,
            "embed_dim": self.embedding_dim,
            "depth": self.transformer_depth,
            "num_heads": self.attention_heads,
            "lr": self.learning_rate,
            "epochs": self.epochs,
            "lrf": self.final_learning_rate_ratio,
            "use_gcn": True,
            "batch_size_gcn": self.graph_batch_size,
            "neighbor": self.spatial_neighbors,
            "lambda_reg": 1e-6,
            "weight_decay": self.weight_decay,
            "random_seed": self.random_seed,
        }

    def to_data_preparation_kwargs(self) -> Dict[str, object]:
        """Return graph-construction arguments used before model training."""
        return {"cgraph_k": self.communication_neighbors}


@dataclass(frozen=True)
class LiverCancerTutorialConfig:
    """Define the traceable historical-best liver cancer tutorial contract."""

    data_directory: Path = Path("/home/lixiangyu/TOSICA-main/data/liver_cancer")
    historical_project_name: str = "chc_all_bs_128_5no_binning_scgpt_binning_epoch10with_no_pathway"
    train_batches: tuple = ("1", "2")
    validation_batches: tuple = ("3",)
    prediction_batches: tuple = ("4",)
    bin_count: int = 8
    preserve_zero: bool = True
    historical_accuracy: float = 0.8330308529945554

    @property
    def train_valid_path(self) -> Path:
        """Return the prepared training and validation AnnData path."""
        return self.data_directory / "liver_cancer_train_scgpt_binning_8_with_0_chc_all_train_valid_CGraph_k_2.h5ad"

    @property
    def query_paths(self) -> Dict[str, Path]:
        """Return query graph paths keyed by held-out batch identifier."""
        return {
            batch_id: self.data_directory / f"liver_cancer_query_{batch_id}_path_scores_graph_k_2.h5ad"
            for batch_id in self.prediction_batches
        }

    def as_dict(self) -> Dict[str, object]:
        """Return a serializable audit record for notebooks and manifests."""
        config_values = asdict(self)
        config_values["data_directory"] = str(self.data_directory)
        config_values["train_valid_path"] = str(self.train_valid_path)
        config_values["query_paths"] = {key: str(value) for key, value in self.query_paths.items()}
        return config_values


BEST_LIVER_CANCER_TRAINING = TrainingConfig()
BEST_LIVER_CANCER_TUTORIAL = LiverCancerTutorialConfig()
