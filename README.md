# K-SPADA

K-SPADA is an organized, installable version of the local TOSICA-derived research code. The package keeps the original workflow easy to trace by placing the model, training, prediction, communication-graph construction, visualization, and data-processing logic in a small set of focused modules.

## Project Structure

```text
K-SPADA/
├── KSPADA/
│   ├── __init__.py
│   ├── KSPADA_model.py
│   ├── train.py
│   ├── pre.py
│   ├── createCGraph.py
│   ├── customized_linear.py
│   ├── config.py
│   ├── visualization.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── binning.py
│   │   ├── liver_cancer.py
│   │   └── fly3d.py
│   └── resources/
├── tutorials/
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_prediction_and_visualization.ipynb
│   └── 04_binning_data_preparation.ipynb
├── requirement.txt
├── requirements.txt
├── setup.py
└── pyproject.toml
```

## Core Modules

- `KSPADA/KSPADA_model.py` implements spot embedding, the Transformer encoder, the GCN-derived spatial bias, and the classification head.
- `KSPADA/train.py` contains the original training workflow and the path-oriented `train_model()` entry point.
- `KSPADA/pre.py` contains model loading, prediction, prediction accuracy, and the `predict_model()` entry point.
- `KSPADA/createCGraph.py` constructs pathway activity scores, ligand-receptor communication graphs, and combined training and validation graph fields.
- `KSPADA/data_processing/` prepares AnnData objects before model training or prediction, including normalization, gene alignment, batch splitting, and expression binning.
- `KSPADA/visualization.py` reproduces the spatial prediction visualization logic used by the handoff training code.
- `KSPADA/config.py` stores the reproducible liver cancer tutorial configuration and model hyperparameters.

## Installation

The prepared Conda environment is located at:

```text
/home/lixiangyu/anaconda3/envs/KSPADA
```

Activate the environment and install the package in editable mode:

```bash
conda activate KSPADA
cd /home/lixiangyu/sh/program/K-SPADA
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
```

`requirement.txt` references `requirements.txt`, so the dependency list is maintained in one place.

Register the Jupyter kernel if it is not already available:

```bash
python -m ipykernel install --user --name kspada --display-name "Python (KSPADA)"
```

All tutorial notebooks are configured to use the `Python (KSPADA)` kernel.

## Tutorial Workflow

The notebooks execute their main operations directly. They do not use `RUN_*` switches.

Run the tutorials in the following order:

1. `tutorials/01_data_preparation.ipynb`
2. `tutorials/02_model_training.ipynb`
3. `tutorials/03_prediction_and_visualization.ipynb`
4. `tutorials/04_binning_data_preparation.ipynb`

### 1. Data Preparation

`tutorials/01_data_preparation.ipynb` performs the following operations:

1. Loads the liver cancer source AnnData object.
2. Verifies the required `x`, `y`, `batch`, and `Celltype` columns.
3. Uses batches `1` and `2` for training, batch `3` for validation, and batch `4` as the query sample.
4. Writes separate training-validation and query source files.
5. Normalizes expression values and aligns the gene spaces.
6. Splits the data by biological batch.
7. Calculates pathway activity scores.
8. Constructs communication graphs and the final training-validation graph file.

The notebook sets `FORCE = True`, so rerunning it overwrites existing intermediate and output files instead of stopping with `FileExistsError`.

The workflow call uses the following file-oriented arguments:

```python
prepared_paths = KSPADA.prepare_data_workflow(
    train_h5ad=train_valid_source,
    query_h5ad=query_source,
    output_dir=OUTPUT_DIR,
    train_batches=TRAIN_BATCHES,
    valid_batches=VALIDATION_BATCHES,
    query_batches=QUERY_BATCHES,
    cgraph_k=CGRAPH_K,
    force=FORCE,
)
```

### 2. Model Training

`tutorials/02_model_training.ipynb` validates the prepared graph fields and directly calls `KSPADA.train_model()` with the selected liver cancer configuration.

The training code automatically uses `cuda:0` when CUDA is available and falls back to CPU otherwise:

```python
TRAINING_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

CPU training is supported but can be substantially slower than GPU training.

The model directory is written under:

```text
/home/lixiangyu/sh/program/K-SPADA/outputs/liver_cancer_best/liver_cancer_chc_to_hcc_best
```

Expected training artifacts include:

- `mask.npy`
- `pathway.csv`
- `label_dictionary.csv`
- `model-9.pth`

### 3. Prediction and Visualization

`tutorials/03_prediction_and_visualization.ipynb` performs the following operations:

1. Selects the newly trained `model-9.pth` checkpoint when available.
2. Falls back to the historical handoff checkpoint when a newly trained checkpoint is unavailable.
3. Predicts the cell types of liver cancer batch `4`.
4. Calculates overall and per-class classification accuracy.
5. Generates spatial reference-label and prediction-label plots.

The query gene names and gene order must match the model input genes exactly.

### 4. Binning Data Preparation

`tutorials/04_binning_data_preparation.ipynb` demonstrates zero-preserving cell-wise quantile binning and applies it to the complete liver cancer AnnData object.

With `BIN_COUNT = 8`, expression values are represented by integer bins from `0` to `7`. Original zero values remain zero, while nonzero values are assigned to the remaining quantile bins.

The generated example file is written to:

```text
/home/lixiangyu/sh/program/K-SPADA/outputs/tutorial_binning/all_hvg_3000_scgpt_binned_0_7.h5ad
```

## Selected Hyperparameters

The liver cancer training tutorial uses the following configuration:

| Hyperparameter | Value |
|---|---:|
| Pathway collection | `human_gobp` |
| Expression bins | `0-7` |
| Communication graph neighbors | `2` |
| Spatial graph neighbors | `5` |
| Graph batch size | `128` |
| Embedding dimension | `48` |
| Transformer depth | `2` |
| Attention heads | `4` |
| Adam learning rate | `0.001` |
| Weight decay | `5e-5` |
| Final learning-rate ratio | `0.01` |
| Epochs | `10` |
| Random seed | `1` |

The retained historical accuracy for liver cancer batch `4` is `0.8330308529945554`. This result supports the tutorial defaults; the reorganized package does not claim a new exhaustive hyperparameter search.

## Data Requirements

Training and prediction data must use matching gene names and gene order. The expected AnnData fields are:

- `adata.obs["x"]`: first spatial coordinate.
- `adata.obs["y"]`: second spatial coordinate.
- `adata.obs["batch"]`: biological sample or batch identifier.
- `adata.obs["Celltype"]`: training label and optional query reference label.
- `adata.uns["train_edge_index"]`: training graph edges in a combined training-validation object.
- `adata.uns["valid_edge_index"]`: validation graph edges in a combined training-validation object.
- `adata.uns["edge_index"]`: graph edges in an individual query object.

The tutorials use data from:

```text
/home/lixiangyu/TOSICA-main/data/liver_cancer
/home/lixiangyu/TOSICA-main/data/Fly3d
```

## Public API

The main package entry points are:

```python
import KSPADA

KSPADA.prepare_data_workflow(...)
KSPADA.train_model(...)
KSPADA.predict_model(...)
KSPADA.bin_adata(...)
KSPADA.plot_prediction_spatial(...)
```

