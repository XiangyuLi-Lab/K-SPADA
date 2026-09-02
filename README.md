# K-SPADA

K-SPADA is an installable package for spatial transcriptomics data preparation, communication-graph construction, model training, prediction, and spatial visualization.

## Project Structure

```text
K-SPADA/
├── KSPADA/
│   ├── KSPADA_model.py
│   ├── train.py
│   ├── pre.py
│   ├── createCGraph.py
│   ├── customized_linear.py
│   ├── config.py
│   ├── visualization.py
│   ├── data_processing/
│   │   ├── binning.py
│   │   ├── transfer_dataset.py
│   │   └── cross_dataset.py
│   └── resources/
├── tutorials/
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_prediction_and_visualization.ipynb
│   └── 04_binning_data_preparation.ipynb
├── requirements.txt
├── setup.py
└── pyproject.toml
```

## Installation

```bash
conda create -n KSPADA_env python=3.8 -y
conda activate KSPADA_env
cd /your_path/K-SPADA
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
python -m ipykernel install --user --name kspada --display-name "Python (KSPADA)"
```

## Example Data Layout

The repository does not include datasets, model weights, or generated outputs. Replace `/your_path` with your own directory and use generic file names such as:

```text
/your_path/example_dataset/
├── source_data.h5ad
├── reference_data.h5ad
├── train_valid_graph.h5ad
└── query_graph.h5ad
```

Input AnnData objects must contain:

- `adata.obs['x']`: spatial x coordinate.
- `adata.obs['y']`: spatial y coordinate.
- `adata.obs['batch']`: sample identifier.
- `adata.obs['Celltype']`: training label.

## Tutorial Order

1. `tutorials/01_data_preparation.ipynb`: prepare reference and query graph files.
2. `tutorials/02_model_training.ipynb`: train a model with explicit parameters.
3. `tutorials/03_prediction_and_visualization.ipynb`: predict one query sample and create spatial plots.
4. `tutorials/04_binning_data_preparation.ipynb`: apply zero-preserving quantile binning.

All Notebook code is procedural, and all paths, sample names, and model parameters are defined explicitly in the corresponding Notebook.

## Main API

```python
import KSPADA

KSPADA.prepare_data_workflow(...)
KSPADA.train_model(...)
KSPADA.predict_model(...)
KSPADA.bin_adata(...)
KSPADA.plot_prediction_spatial(...)
```
