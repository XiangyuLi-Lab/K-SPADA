# K-SPADA Design

## Goals

- Keep the package structure close to the original TOSICA-derived workflow.
- Assign each core responsibility to one primary module.
- Keep data preparation, graph construction, training, prediction, and visualization separate.
- Keep tutorial parameters explicit and paths portable.

## Data Flow

```text
AnnData
   │
   ├── data_processing/ ──► createCGraph.py
   │                              │
   │                              ▼
   ├──────────────────────────► train.py ──► checkpoint
   │                                             │
   └──────────────────────────► pre.py ◄─────────┘
                                      │
                                      ▼
                              visualization.py
```

## Module Responsibilities

- `KSPADA_model.py`: model architecture.
- `train.py`: graph sampling, optimization, and training entry points.
- `pre.py`: checkpoint loading and prediction entry points.
- `createCGraph.py`: pathway scoring and communication-graph construction.
- `customized_linear.py`: gene-to-pathway masked projection.
- `config.py`: default tutorial parameters.
- `visualization.py`: spatial scatter plots.
- `data_processing/`: shared preprocessing, binning, generic transfer preparation, and cross-dataset preparation.

## Compatibility

- `KSPADA.train()` and `KSPADA.pre()` retain the original-style interfaces.
- `train_model()`, `predict_model()`, and `prepare_data_workflow()` provide path-oriented interfaces.
- The distribution name is `kspada`; the import name is `KSPADA`.
- Datasets, model weights, and generated outputs are not included in the Python package.
