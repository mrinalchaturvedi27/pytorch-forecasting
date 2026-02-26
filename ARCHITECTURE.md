# PyTorch Forecasting — Deep Architectural Analysis

This document provides a complete architectural analysis of the `pytorch-forecasting` codebase,
intended for core contributors, future maintainers, and advanced users who need a thorough
understanding of the system.

---

## Table of Contents

1. [High-Level Purpose](#1-high-level-purpose)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Abstractions](#3-core-abstractions)
4. [Model Execution Flow](#4-model-execution-flow)
5. [Data Flow](#5-data-flow)
6. [Design Patterns](#6-design-patterns)
7. [Dependency Structure](#7-dependency-structure)
8. [Testing Framework](#8-testing-framework)
9. [Active Development Zones](#9-active-development-zones)
10. [Complexity & Technical Debt](#10-complexity--technical-debt)
11. [Performance Analysis](#11-performance-analysis)
12. [Extension Mechanisms](#12-extension-mechanisms)
13. [Contribution Entry Points](#13-contribution-entry-points)
14. [Roadmap Signals](#14-roadmap-signals)
15. [Strategic Summary](#15-strategic-summary)

---

## 1. High-Level Purpose

### Problem Solved

`pytorch-forecasting` solves the problem of applying deep learning to time series forecasting
in a structured, reproducible, and practitioner-friendly manner. Raw time series data requires
non-trivial preprocessing (handling of missing values, mixed frequencies, static and
time-varying features, unknown/known covariates), and state-of-the-art models (TFT, DeepAR,
N-BEATS, N-HiTS, etc.) expose complex hyperparameter surfaces. The library hides this complexity
behind a unified API.

### Role in the Ecosystem

The library sits at the intersection of three communities:

- **PyTorch** — low-level tensor operations and neural network primitives.
- **PyTorch Lightning** — training loop abstraction, multi-GPU support, logging, checkpointing.
- **scikit-learn / sktime** — scikit-compatible estimator interface, tagging, and model registry.

It acts as a *domain-specific framework layer* on top of Lightning, providing time series
dataset handling, multi-horizon metrics, and probabilistic forecasting out of the box.

### Target Users

| User Type | Use Case |
|-----------|----------|
| ML Practitioners | Apply SOTA models to real business forecasting problems with minimal boilerplate |
| Researchers | Prototype new architectures on standard datasets using provided base classes |
| Platform Engineers | Integrate trained models into production pipelines via the Lightning ecosystem |

### Design Philosophy

- **Convention over configuration** — sensible defaults reduce the setup burden for beginners.
- **Progressive disclosure** — simple API surface with deep extensibility for experts.
- **Framework composability** — models are Lightning modules; datasets are PyTorch datasets;
  metrics are `torchmetrics`-compatible.
- **Probabilistic first** — distribution losses and quantile forecasting are first-class citizens,
  not afterthoughts.
- **Separation of concerns** — data preprocessing, model architecture, loss computation, and
  training orchestration occupy distinct, interchangeable layers.

---

## 2. Architecture Overview

### Major Modules / Packages

```
pytorch_forecasting/
├── _registry/          Registry of all estimators (models, metrics)
├── base/               skbase-compatible _BaseObject root
├── callbacks/          Lightning callbacks (e.g., prediction capture)
├── data/               Dataset, DataModule, encoders/normalizers
├── layers/             Reusable nn.Module building blocks
├── metrics/            Loss functions and evaluation metrics
├── models/             Model implementations + base model classes
├── tuning/             Optuna / Lightning-based hyperparameter tuning
└── utils/              General utility functions and mixins
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `data/` | Converts raw DataFrames into PyTorch `DataLoader`s; handles normalization, categorical encoding, missing values, variable-length history |
| `models/` | Houses all model architectures and base classes; all are `LightningModule` subclasses |
| `metrics/` | Defines training objectives (loss) and evaluation metrics; supports point, quantile, and distributional outputs |
| `layers/` | Low-level `nn.Module` blocks (attention, embeddings, recurrent cells, decomposition); shared across multiple models |
| `callbacks/` | Hooks into Lightning's callback system for custom behavior during prediction |
| `tuning/` | Thin wrapper around Lightning's LR finder + Optuna integration |
| `_registry/` | Central registry (skbase-based) for dynamic look-up of all estimator classes |
| `base/` | Root `_BaseObject` imported from `skbase` for consistent tagging and metadata |
| `utils/` | Pure utility functions (masking, padding, tensor operations, mixins) |

### Module Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           User / Application                                │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   TimeSeriesDataSet      │  ← data/
                    │   (+ DataModule v2)      │
                    └────────────┬────────────┘
                                 │  DataLoader (normalized batches)
                    ┌────────────▼────────────┐
                    │     BaseModel           │  ← models/base/
                    │  (LightningModule)      │
                    │  ┌──────────────────┐   │
                    │  │  nn.Module       │   │  ← layers/
                    │  │  (architecture)  │   │
                    │  └──────────────────┘   │
                    └────────────┬────────────┘
                                 │  raw predictions
                    ┌────────────▼────────────┐
                    │       Metric            │  ← metrics/
                    │  (loss + evaluation)    │
                    └────────────┬────────────┘
                                 │  scalar loss / output tensors
                    ┌────────────▼────────────┐
                    │  Lightning Trainer      │  (external: lightning.pytorch)
                    │  + Callbacks            │  ← callbacks/
                    └─────────────────────────┘
```

### System Layers

| Layer | Components |
|-------|-----------|
| **API / User Layer** | `TimeSeriesDataSet`, model `__init__`, `from_dataset()`, `Trainer.fit()` |
| **Data Layer** | `TimeSeriesDataSet`, encoders, `DataModule` |
| **Modeling Layer** | `BaseModel` subclasses, `layers/` building blocks |
| **Loss / Metric Layer** | `Metric`, `DistributionLoss`, `QuantileLoss`, `MultiHorizonMetric` |
| **Training Orchestration** | PyTorch Lightning `Trainer`, `callbacks/` |
| **Registry / Metadata** | `_registry/`, `_BaseObject` tags |
| **Utilities** | `utils/` functions and mixins |

---

## 3. Core Abstractions

### Base Classes and Abstract Classes

#### `pytorch_forecasting/base/_base_object.py`
- **`_BaseObject`** (imported from `skbase`)
  - Root of the metadata/tagging hierarchy for package wrappers (`_pkg` classes).
  - Provides `get_class_tags()`, `create_test_instance()`, `get_test_params()`.

#### `pytorch_forecasting/models/base/_base_object.py`
- **`_BasePtForecaster_Common`** (extends `_BaseObject`)
  - Common base for all `_pkg` wrapper classes.
  - Defines `get_cls()`, `name()`, `create_test_instance()`.
- **`_BasePtForecaster`** (extends `_BasePtForecaster_Common`)
  - Package wrapper for v1-style models.
- **`_BasePtForecaster_v2`** (extends `_BasePtForecaster_Common`)
  - Package wrapper for v2-style models.

#### `pytorch_forecasting/models/base/_base_model.py` (v1 — stable)
- **`BaseModel`** (extends `LightningModule`, `InitialParameterRepresenterMixIn`)
  - Root for all time series models.
  - Key methods: `forward()`, `training_step()`, `validation_step()`, `predict_step()`,
    `from_dataset()`, `to_prediction()`, `to_quantiles()`, `interpret_output()`.
  - Handles: automatic checkpointing args, optimizer configuration, output logging,
    gradient clipping, teacher forcing.

- **`BaseModelWithCovariates`** (extends `BaseModel`)
  - Adds covariate embedding support via `MultiEmbedding`.
  - Provides `extract_features()` helpers for static/time-varying covariates.

- **`AutoRegressiveBaseModel`** (extends `BaseModel`)
  - Overrides prediction to use teacher forcing or autoregressive decoding.
  - Introduces `output_to_prediction()` for converting raw outputs to next-step inputs.

- **`AutoRegressiveBaseModelWithCovariates`** (extends `AutoRegressiveBaseModel`, `BaseModelWithCovariates`)
  - Combines auto-regressive decoding with covariate support.

#### `pytorch_forecasting/models/base/_base_model_v2.py` (v2 — experimental)
- **`BaseModel`** (extends `LightningModule`)
  - Simplified skeleton for the next-generation API.
  - Cleaner separation: `forward()` is fully architecture-defined; training loop uses
    standard Lightning hooks.
  - Supports `"adam"` / `"sgd"` strings for optimizer, built-in LR scheduler selection.

#### `pytorch_forecasting/models/base/_tslib_base_model_v2.py` (experimental)
- **`TslibBaseModel`** (extends v2 `BaseModel`)
  - Adapter for models originating from the `tslib` ecosystem.
  - Accepts `metadata` dict from `TslibDataModule`.

#### `pytorch_forecasting/metrics/base_metrics/_base_metrics.py`
- **`Metric`** (extends `torchmetrics.Metric`)
  - Abstract base for all losses and metrics.
  - Key abstract/overridable methods: `loss()`, `compute()`, `to_prediction()`,
    `to_quantiles()`, `rescale_parameters()`.

- **`MultiHorizonMetric`** (extends `Metric`)
  - Handles per-horizon weighting.
  - Provides `update()` and `compute()` for multi-step evaluation.

- **`DistributionLoss`** (extends `Metric`)
  - Parametric distribution losses; subclasses define the distribution.
  - Core methods: `map_x_to_distribution()`, `log_prob()`, `sample()`.

- **`MultivariateDistributionLoss`** (extends `DistributionLoss`)
  - For joint distributions over multiple targets.

- **`MultiLoss`** (extends `Metric`)
  - Weighted sum of multiple loss functions (one per target in multi-target settings).

- **`CompositeMetric`** (extends `Metric`)
  - Algebraic combination (`+`, `*`) of metrics for convenience.

### Inheritance Hierarchy (Estimator Classes)

```
LightningModule  (lightning.pytorch)
└── BaseModel  (v1)                            models/base/_base_model.py
    ├── BaseModelWithCovariates
    │   ├── TemporalFusionTransformer           models/temporal_fusion_transformer/
    │   ├── NHiTS                               models/nhits/
    │   ├── DecoderMLP                          models/mlp/
    │   ├── TIDE                                models/tide/
    │   ├── TimesXer                            models/timexer/
    │   └── RecurrentNetwork (non-AR variant)   models/rnn/
    ├── AutoRegressiveBaseModel
    │   ├── AutoRegressiveBaseModelWithCovariates
    │   │   ├── DeepAR                          models/deepar/
    │   │   └── RecurrentNetwork (AR variant)   models/rnn/
    │   └── xLSTM                               models/xlstm/
    ├── NBeatsAdapter → BaseModel
    │   ├── NBeats                              models/nbeats/
    │   └── NBeatsKAN                           models/nbeats/
    └── SAMformer                               models/samformer/

LightningModule  (lightning.pytorch)
└── BaseModel  (v2, experimental)              models/base/_base_model_v2.py
    ├── TslibBaseModel                         models/base/_tslib_base_model_v2.py
    │   └── DLinear (v2)                       models/dlinear/
    └── TemporalFusionTransformer (v2)         models/temporal_fusion_transformer/

torchmetrics.Metric
└── Metric                                     metrics/base_metrics/
    ├── MultiHorizonMetric
    │   ├── MAE, MAPE, MASE, RMSE, SMAPE
    │   ├── CrossEntropy, PoissonLoss, TweedieLoss
    │   └── QuantileLoss
    ├── DistributionLoss
    │   ├── NormalDistributionLoss
    │   ├── LogNormalDistributionLoss
    │   ├── NegativeBinomialDistributionLoss
    │   ├── BetaDistributionLoss
    │   ├── ImplicitQuantileNetworkDistributionLoss
    │   └── MQF2DistributionLoss
    ├── MultivariateDistributionLoss
    │   └── MultivariateNormalDistributionLoss
    ├── MultiLoss
    └── CompositeMetric
```

### How Polymorphism Is Used

- **`to_prediction()`** — every `Metric` subclass overrides this to convert its raw output
  (e.g., distribution parameters) to point predictions. The base model calls this uniformly,
  making model code agnostic to loss type.
- **`to_quantiles()`** — quantile metrics return explicit quantile bands; distribution losses
  sample or analytically compute them; point losses fall back to a single quantile.
- **`loss()`** — called identically regardless of whether the objective is MAE, log-likelihood,
  or quantile pinball loss.
- **`forward()`** — each model defines its own forward pass; `BaseModel.training_step()` calls
  it generically.
- **`from_dataset()`** — each model overrides this factory to extract the exact configuration
  it needs from the dataset metadata.

### How Extensibility Is Achieved

1. **Subclassing** — implement a new model by extending `BaseModel` or `BaseModelWithCovariates`.
2. **`from_dataset()` factory** — override to read dataset metadata and construct the model,
   keeping architecture and dataset concerns decoupled.
3. **Metric protocol** — plug in any `Metric` subclass as the loss function; no changes to
   model or training code required.
4. **Layer library** — compose any model from the shared `layers/` building blocks.
5. **Registry** — register a new model in `_registry/_lookup.py` to make it discoverable via
   the package-level API and test infrastructure.
6. **`_pkg` wrappers** — thin wrapper classes (implementing `_BasePtForecaster`) provide
   metadata tags and `create_test_instance()` for the registry.

---

## 4. Model Execution Flow

### Initialization

```python
dataset = TimeSeriesDataSet(df, ...)           # 1. Prepare dataset
model = TemporalFusionTransformer.from_dataset( # 2. Factory reads dataset metadata
    dataset,
    learning_rate=0.03,
    hidden_size=64,
    ...
)
```

Inside `from_dataset()`:
- Reads `dataset.get_parameters()` to extract encoder/decoder lengths, feature counts, etc.
- Instantiates the model with the correct architecture sizes.
- Stores hyperparameters via `self.save_hyperparameters()`.

### Training (`trainer.fit()`)

```
Trainer.fit(model, train_dataloader, val_dataloader)
    → LightningModule.training_step(batch, batch_idx)
        → BaseModel.training_step()
            → self.forward(x)              # architecture-specific
            → self.loss.loss(y_hat, y)     # metric-specific
            → self.log(...)
```

Steps inside `BaseModel.training_step()`:
1. Unpack `batch` into `(x, y)`.
2. Call `self.forward(x)` → raw output dict / tensor.
3. Compute loss via `self.loss.loss(predictions, targets)`.
4. Log training metrics.
5. Return scalar loss for backpropagation.

For auto-regressive models, step 2 involves:
- Encoding the history with the encoder network.
- Iteratively decoding one step at a time, feeding predictions back as inputs.

### Prediction

```python
predictions = model.predict(dataloader, mode="prediction")
```

Internally:
1. `Trainer.predict()` calls `predict_step()` for each batch.
2. `BaseModel.predict_step()` calls `forward()`.
3. Output is passed through `self.loss.to_prediction()` (or `to_quantiles()` for quantiles).
4. `PredictCallback` collects outputs across batches.
5. Results are denormalized using the dataset's normalizer.

### Output Generation

```
forward() output
    ↓
Metric.to_prediction()  →  point forecast tensor  [batch, time_steps]
Metric.to_quantiles()   →  quantile tensor         [batch, time_steps, n_quantiles]
```

If `DistributionLoss` is used:
- `forward()` outputs distribution parameters (e.g., `mu`, `sigma` for Normal).
- `to_prediction()` returns the distribution mean.
- `to_quantiles()` calls `torch.distributions.*` quantile functions.
- `sample()` draws Monte Carlo samples for simulation.

---

## 5. Data Flow

### Data Entry

Raw tabular data (pandas `DataFrame`) enters via `TimeSeriesDataSet.__init__()`.

```python
TimeSeriesDataSet(
    data=df,
    time_idx="time_idx",
    target="value",
    group_ids=["series_id"],
    max_encoder_length=60,
    max_prediction_length=20,
    static_categoricals=["store"],
    time_varying_known_reals=["price"],
    time_varying_unknown_reals=["value"],
    target_normalizer=GroupNormalizer(groups=["store"]),
)
```

### Preprocessing Steps

1. **Sorting** — data is sorted by `group_ids` + `time_idx`.
2. **Index building** — start/end indices per sample are computed for fast slicing.
3. **Categorical encoding** — `NaNLabelEncoder` maps string categories to integers.
4. **Normalization** — `TorchNormalizer` / `GroupNormalizer` / `EncoderNormalizer` fit on
   training data and transform the target and continuous reals.
5. **Missing value handling** — configurable fill strategies (forward-fill, zero, etc.).
6. **Lag features** — optional lags of target/covariates are appended.

### Transformation Layers (Encoders in `data/encoders.py`)

| Encoder | Description |
|---------|-------------|
| `TorchNormalizer` | Global center + scale normalization; supports log / log1p / logit transforms |
| `GroupNormalizer` | Per-group normalization; prevents data leakage across series |
| `EncoderNormalizer` | Fits on the encoder window only; re-fits per sample during training |
| `MultiNormalizer` | Wraps multiple normalizers for multi-target settings |
| `NaNLabelEncoder` | Ordinal encoding of categoricals, with explicit `NaN` class |

### Model Interaction

`TimeSeriesDataSet.__getitem__()` returns a dict:
```python
{
    "x_cat": LongTensor[encoder_len + decoder_len, n_cat],
    "x_cont": FloatTensor[encoder_len + decoder_len, n_cont],
    "encoder_length": int,
    "decoder_length": int,
    "encoder_target": FloatTensor,
    "decoder_target": FloatTensor,
    "target_scale": FloatTensor,        # normalizer parameters
    "groups": LongTensor,               # group membership
    ...
}
```

`BaseModel.training_step()` unpacks this batch and passes `x` to `self.forward(x)`.
Each model interprets the batch dict according to its architecture.

### Output Formatting

- Point predictions: `FloatTensor[batch, prediction_len]`
- Quantile predictions: `FloatTensor[batch, prediction_len, n_quantiles]`
- Distribution parameters: model-specific dict (e.g., `{"loc": ..., "scale": ...}`)

Denormalization is performed by `Metric.rescale_parameters()` or by calling
`dataset.target_normalizer.inverse_transform()`.

---

## 6. Design Patterns

### Factory Pattern
**Where:** `BaseModel.from_dataset(dataset, ...)` on every model class.
**Why:** Decouples model instantiation from dataset metadata parsing. Users do not need to
manually pass architecture sizes; the factory reads them from the dataset.

### Template Method Pattern
**Where:** `BaseModel.training_step()`, `validation_step()`, `predict_step()`.
**Why:** The base class defines the overall algorithm (unpack → forward → loss → log),
while subclasses override only `forward()` and `from_dataset()`.

### Strategy Pattern
**Where:** `loss` parameter accepted by every model constructor.
**Why:** Any `Metric` subclass can be swapped in as the training objective without changing
model code. The model delegates all loss-related decisions to the `Metric` object.

### Adapter Pattern
**Where:** `_pkg` wrapper classes (e.g., `TemporalFusionTransformerPkg`).
**Why:** Wrap the raw `LightningModule` classes in a `skbase`-compatible interface so they
can be indexed, tagged, and discovered via the registry.

**Where:** `TslibBaseModel` adapting tslib models to the pytorch-forecasting v2 API.
**Why:** Allows models from an external ecosystem to work within the training/inference
pipeline without rewriting them.

### Composite Pattern
**Where:** `MultiLoss`, `CompositeMetric`.
**Why:** Multiple metrics can be combined arithmetically into a single trainable objective,
transparent to the model.

### Decorator Pattern
**Where:** `EncoderNormalizer` wrapping a `TorchNormalizer` to add per-sample fitting.
**Where:** `MultiNormalizer` wrapping a list of normalizers.
**Why:** Transparently augments the normalization behavior without changing the normalizer
interface.

### Registry / Lookup Pattern
**Where:** `pytorch_forecasting/_registry/_lookup.py`.
**Why:** Central catalog of all estimator classes; enables automated testing, documentation
generation, and dynamic instantiation.

### Mixin Pattern
**Where:** `InitialParameterRepresenterMixIn`, `OutputMixIn`, `TupleOutputMixIn` in `utils/`.
**Why:** Orthogonal behaviors (parameter persistence, output naming) are injected into the
class hierarchy without creating deep diamond inheritance.

---

## 7. Dependency Structure

### Inter-Module Dependencies

```
utils/ ←──────────────── (imported by almost everything)
       ↑
data/encoders ←────────── data/timeseries
       ↑
metrics/base_metrics ←─── metrics/ (point, quantile, distributions)
       ↑
models/base/ ←──────────── models/* (all model implementations)
       ↑
layers/ ←───────────────── models/* (shared nn.Module blocks)
       ↑
callbacks/ ←────────────── models/base/ (predict callback)
       ↑
_registry/ ←──────────── (indexes models/ and metrics/)
       ↑
tuning/ ←────────────── (wraps lightning Trainer + optuna)
```

### Central Modules with Most Dependents

| Module | Role | # Dependents (approx.) |
|--------|------|------------------------|
| `utils/` | Shared helpers | All modules |
| `metrics/base_metrics/` | Loss interface | All models + evaluation code |
| `models/base/_base_model.py` | Model superclass | All v1 model implementations |
| `data/timeseries/` | Dataset class | All model factories + tests |
| `data/encoders.py` | Normalization | Dataset + base model |

### Dependency Graph (Simplified)

```
tuning ──→ models/base ──→ metrics ──→ utils
              ↑                ↑
           layers/         data/encoders
              ↑                ↑
           models/*        data/timeseries
              ↑
         _registry ──→ models/* + metrics/*
```

---

## 8. Testing Framework

### Test Structure

```
tests/                          # Top-level integration tests
├── conftest.py                 # Shared fixtures (datasets, dataloaders)
├── test_data/                  # Tests for TimeSeriesDataSet + encoders
├── test_metrics.py             # Tests for all Metric subclasses
├── test_models/                # Per-model integration tests
└── test_utils/                 # Tests for utility functions

pytorch_forecasting/tests/      # Package-level tests
├── test_all_estimators.py      # Runs get_test_params / create_test_instance on all registry entries
├── test_class_register.py      # Validates registry integrity
└── _data_scenarios.py          # Shared dataset fixtures for model tests
```

### Test Design Philosophy

- **Fixtures over hard-coded data** — `conftest.py` provides parameterized dataset fixtures
  covering multiple scenarios (with/without covariates, multi-target, etc.).
- **Registry-driven testing** — `test_all_estimators.py` automatically discovers all
  registered models and runs smoke tests, reducing the risk of untested new additions.
- **Parameterization** — `pytest.mark.parametrize` covers multiple loss functions,
  normalizer types, and dataset configurations per model.
- **Lightweight instances** — test models use small hidden sizes and short sequences to
  keep CI fast.

### How Estimators Are Verified

1. `create_test_instance()` on the `_pkg` wrapper creates a minimal model instance.
2. `from_dataset()` is called with a synthetic `TimeSeriesDataSet`.
3. A one-batch training step is executed.
4. `to_prediction()` and `to_quantiles()` are exercised on the output.
5. Prediction shapes and types are asserted.

---

## 9. Active Development Zones

### Under Active Development

- **v2 base model** (`models/base/_base_model_v2.py`, `_tslib_base_model_v2.py`) — marked
  experimental; API is still evolving.
- **New model variants** — `_tft_v2.py`, `_tft_pkg_v2.py` suggest TFT is being ported to
  the v2 API.
- **tslib integration** — `DLinear`, `TimesXer`, `SAMformer`, and `xLSTM` are recent
  additions following the tslib architecture pattern.
- **`data/_tslib_data_module.py`** — new `EncoderDecoderTimeSeriesDataModule` for v2 models
  is actively being built out.

### Experimental Areas

- All files/classes carrying a v2 suffix or explicit experimental disclaimer.
- `TslibBaseModel` — marked with a `warn()` in `__init__`.
- `NBeatsKAN` — uses Kolmogorov-Arnold Network layers, a recent research direction.

### Stable Areas

- `TimeSeriesDataSet` (v1) and all its encoders.
- `BaseModel` (v1) and the full v1 model zoo (TFT, NBeats, NHiTS, DeepAR, RNN, MLP).
- `metrics/` — all point, quantile, and distribution losses.
- `layers/` building blocks used by stable models.

---

## 10. Complexity & Technical Debt

### Tightly Coupled Modules

- `BaseModel` (v1) imports `TimeSeriesDataSet` directly, coupling the model layer to the
  data layer. The `from_dataset()` pattern is the primary mechanism for this coupling.
- `BaseModel.training_step()` handles loss computation, metric logging, and gradient
  accumulation in one method — high cyclomatic complexity.

### Large or Complex Files

| File | Approximate Size | Complexity Driver |
|------|-----------------|-------------------|
| `models/base/_base_model.py` | ~2,000+ lines | Combines training loop, prediction, plotting, serialization |
| `data/timeseries/_timeseries.py` | ~1,500+ lines | Full dataset logic, preprocessing, sampling |
| `models/temporal_fusion_transformer/_tft.py` | ~800+ lines | Complex attention + variable selection architecture |

### Unclear Abstractions

- The distinction between v1 and v2 base models is not yet clearly documented for end users;
  the coexistence of both creates confusion about which to extend.
- `_pkg` wrapper classes duplicate some metadata from the underlying model class, creating
  a risk of inconsistency.

### Design Inconsistencies

- Some models (NBeats) use an intermediate `NBeatsAdapter` class that is not part of the
  primary hierarchy; its purpose is not obvious without reading the code.
- `to_prediction()` / `to_quantiles()` are defined on both `BaseModel` (v1) and `Metric`,
  potentially confusing callers about which to call.

### Separation of Concerns Violations

- `BaseModel` (v1) includes visualization/plotting logic (`plot_prediction()`,
  `plot_interpretation()`), which arguably belongs in a separate utility or callback.
- Normalization (denormalization of predictions) is partially handled in `Metric.rescale_parameters()`
  and partially in `BaseModel.predict()`, splitting one concern across two layers.

---

## 11. Performance Analysis

### Computational Bottlenecks

- **Auto-regressive decoding** (`AutoRegressiveBaseModel`) — sequential token generation
  cannot be parallelized along the prediction horizon; the loop in `predict()` is inherently
  O(prediction_length).
- **TFT attention** — multi-head attention over long encoder sequences is O(n²) in sequence
  length; long histories are expensive.
- **`TimeSeriesDataSet.__getitem__()`** — per-sample preprocessing and normalization happen
  at data loading time; heavy transformations can bottleneck data pipelines.

### Memory-Heavy Components

- **Distribution losses with sampling** — Monte Carlo sampling (`sample()`) creates large
  intermediate tensors proportional to `n_samples × batch × horizon`.
- **NBeats / NHiTS stacks** — multiple expansion layers in each stack increase memory
  footprint relative to sequence length.
- **Embedding tables** (`MultiEmbedding`) — large cardinality categoricals can consume
  significant memory.

### Scalability Risks

- `TimeSeriesDataSet` loads the full DataFrame into memory; very large datasets may require
  chunked or streaming variants.
- No built-in distributed data sharding; relies entirely on Lightning's DDP strategy.

### GPU Acceleration Opportunities

- Attention layers in `layers/_attention/` are already GPU-compatible; FlashAttention
  integration would yield significant speedups for long sequences.
- Normalizer `fit()` / `transform()` operations run on CPU; moving group statistics to GPU
  tensors would reduce transfer overhead.

---

## 12. Extension Mechanisms

### Adding a New Estimator (Model)

**Step 1 — Implement the model class**

Create `pytorch_forecasting/models/<model_name>/_<model_name>.py`:

```python
from pytorch_forecasting.models.base import BaseModelWithCovariates

class MyModel(BaseModelWithCovariates):
    def __init__(self, hidden_size: int, loss, **kwargs):
        super().__init__(loss=loss, **kwargs)
        self.save_hyperparameters()
        self.net = nn.Linear(hidden_size, hidden_size)

    @classmethod
    def from_dataset(cls, dataset, **kwargs):
        # extract metadata from dataset
        return cls(hidden_size=kwargs.get("hidden_size", 64), **kwargs)

    def forward(self, x: dict) -> dict:
        # x contains encoder/decoder tensors
        out = self.net(x["encoder_cont"])
        return self.to_network_output(prediction=out)
```

Required method overrides:
- `forward(x)` — must return a dict/namedtuple accepted by `self.loss.loss()`.
- `from_dataset(dataset, **kwargs)` — factory method.

Optional overrides:
- `to_prediction(net_output)` — if the default behavior is insufficient.
- `interpret_output(net_output)` — for custom interpretability.

**Step 2 — Create a `_pkg` wrapper**

Create `pytorch_forecasting/models/<model_name>/_<model_name>_pkg.py`:

```python
from pytorch_forecasting.models.base._base_object import _BasePtForecaster
from pytorch_forecasting.models.<model_name> import MyModel

class MyModelPkg(_BasePtForecaster):
    _tags = {
        "info:name": "MyModel",
        "info:compute": "low",
        "authors": ["your_github_handle"],
    }

    @classmethod
    def get_cls(cls):
        return MyModel

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [{"hidden_size": 16}]
```

**Step 3 — Register the model**

Add to `pytorch_forecasting/_registry/_lookup.py`:

```python
("MyModel", "pytorch_forecasting.models.<model_name>._<model_name>_pkg", "MyModelPkg"),
```

**Step 4 — Expose via `__init__.py`**

Add to `pytorch_forecasting/models/__init__.py`:

```python
from pytorch_forecasting.models.<model_name> import MyModel
```

**Step 5 — Test**

The registry-driven `test_all_estimators.py` will automatically pick up the new model
and run smoke tests via `get_test_params()` and `create_test_instance()`.

---

### Adding a New Probabilistic Distribution

Create a subclass of `DistributionLoss` in `pytorch_forecasting/metrics/`:

```python
from pytorch_forecasting.metrics.base_metrics import DistributionLoss
import torch.distributions as D

class GammaDistributionLoss(DistributionLoss):
    distribution_class = D.Gamma
    distribution_arguments = ["concentration", "rate"]  # parameter names

    def map_x_to_distribution(self, x: torch.Tensor) -> D.Gamma:
        # x shape: [batch, time, n_params]
        concentration = x[..., 0].exp() + 1e-8
        rate = x[..., 1].exp() + 1e-8
        return D.Gamma(concentration, rate)
```

Required overrides:
- `map_x_to_distribution(x)` — maps raw model outputs to a `torch.distributions` object.

The base class provides `loss()`, `to_prediction()`, `to_quantiles()`, and `sample()`
automatically by delegating to the distribution object.

Add to `pytorch_forecasting/metrics/__init__.py` and register in `_lookup.py`.

---

### Adding a New Point / Quantile Metric

Extend `MultiHorizonMetric`:

```python
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric

class HuberLoss(MultiHorizonMetric):
    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(y_pred, target, reduction="none")
```

Required override: `loss(y_pred, target)` returning element-wise losses.

---

## 13. Contribution Entry Points

### Beginner Level

- **Documentation** — improve docstrings, add usage examples to existing model classes,
  expand tutorials in `docs/source/tutorials/`.
- **Tests** — add edge-case tests to `tests/test_data/` or `tests/test_metrics.py`
  (e.g., empty batches, single-sample groups, all-NaN targets).
- **Bug fixes in utilities** — `utils/` functions are self-contained and easy to reason about.
- **`CHANGELOG.md` maintenance** — tag PRs with correct changelog entries.

### Intermediate Level

- **New metrics** — implement a new `MultiHorizonMetric` or `DistributionLoss` and
  register it; tests are largely auto-generated.
- **Data encoders** — add new transformation types to `TorchNormalizer`
  (e.g., Box-Cox, Yeo-Johnson).
- **New layers** — add reusable `nn.Module` blocks to `layers/`; useful for new model
  architectures.
- **Port a v1 model to v2 API** — follow the pattern of `_tft_v2.py` for an existing model.
- **DataModule enhancements** — extend `EncoderDecoderTimeSeriesDataModule` with additional
  split strategies or cross-validation support.

### Advanced Architecture Level

- **Complete v2 base model API** — finalize `_base_model_v2.py`, stabilize the contract
  between models and the new DataModule, deprecate v1 cleanly.
- **Streaming / out-of-core datasets** — redesign `TimeSeriesDataSet` to support datasets
  larger than RAM via chunked loading or `IterableDataset`.
- **Distributed training improvements** — add sharded dataset support; ensure normalizers
  fit correctly under DDP.
- **FlashAttention integration** — drop-in replacement in `layers/_attention/` for
  O(n) memory attention on long sequences.
- **Conformal prediction** — add a post-hoc prediction interval layer wrapping any model.
- **Native multi-step probabilistic evaluation** — CRPS, energy scores, calibration plots
  as first-class metrics.

---

## 14. Roadmap Signals

### Implied Long-Term Evolution

- **v2 API stabilization** — the experimental v2 base classes, v2 model variants, and new
  `TslibDataModule` all point toward a cleaner, more modular second-generation API designed
  to ease integration of models from the broader tslib ecosystem.

- **Ecosystem unification** — the `skbase` compatibility layer (`_BaseObject`, `_registry/`)
  signals intent to align with the wider `sktime` / `skbase` ecosystem for
  interoperability, automated testing, and model discovery.

- **Probabilistic forecasting expansion** — the `MQF2DistributionLoss` and
  `ImplicitQuantileNetworkDistributionLoss` additions suggest growing investment in
  joint/multivariate probabilistic outputs.

### Architectural Transitions in Progress

- **v1 → v2 migration** — co-existence of `_base_model.py` and `_base_model_v2.py`
  represents a planned but incomplete migration. New models targeting the library should
  monitor which API stabilizes first.
- **DataModule introduction** — moving from bare `DataLoader` construction to a full
  Lightning `DataModule` abstraction for better separation of data and model concerns.

### Major Redesigns Implied

- **Decoupling `BaseModel` from `TimeSeriesDataSet`** — the v2 API appears to be removing
  the tight `from_dataset()` coupling in favor of passing metadata via `DataModule`.
- **Modular normalizer pipeline** — the multiple normalizer classes suggest a future
  sklearn-style `Pipeline` for feature preprocessing may emerge.

---

## 15. Strategic Summary

### Main Strengths

- **Comprehensive v1 API** — `TimeSeriesDataSet` + `BaseModel` is battle-tested, covering
  an unusually wide range of real-world time series scenarios out of the box.
- **Probabilistic forecasting** — rich support for distribution losses, quantile forecasting,
  and sampling-based uncertainty quantification is a clear differentiator.
- **Lightning integration** — inheriting Lightning's training loop removes enormous
  boilerplate and gives users multi-GPU, mixed-precision, and gradient checkpointing for
  free.
- **Model zoo breadth** — TFT, DeepAR, N-BEATS, N-HiTS, DLinear, TIDE, xLSTM, and more
  cover the range from classical decomposition to state-of-the-art attention models.
- **Registry-driven testing** — new models are automatically swept into the CI test suite
  via `test_all_estimators.py`, reducing the risk of regressions.

### Biggest Architectural Challenges

- **Dual API surface** — v1 and v2 base classes coexist without a clear deprecation
  timeline, creating confusion for contributors and slowing down stabilization of the
  newer design.
- **`BaseModel` (v1) size** — the ~2,000-line base class violates the single-responsibility
  principle and is difficult to extend or refactor safely.
- **In-memory dataset limitation** — `TimeSeriesDataSet` loads the full DataFrame into
  memory, which limits scalability for large-scale industrial applications.
- **Normalization split** — denormalization logic spread across `Metric` and `BaseModel`
  makes it hard to reason about the full data transformation pipeline.

### Most Impactful Areas for Contribution

1. **v2 API completion** — stabilizing `_base_model_v2.py` and the new `DataModule`
   unblocks all future model additions in a cleaner architecture.
2. **Streaming dataset support** — would open the library to industrial-scale deployments.
3. **New probabilistic metrics** (CRPS, energy score) — fills a gap in rigorous evaluation.
4. **Documentation and tutorials** — the existing model zoo is underutilized because
   end-to-end examples for many models are sparse.

### Fastest Path to Becoming a Core Contributor

1. **Start with tests** — add test cases to `tests/test_data/` or `tests/test_models/`;
   this builds familiarity with the data pipeline and model interface.
2. **Implement a metric** — a new `MultiHorizonMetric` or `DistributionLoss` touches
   the metrics layer, registry, and test infrastructure in a self-contained way.
3. **Port or add a model** — use the `_pkg` + registry pattern to add a model;
   this end-to-end exercise covers data, modeling, metrics, and CI.
4. **Engage with the v2 migration** — contributing to `_base_model_v2.py` and the new
   `DataModule` places you at the center of the library's architectural future.
5. **Read and improve `BaseModel` (v1)** — deep familiarity with the base class is the
   single most effective way to understand the full system; refactoring or documenting
   individual methods is a high-value, incremental contribution.
