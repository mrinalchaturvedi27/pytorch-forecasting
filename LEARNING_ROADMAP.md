# PyTorch Forecasting — 7-Day Learning Roadmap

This roadmap is designed to take you from zero to deep familiarity with the
`pytorch-forecasting` codebase in one focused week. Each day builds on the
previous one, moving from orientation and setup through the data layer, metrics,
model base classes, the full model zoo, layers and extension mechanisms, and
finally testing and contribution. The roadmap draws directly from
[`ARCHITECTURE.md`](./ARCHITECTURE.md), which you should keep open as your
reference companion throughout the week.

---

## Prerequisites

Before starting, ensure you are comfortable with:

- **Python 3.9+** and standard OOP concepts (inheritance, mixins, classmethods)
- **PyTorch** fundamentals — tensors, `nn.Module`, autograd, `DataLoader`
- **PyTorch Lightning** basics — `LightningModule`, `Trainer`, `training_step()`
- **pandas** — `DataFrame`, multi-index operations, groupby
- **Basic time series concepts** — encoder/decoder windows, multi-horizon forecasting,
  seasonality, stationarity

---

## Day 1 — Orientation, Setup & High-Level Architecture

**Goal:** Understand what the library solves, set up a working environment, and map
the top-level structure to the architectural layers described in `ARCHITECTURE.md §1–2`.

### Concepts to Understand

| Concept | Where to Look |
|---------|---------------|
| Library purpose and target users | `ARCHITECTURE.md §1` |
| Design philosophy (convention over configuration, probabilistic-first) | `ARCHITECTURE.md §1` |
| Major module responsibilities | `ARCHITECTURE.md §2` |
| Module interaction diagram | `ARCHITECTURE.md §2 — Module Interaction Diagram` |
| System layers (API → Data → Modeling → Loss → Training) | `ARCHITECTURE.md §2 — System Layers` |

### Files to Read

1. **`README.md`** — high-level overview, feature list, installation instructions.
2. **`ARCHITECTURE.md §1–2`** — purpose, module map, system layers.
3. **`pytorch_forecasting/__init__.py`** — see what the public API surface exposes.
4. **`pytorch_forecasting/models/__init__.py`** — enumerate available models.
5. **`pytorch_forecasting/metrics/__init__.py`** — enumerate available metrics/losses.
6. **`pyproject.toml`** — understand the dependency graph (Lightning, torchmetrics,
   skbase, optuna).

### Hands-On Tasks

```bash
# 1. Install the package in editable mode
pip install -e ".[dev]"

# 2. Run the Stallion tutorial end-to-end to see the full API in action
jupyter notebook examples/stallion.ipynb     # interactive notebook
# or run the script version directly:
python examples/stallion.py
```

- Trace each call in the tutorial to the module it lives in.
- After the tutorial runs successfully, draw your own module interaction diagram
  by hand; compare it with the one in `ARCHITECTURE.md §2`.

### Key Questions to Answer by End of Day 1

1. What is the role of `TimeSeriesDataSet` vs `DataLoader` vs `DataModule`?
2. Why does the library need its own base class (`BaseModel`) on top of
   `LightningModule`?
3. What does "probabilistic-first" mean for the metric/loss design?

---

## Day 2 — Data Layer: TimeSeriesDataSet, Encoders & Normalizers

**Goal:** Deeply understand how raw `DataFrame` data is transformed into
normalized PyTorch tensors ready for model consumption. This layer underlies
everything else.

### Concepts to Understand

| Concept | Where to Look |
|---------|---------------|
| TimeSeriesDataSet construction and parameters | `ARCHITECTURE.md §5 — Data Entry` |
| Preprocessing pipeline (sorting → indexing → encoding → normalizing) | `ARCHITECTURE.md §5 — Preprocessing Steps` |
| Encoder taxonomy (`TorchNormalizer`, `GroupNormalizer`, `EncoderNormalizer`, `MultiNormalizer`, `NaNLabelEncoder`) | `ARCHITECTURE.md §5 — Transformation Layers` |
| `__getitem__` output dict and its fields | `ARCHITECTURE.md §5 — Model Interaction` |
| DataModule v2 vs v1 DataLoader construction | `ARCHITECTURE.md §9 — Under Active Development` |

### Files to Read

1. **`pytorch_forecasting/data/timeseries/_timeseries.py`** — the main
   `TimeSeriesDataSet` class. Focus on:
   - `__init__()` — parameter list and validation logic
   - `_construct_index()` — how per-sample slices are built
   - `__getitem__()` — what each batch key contains
   - `get_parameters()` — metadata extracted by `from_dataset()`
2. **`pytorch_forecasting/data/timeseries/_timeseries_v2.py`** — the experimental
   v2 dataset; note how it differs from v1.
3. **`pytorch_forecasting/data/encoders.py`** — all normalizer classes. Study:
   - `TorchNormalizer.fit()` and `transform()`
   - `GroupNormalizer` — how per-group statistics are stored and applied
   - `EncoderNormalizer` — why it re-fits per sample
   - `NaNLabelEncoder` — ordinal encoding with explicit NaN class
4. **`pytorch_forecasting/data/data_module.py`** — `TimeSeriesDataModule` (v1
   Lightning DataModule wrapper).
5. **`pytorch_forecasting/data/_tslib_data_module.py`** — experimental v2
   `EncoderDecoderTimeSeriesDataModule`.
6. **`pytorch_forecasting/data/examples.py`** — synthetic dataset generators
   used in tests and tutorials.
7. **`tests/test_data/`** — read the test fixtures in `conftest.py` to see
   what dataset configurations are exercised.

### Hands-On Tasks

```python
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# Build a tiny dataset from scratch
df = pd.DataFrame({
    "time_idx": list(range(30)) * 2,
    "series":   ["A"] * 30 + ["B"] * 30,
    "value":    list(range(30)) + list(range(10, 40)),
    "price":    [1.0] * 60,
})

dataset = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="value",
    group_ids=["series"],
    max_encoder_length=10,
    max_prediction_length=5,
    time_varying_unknown_reals=["value"],
    time_varying_known_reals=["price"],
    target_normalizer=GroupNormalizer(groups=["series"]),
)

# Inspect a single sample
sample = dataset[0]
print({k: (v.shape if hasattr(v, "shape") else v) for k, v in sample.items()})

# Inspect target_scale — the normalizer parameters stored per sample
# (used during denormalization after prediction)
print("target_scale:", sample["target_scale"])   # FloatTensor: [center, scale]
```

- Experiment with changing `target_normalizer` between `TorchNormalizer`,
  `GroupNormalizer`, and `EncoderNormalizer`. Observe how `target_scale`
  changes in the batch dict.
- Try introducing missing values and different `fill_strategy` options.

### Key Questions to Answer by End of Day 2

1. Why does `GroupNormalizer` prevent data leakage across series?
2. What is `target_scale` in the batch dict and how is it used during
   denormalization?
3. How does `_construct_index()` handle variable-length encoder histories?

---

## Day 3 — Metrics Layer: Losses, Distributions & Quantiles

**Goal:** Understand the full metrics hierarchy — from point losses to full
probabilistic distribution losses — and how the metric object is the sole
interface between model outputs and training objectives.

### Concepts to Understand

| Concept | Where to Look |
|---------|---------------|
| `Metric` abstract base class | `ARCHITECTURE.md §3 — Base Classes` |
| `MultiHorizonMetric` — per-horizon weighting | `ARCHITECTURE.md §3` |
| `DistributionLoss` — parametric distributions | `ARCHITECTURE.md §3` |
| `MultivariateDistributionLoss` — joint distributions | `ARCHITECTURE.md §3` |
| `MultiLoss` and `CompositeMetric` — composite objectives | `ARCHITECTURE.md §3` |
| `to_prediction()` / `to_quantiles()` polymorphism | `ARCHITECTURE.md §3 — How Polymorphism Is Used` |
| Strategy pattern: swapping losses without changing model code | `ARCHITECTURE.md §6 — Strategy Pattern` |

### Files to Read

1. **`pytorch_forecasting/metrics/base_metrics/_base_metrics.py`** — study
   every abstract method:
   - `loss(y_pred, target)` — per-element loss computation
   - `to_prediction(y_pred)` — convert raw output to point prediction
   - `to_quantiles(y_pred, quantiles)` — produce quantile bands
   - `rescale_parameters(parameters, target_scale, ...)` — denormalization
2. **`pytorch_forecasting/metrics/point.py`** — `MAE`, `MAPE`, `RMSE`, `SMAPE`,
   `MASE`. These are the simplest metrics to understand first.
3. **`pytorch_forecasting/metrics/quantile.py`** — `QuantileLoss`. Note how
   the pinball loss is computed and how `to_quantiles()` returns the multiple
   quantile bands.
4. **`pytorch_forecasting/metrics/distributions.py`** — `NormalDistributionLoss`,
   `NegativeBinomialDistributionLoss`. Trace:
   - `map_x_to_distribution(x)` — raw model output → `torch.distributions` object
   - `loss()` using negative log-likelihood
   - `to_prediction()` — returning the distribution mean
   - `sample()` — Monte Carlo sampling
5. **`pytorch_forecasting/metrics/_mqf2_utils.py`** — helper for the
   `MQF2DistributionLoss`, an advanced implicit quantile network loss.
6. **`tests/test_metrics.py`** — understand what properties each metric is
   tested for; use these as a specification.

### Hands-On Tasks

```python
import torch
from pytorch_forecasting.metrics import MAE, QuantileLoss, NormalDistributionLoss

# Point metric
mae = MAE()
y_pred = torch.tensor([[1.0, 2.0, 3.0]])
y_true = torch.tensor([[1.5, 2.5, 3.5]])
print(mae.loss(y_pred, y_true))
print(mae.to_prediction(y_pred))

# Quantile metric
ql = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
# y_pred must have shape [batch, time, n_quantiles]
y_pred_q = torch.randn(2, 5, 3)
y_true_q = torch.randn(2, 5)
print(ql.loss(y_pred_q, y_true_q))

# Distribution metric
ndl = NormalDistributionLoss()
# y_pred must have shape [batch, time, 2] — (mu, log_sigma)
y_pred_d = torch.randn(2, 5, 2)
dist = ndl.map_x_to_distribution(y_pred_d)
print(dist.mean, dist.stddev)
print(ndl.to_prediction(y_pred_d))
print(ndl.to_quantiles(y_pred_d, quantiles=[0.1, 0.5, 0.9]).shape)
```

- Implement a minimal `HuberLoss` by extending `MultiHorizonMetric` (see
  `ARCHITECTURE.md §12 — Adding a New Point / Quantile Metric`).
- Verify your metric passes the same structural tests in `test_metrics.py`.

### Key Questions to Answer by End of Day 3

1. What must `map_x_to_distribution()` return, and why must it return a
   `torch.distributions` object?
2. How does `MultiLoss` handle per-target objectives in multi-target settings?
3. What is the difference between `loss()` and `compute()` on a `Metric`?

---

## Day 4 — Model Base Classes & Training / Prediction Pipeline

**Goal:** Deeply understand `BaseModel`, `BaseModelWithCovariates`, and the
`AutoRegressiveBaseModel` — the scaffolding on which every model in the library
is built — and trace a full training and prediction pass end-to-end.

### Concepts to Understand

| Concept | Where to Look |
|---------|---------------|
| `BaseModel` lifecycle methods | `ARCHITECTURE.md §3 — Base Classes` |
| `from_dataset()` factory pattern | `ARCHITECTURE.md §3`, `§6 — Factory Pattern` |
| Template method pattern in `training_step()` | `ARCHITECTURE.md §6 — Template Method Pattern` |
| Full training flow | `ARCHITECTURE.md §4 — Training` |
| Full prediction flow | `ARCHITECTURE.md §4 — Prediction` |
| Output formatting (point / quantile / distribution) | `ARCHITECTURE.md §4 — Output Generation` |
| Auto-regressive decoding | `ARCHITECTURE.md §4 — Training`, `§3 — AutoRegressiveBaseModel` |
| Covariate embedding via `MultiEmbedding` | `ARCHITECTURE.md §3 — BaseModelWithCovariates` |

### Files to Read

1. **`pytorch_forecasting/models/base/_base_model.py`** — this is the largest
   and most important file. Read it in sections:
   - `__init__()` and `save_hyperparameters()` — how kwargs flow through
   - `from_dataset()` — the factory contract
   - `training_step()` / `validation_step()` — the template method
   - `predict_step()` and `predict()` — multi-mode prediction
   - `to_prediction()` / `to_quantiles()` — output conversion
   - `to_network_output()` — how models return predictions
   - `plot_prediction()` / `plot_interpretation()` — visualization hooks
2. **`pytorch_forecasting/models/base/_base_model_v2.py`** — the experimental
   v2 `BaseModel`. Compare its `training_step()` with v1 — note the
   simplification in the optimizer configuration.
3. **`pytorch_forecasting/models/base/_tslib_base_model_v2.py`** — the tslib
   adapter; note how `metadata` replaces `from_dataset()`.
4. **`pytorch_forecasting/models/base/_base_object.py`** — `_BasePtForecaster`
   and `_BasePtForecaster_v2` wrapper classes.
5. **`pytorch_forecasting/callbacks/`** — `PredictCallback` and how it
   collects outputs across batches.
6. **`pytorch_forecasting/utils/`** — `InitialParameterRepresenterMixIn`,
   `OutputMixIn`, masking utilities used by the base model.

### Hands-On Tasks

```python
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
import lightning.pytorch as pl

# Build dataset (reuse Day 2 setup or use the Stallion example)
# ...

# Instantiate via factory
model = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
)

# Inspect how from_dataset read dataset metadata
print(model.hparams)

# Train for a few steps to watch the template method in action
trainer = pl.Trainer(max_epochs=1, limit_train_batches=3, enable_progress_bar=False)
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
```

- Set a breakpoint inside `BaseModel.training_step()` and step through it
  to trace `forward()` → `loss.loss()` → `log()`.
- Repeat with `DeepAR` to see how `AutoRegressiveBaseModel` differs in
  its prediction loop.

### Key Questions to Answer by End of Day 4

1. What is the contract of `forward()` — what must it return?
2. How does `from_dataset()` know which covariate sizes to use without
   the caller specifying them?
3. What is the difference between how `BaseModel` and
   `AutoRegressiveBaseModel` generate multi-step predictions?

---

## Day 5 — Model Zoo: Exploring Key Architectures

**Goal:** Survey the breadth of the model zoo, understand the architectural
choices of the major models, and see how each uses the base class infrastructure.

### Models to Study

Work through models in increasing architectural complexity:

#### 5a. Baseline and MLP (Simplest)

| File | What to Learn |
|------|---------------|
| `pytorch_forecasting/models/baseline.py` | The simplest possible model; how `forward()` can be minimal |
| `pytorch_forecasting/models/mlp/_decoder_mlp.py` | Fully-connected decoder; how `extract_features()` is used |

#### 5b. Recurrent Models

| File | What to Learn |
|------|---------------|
| `pytorch_forecasting/models/rnn/_rnn.py` | `RecurrentNetwork`; dual AR / non-AR variant controlled by `cell_type` |
| `pytorch_forecasting/models/deepar/_deepar.py` | `DeepAR` — distributional auto-regressive forecasting; `output_to_prediction()` |

#### 5c. N-BEATS / N-HiTS (Decomposition-based)

| File | What to Learn |
|------|---------------|
| `pytorch_forecasting/models/nbeats/_nbeats.py` | `NBeats` — doubly-residual stacks; `NBeatsAdapter` role |
| `pytorch_forecasting/models/nbeats/_nbeats_kan.py` | `NBeatsKAN` — KAN layers replacing MLP blocks |
| `pytorch_forecasting/models/nhits/_nhits.py` | `NHiTS` — multi-rate sampling and interpolation |

#### 5d. Temporal Fusion Transformer (Flagship)

| File | What to Learn |
|------|---------------|
| `pytorch_forecasting/models/temporal_fusion_transformer/_tft.py` | Full TFT — variable selection, LSTM encoder, attention decoder |
| `pytorch_forecasting/models/temporal_fusion_transformer/_tft_v2.py` | Experimental v2 port; compare initialization and forward pass |
| `examples/stallion.ipynb` | End-to-end TFT training and interpretation tutorial |

#### 5e. Modern / tslib-based Models

| File | What to Learn |
|------|---------------|
| `pytorch_forecasting/models/dlinear/_dlinear.py` | `DLinear` — linear decomposition; simplest v2 model |
| `pytorch_forecasting/models/tide/_tide.py` | `TiDE` — dense encoder-decoder for long-horizon forecasting |
| `pytorch_forecasting/models/timexer/_timexer.py` | `TimesXer` — patch-based exogenous transformer |
| `pytorch_forecasting/models/samformer/_samformer.py` | `SAMformer` — sharpness-aware minimization transformer |
| `pytorch_forecasting/models/xlstm/_xlstm.py` | `xLSTM` — extended LSTM with exponential gating |

### Cross-Cutting Comparison Tasks

After reading each model, fill in this comparison table for your notes:

| Model | Base Class | Loss Default | Auto-Regressive? | Handles Covariates? | Key Innovation |
|-------|-----------|-------------|-----------------|--------------------|-|
| Baseline | `BaseModel` | MAE | No | No | Naive benchmark |
| DecoderMLP | `BaseModelWithCovariates` | MAE | No | Yes | Simple feedforward |
| RNN | `BaseModelWithCovariates` / AR | MAE | Optional | Yes | LSTM/GRU encoder |
| DeepAR | `AutoRegressiveBaseModelWithCovariates` | NormalDist | Yes | Yes | Probabilistic AR |
| NBeats | `BaseModel` (via Adapter) | MAE | No | No | Doubly-residual |
| NHiTS | `BaseModelWithCovariates` | MAE | No | Yes | Multi-rate sampling |
| TFT | `BaseModelWithCovariates` | QuantileLoss | No | Yes | Variable selection + attention |
| DLinear | `TslibBaseModel` (v2) | MAE | No | Yes | Linear decomposition |

### Hands-On Tasks

```python
# Compare TFT and NHiTS on the same dataset
from pytorch_forecasting import NHiTS

nhits = NHiTS.from_dataset(training, learning_rate=1e-3, hidden_size=32)
trainer.fit(nhits, train_dataloaders=train_dl, val_dataloaders=val_dl)

# Try a probabilistic model
from pytorch_forecasting import DeepAR
from pytorch_forecasting.metrics import NormalDistributionLoss

deepar = DeepAR.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=16,
    loss=NormalDistributionLoss(),
)
trainer.fit(deepar, ...)
```

### Key Questions to Answer by End of Day 5

1. Why does `NBeats` need an `NBeatsAdapter` intermediate class?
2. How does TFT's variable selection network decide which features matter?
3. How does `DLinear` (v2) differ structurally from `NHiTS` (v1) beyond
   the base class they inherit from?

---

## Day 6 — Layers Library & Extension Mechanisms

**Goal:** Understand the shared `layers/` building blocks, the registry and
`_pkg` wrapper pattern, and practice extending the library with a new metric
and a new model.

### Concepts to Understand

| Concept | Where to Look |
|---------|---------------|
| Layer library taxonomy | `ARCHITECTURE.md §2`, `pytorch_forecasting/layers/` |
| Adding a new model (5-step process) | `ARCHITECTURE.md §12 — Adding a New Estimator` |
| `_pkg` wrapper and registry | `ARCHITECTURE.md §6 — Adapter Pattern`, `§12` |
| Adding a distribution loss | `ARCHITECTURE.md §12 — Adding a New Probabilistic Distribution` |
| Adding a point / quantile metric | `ARCHITECTURE.md §12 — Adding a New Point / Quantile Metric` |
| Registry lookup | `ARCHITECTURE.md §6 — Registry / Lookup Pattern` |

### Files to Read

**Layers:**

| File | Contents |
|------|----------|
| `pytorch_forecasting/layers/_attention/` | Multi-head self-attention and interpretable attention used by TFT |
| `pytorch_forecasting/layers/_embeddings/` | `MultiEmbedding` — categorical embedding table |
| `pytorch_forecasting/layers/_recurrent/` | Shared LSTM/GRU cells |
| `pytorch_forecasting/layers/_normalization/` | Layer norm, group norm variants |
| `pytorch_forecasting/layers/_nbeats/` | N-BEATS stack and block definitions |
| `pytorch_forecasting/layers/_blocks/` | Shared MLP and ResNet blocks |
| `pytorch_forecasting/layers/_kan/` | Kolmogorov-Arnold Network layer (used by NBeatsKAN) |
| `pytorch_forecasting/layers/_decomposition/` | Trend/seasonality decomposition (used by DLinear) |

**Registry:**

| File | Contents |
|------|----------|
| `pytorch_forecasting/_registry/_lookup.py` | Master list of all registered estimators |
| `pytorch_forecasting/base/_base_object.py` | `_BaseObject` root from `skbase` |
| `pytorch_forecasting/models/base/_base_object.py` | `_BasePtForecaster`, `_BasePtForecaster_v2` |
| `pytorch_forecasting/tests/test_class_register.py` | Registry integrity tests |
| `pytorch_forecasting/tests/test_all_estimators.py` | Registry-driven smoke tests |

### Hands-On Tasks — Build a New Metric

Implement a **Huber loss** by extending `MultiHorizonMetric` and register it:

```python
# pytorch_forecasting/metrics/point.py  (add to existing file)
import torch.nn.functional as F
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric

class HuberLoss(MultiHorizonMetric):
    """Huber loss (smooth L1) — robust to outliers.

    Computes ``delta * (sqrt(1 + (error/delta)^2) - 1)`` element-wise,
    which behaves like L2 loss near zero and L1 loss for large errors.

    Parameters
    ----------
    delta : float, optional
        Threshold at which the loss switches from quadratic to linear.
        Default: ``1.0``.
    **kwargs
        Additional keyword arguments forwarded to ``MultiHorizonMetric``.
    """
    def __init__(self, delta: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def loss(self, y_pred, target):
        return F.huber_loss(y_pred, target, delta=self.delta, reduction="none")
```

Add it to `pytorch_forecasting/metrics/__init__.py`, then verify:

```python
from pytorch_forecasting.metrics import HuberLoss
hl = HuberLoss(delta=0.5)
print(hl.loss(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 3.0])))
```

### Hands-On Tasks — Build a New Model Skeleton

Follow the 5-step process in `ARCHITECTURE.md §12` to create a minimal
`LinearBaseline` model that predicts the last observed value repeated for
every forecast step:

1. Create `pytorch_forecasting/models/linear_baseline/_linear_baseline.py`
2. Create `pytorch_forecasting/models/linear_baseline/_linear_baseline_pkg.py`
3. Register in `_registry/_lookup.py`
4. Expose in `pytorch_forecasting/models/__init__.py`
5. Verify `test_all_estimators.py` picks it up automatically

### Key Questions to Answer by End of Day 6

1. Why does the registry use lazy string imports instead of importing classes
   directly?
2. What tags must a `_pkg` wrapper expose for the smoke-test infrastructure
   to work?
3. How does `MultiEmbedding` handle variable-cardinality categorical features
   from different dataset configurations?

---

## Day 7 — Testing, Performance, and the Contribution Path

**Goal:** Understand the test architecture, identify performance bottlenecks,
navigate the v1 → v2 migration, and map your own contribution path.

### Concepts to Understand

| Concept | Where to Look |
|---------|---------------|
| Test structure and philosophy | `ARCHITECTURE.md §8` |
| Registry-driven test automation | `ARCHITECTURE.md §8 — How Estimators Are Verified` |
| Computational bottlenecks | `ARCHITECTURE.md §11 — Computational Bottlenecks` |
| Memory-heavy components | `ARCHITECTURE.md §11 — Memory-Heavy Components` |
| Technical debt and design inconsistencies | `ARCHITECTURE.md §10` |
| v1 → v2 migration status | `ARCHITECTURE.md §9`, `§14` |
| Contribution entry points by skill level | `ARCHITECTURE.md §13` |
| Strategic contribution priorities | `ARCHITECTURE.md §15 — Most Impactful Areas` |

### Files to Read

**Tests:**

| File | Purpose |
|------|---------|
| `tests/conftest.py` | Shared fixtures — understand what dataset scenarios are covered |
| `tests/test_data/` | `TimeSeriesDataSet` edge cases; encoders and samplers |
| `tests/test_metrics.py` | Metric contracts; shape assertions; distribution sampling |
| `tests/test_models/` | Per-model integration tests; parametrize patterns |
| `pytorch_forecasting/tests/test_all_estimators.py` | Registry-driven smoke tests |
| `pytorch_forecasting/tests/test_class_register.py` | Registry integrity |
| `pytorch_forecasting/tests/_data_scenarios.py` | Shared synthetic dataset fixtures for model tests |

**Performance / Scalability:**

| File | Bottleneck to Study |
|------|---------------------|
| `pytorch_forecasting/data/timeseries/_timeseries.py` | `__getitem__()` — per-sample CPU preprocessing |
| `pytorch_forecasting/models/base/_base_model.py` | Auto-regressive prediction loop |
| `pytorch_forecasting/layers/_attention/` | O(n²) attention; FlashAttention integration opportunity |

**v2 Migration:**

| File | Role in Migration |
|------|------------------|
| `pytorch_forecasting/models/base/_base_model_v2.py` | New base class skeleton |
| `pytorch_forecasting/models/base/_tslib_base_model_v2.py` | tslib adapter |
| `pytorch_forecasting/data/_tslib_data_module.py` | v2 DataModule |
| `pytorch_forecasting/models/temporal_fusion_transformer/_tft_v2.py` | TFT ported to v2 |
| `pytorch_forecasting/models/dlinear/_dlinear.py` | Simplest complete v2 model to study |
| `examples/ptf_V2_example.ipynb` | End-to-end v2 tutorial |
| `examples/tslib_v2_example.ipynb` | tslib v2 tutorial |

### Hands-On Tasks

```bash
# Run the full test suite to understand baseline CI health
pytest tests/ -x -q --timeout=120

# Run only registry-driven tests to validate any new model/metric you added
pytest pytorch_forecasting/tests/ -v

# Run a specific model test with verbose output
pytest tests/test_models/test_tft.py -v
```

- Read `tests/conftest.py` in full; identify every dataset scenario fixture
  and understand which edge cases each one covers.
- Profile `TimeSeriesDataSet.__getitem__()` on a medium-sized dataset (10,000
  rows, 100 series) to observe the per-sample CPU cost.
- Compare the `DLinear` v2 model to `NHiTS` v1 — write a short note on what
  the v2 API simplifies and what it gives up.

### Contribution Path Decision Tree

Use the decision tree from `ARCHITECTURE.md §15` and your week of learning to
choose your first contribution:

```
Are you comfortable with data/tests?
  ├── YES → Add an edge-case test to tests/test_data/ or tests/test_metrics.py
  │          Great first PR; builds confidence and codebase familiarity.
  └── NO  → Re-read Day 2 and run test_data/ tests with -v to build intuition.

Do you want to add new functionality?
  ├── New metric  → Extend MultiHorizonMetric or DistributionLoss (Day 6 template)
  ├── New model   → Follow the 5-step process in ARCHITECTURE.md §12
  └── New layer   → Add a reusable nn.Module to layers/ and use it in a model

Are you interested in architecture/internals?
  ├── Port a v1 model to v2 API → Follow _tft_v2.py as a template
  ├── Improve BaseModel (v1)    → Refactor or document individual methods
  └── DataModule enhancements   → Extend EncoderDecoderTimeSeriesDataModule
```

### Key Questions to Answer by End of Day 7

1. What does `create_test_instance()` do and why does every `_pkg` class
   need to implement `get_test_params()`?
2. Why is the auto-regressive prediction loop O(prediction_length) and
   what would need to change architecturally to batch-parallelize it?
3. What is the clearest signal in the codebase that indicates which v2
   component to stabilize next?

---

## Summary: Files by Day

| Day | Core Files | Supporting Files |
|-----|-----------|-----------------|
| 1 | `README.md`, `ARCHITECTURE.md §1–2`, `pytorch_forecasting/__init__.py` | `pyproject.toml`, `examples/stallion.py` |
| 2 | `data/timeseries/_timeseries.py`, `data/encoders.py` | `data/data_module.py`, `data/_tslib_data_module.py`, `data/examples.py` |
| 3 | `metrics/base_metrics/_base_metrics.py`, `metrics/point.py`, `metrics/quantile.py`, `metrics/distributions.py` | `metrics/_mqf2_utils.py`, `tests/test_metrics.py` |
| 4 | `models/base/_base_model.py` | `models/base/_base_model_v2.py`, `models/base/_base_object.py`, `callbacks/`, `utils/` |
| 5 | `models/temporal_fusion_transformer/_tft.py`, `models/nbeats/_nbeats.py`, `models/nhits/_nhits.py`, `models/deepar/_deepar.py` | `models/dlinear/`, `models/rnn/`, `models/tide/`, `models/xlstm/` |
| 6 | `layers/` (all subdirs), `_registry/_lookup.py`, `models/base/_base_object.py` | `tests/test_class_register.py`, `tests/test_all_estimators.py` |
| 7 | `tests/conftest.py`, `tests/test_models/`, `models/base/_base_model_v2.py` | `examples/ptf_V2_example.ipynb`, `ARCHITECTURE.md §10–15` |

---

## Quick Reference: Key Concepts Across Days

| Concept | Primary Day | File |
|---------|-------------|------|
| `TimeSeriesDataSet.__getitem__()` | 2 | `data/timeseries/_timeseries.py` |
| `GroupNormalizer` / `TorchNormalizer` | 2 | `data/encoders.py` |
| `Metric.loss()` / `to_prediction()` | 3 | `metrics/base_metrics/_base_metrics.py` |
| `DistributionLoss.map_x_to_distribution()` | 3 | `metrics/distributions.py` |
| `BaseModel.training_step()` | 4 | `models/base/_base_model.py` |
| `BaseModel.from_dataset()` | 4 | `models/base/_base_model.py` |
| Auto-regressive decoding | 4–5 | `models/deepar/_deepar.py` |
| TFT variable selection | 5 | `models/temporal_fusion_transformer/_tft.py` |
| `MultiEmbedding` | 6 | `layers/_embeddings/` |
| `_BasePtForecaster` + `_registry` | 6 | `_registry/_lookup.py`, `models/base/_base_object.py` |
| Registry smoke tests | 7 | `pytorch_forecasting/tests/test_all_estimators.py` |
| v2 base model | 4, 7 | `models/base/_base_model_v2.py` |

---

*This roadmap was created from the context in [`ARCHITECTURE.md`](./ARCHITECTURE.md).
Keep that document open as your companion — it contains the full architectural
analysis, design patterns, dependency graph, technical debt notes, and contribution
guide that this roadmap references throughout.*
