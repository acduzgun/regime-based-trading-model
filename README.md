# Regime-Based Trading Model

Research notebooks for comparing regime-aware return prediction models in a low
signal-to-noise financial forecasting setting. The central modeling question is
whether regime variables should be treated differently from the direct predictive
features.

- Base linear model
- Mixture-of-experts model
- Partial-pooling regime ridge model
- Transformer model
- XGBoost model

The main empirical takeaway is that simple, strongly regularized models are often
competitive in this setting. I also tested transformer and XGBoost models, but
with structural assumptions that separate feature variables from regime variables
instead of treating every input column as interchangeable.

The raw dataset is not included. To reproduce the notebooks, place the input parquet file at:

```text
data/model_data.parquet
```

Expected columns:

- `trade_date`: timestamp or date-like column
- `ret_fopen`: target return column
- `x*`: feature columns used as direct return-prediction signals
- `cond*`: regime/conditioning columns used to change how feature signals are interpreted

## Data Treatment

The notebooks use a walk-forward protocol: preprocessing parameters are estimated
on each training window and then applied to the held-out window.

- Missing regime values: `cond2` and `cond3` are forward-filled and then
  backward-filled for any leading missing rows.
- Missing modeling rows: training rows with missing feature, regime, or target
  values are dropped; prediction rows with missing feature/regime values are
  dropped.
- Feature outliers: `x*` features are clipped to `[-3, 3]`.
- Training target outliers: `ret_fopen` is winsorized only inside the training
  window used to fit the model, using the 1st and 99th percentiles. This is used
  to reduce the influence of extreme target observations during estimation.
- Regime variables: `cond*` variables are winsorized on the training window using
  the 1st and 99th percentiles, standardized using training-window mean/std, and
  finally clipped to `[-5, 5]`.
- Backtest returns: realized `ret_fopen` values are not clipped or winsorized in
  the backtest.
- Backtest signal: predictions are clipped to `[-3, 3]` before computing daily
  PnL and Sharpe, so clipping applies to position sizing rather than realized
  returns.

## Modeling Assumption

The dataset is treated as a low signal-to-noise problem, so the baseline emphasis
is on regularized linear structure and stable walk-forward validation rather than
large unconstrained models.

The richer models impose the prior that:

- feature variables carry the direct return-prediction signal;
- regime variables affect how those feature signals should be pooled,
  gated, split, or attended to.

Examples:

- The partial-pooling regime ridge notebook uses regime variables to create
  quantile groups. Each group gets a regularized deviation from a shared global
  linear model, so the model can adapt to regimes without fitting a completely
  separate model per regime bucket.
- The mixture-of-experts notebook learns a soft regime-dependent combination of
  simpler linear experts.
- The transformer notebook tests architectures where feature tokens and regime
  tokens are separate, with the selected architecture using one-way
  regime-to-feature attention.
- The XGBoost notebook uses interaction constraints so each feature can interact
  with regime variables, while feature-feature interactions are restricted.

## Partial-Pooling Regime Ridge

The pooled model is better described as a partial-pooling regime ridge model. It
is not a simple average of models. For a chosen regime variable, the training
data is split into quantile buckets using training-window quantiles. The model
then fits:

```text
prediction = x * beta_shared + x * delta_regime_bucket
```

`beta_shared` is the global linear signal shared across all observations.
`delta_regime_bucket` is the regime-specific adjustment. Both pieces are ridge
regularized, with separate penalties for the shared component and the
regime-specific deviations. This gives a compromise between one global linear
model and fully separate models per regime bucket.

## Train, Validation, And Test Protocol

The notebooks use two related protocols.

For the linear, partial-pooling regime ridge, and mixture-of-experts notebooks,
evaluation is walk-forward by calendar year. With the default expanding-window
setup, the model starts after at least four training years are available. For
each prediction year, it trains on all earlier eligible years, predicts the next
year, then expands the training set to include that year before moving forward.
This gives out-of-sample yearly predictions while preserving time order.

For the transformer and XGBoost notebooks, the main model-selection protocol is
a fixed split:

- Training period: all rows through `2021-12-31`.
- Outer validation period: `2022-01-01` through `2023-06-30`.
- Inner validation: a chronological tail slice from the training period, used
  for early stopping and learning-curve diagnostics.
- Test period: starts on `2023-07-01`.

After selecting the model/hyperparameters on the outer validation period, the
final test run retrains the selected configuration through `2023-06-30` and then
predicts rows from `2023-07-01` onward. Thus validation data is allowed into the
final training set only after model selection is complete; test rows remain held
out.

For Modal-backed notebooks, upload the local data file once:

```bash
python modal_train.py upload
modal deploy modal_train.py
```

Then run the notebooks from the `notebooks/` directory.
