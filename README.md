# Regime-Based Trading Model

Research notebooks for comparing regime-aware return prediction models in a low
signal-to-noise financial forecasting setting. The central modeling question is
whether regime variables should be treated differently from the direct predictive
features.

- Base linear model
- Mixture-of-experts model
- Pooled model
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
- Target outliers: `ret_fopen` is winsorized on the training window using the
  1st and 99th percentiles.
- Regime variables: `cond*` variables are winsorized on the training window using
  the 1st and 99th percentiles, standardized using training-window mean/std, and
  finally clipped to `[-5, 5]`.
- Backtest signal: predictions are clipped to `[-3, 3]` before computing daily
  PnL and Sharpe.

## Modeling Assumption

The dataset is treated as a low signal-to-noise problem, so the baseline emphasis
is on regularized linear structure and stable walk-forward validation rather than
large unconstrained models.

The richer models impose the prior that:

- feature variables carry the direct return-prediction signal;
- regime variables affect how those feature signals should be pooled,
  gated, split, or attended to.

Examples:

- The pooled and mixture-of-experts notebooks use regime variables to define or
  learn conditional model structure around simpler linear predictors.
- The transformer notebook tests architectures where feature tokens and regime
  tokens are separate, with the selected architecture using one-way
  regime-to-feature attention.
- The XGBoost notebook uses interaction constraints so each feature can interact
  with regime variables, while feature-feature interactions are restricted.

For Modal-backed notebooks, upload the local data file once:

```bash
python modal_train.py upload
modal deploy modal_train.py
```

Then run the notebooks from the `notebooks/` directory.
