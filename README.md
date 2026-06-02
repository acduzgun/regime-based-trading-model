# Regime-Based Trading Model

Research notebooks for comparing regime-aware return prediction models:

- Base linear model
- Mixture-of-experts model
- Pooled model
- Transformer model
- XGBoost model

The raw dataset is not included. To reproduce the notebooks, place the input parquet file at:

```text
data/model_data.parquet
```

Expected columns:

- `trade_date`: timestamp or date-like column
- `ret_fopen`: target return column
- `x*`: feature columns
- `cond*`: regime/conditioning columns

For Modal-backed notebooks, upload the local data file once:

```bash
python modal_train.py upload
modal deploy modal_train.py
```

Then run the notebooks from the `notebooks/` directory.
