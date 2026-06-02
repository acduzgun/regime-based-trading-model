from __future__ import annotations

import json
from typing import List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor

from src.preprocessing import WindowSpec, build_yearly_schedule, prepare_arrays

_DEFAULT_XGB_PARAMS = {
    'max_depth': 4,
    'n_estimators': 500,
    'learning_rate': 0.05,
    'subsample': 0.6,
    'colsample_bytree': 0.8,
    'min_child_weight': 100,
    'reg_alpha': 0.1,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'verbosity': 0,
}


def _mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def _annualized_sharpe(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    dates: np.ndarray,
    annualization: int = 252,
) -> float:
    """Annualized daily Sharpe matching backtest(): clip pred, sum PnL by day, sqrt(252)*mean/std."""
    alpha = np.clip(y_pred, -3.0, 3.0)
    pnl = alpha * y_true
    daily = pd.Series(pnl, index=pd.to_datetime(dates).normalize()).groupby(level=0).sum()
    std = daily.std()
    if std < 1e-10:
        return 0.0
    return float(np.sqrt(annualization) * daily.mean() / std)


class _SharpeStopping(xgb.callback.TrainingCallback):
    """Tracks per-round MSE + annualized daily Sharpe on inner train and inner val.
    Early stops when inner val Sharpe fails to improve for `rounds` consecutive rounds.
    """

    def __init__(
        self,
        rounds: int,
        dtrain: xgb.DMatrix,
        dval: xgb.DMatrix,
        y_tr: np.ndarray,
        y_iv: np.ndarray,
        dates_tr: np.ndarray,
        dates_iv: np.ndarray,
        verbose: bool = False,
    ):
        super().__init__()
        self.rounds = rounds
        self.dtrain = dtrain
        self.dval = dval
        self.y_tr = y_tr
        self.y_iv = y_iv
        self.dates_tr = dates_tr
        self.dates_iv = dates_iv
        self.verbose = verbose

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_sharpes: List[float] = []
        self.val_sharpes: List[float] = []
        self.best_sharpe: float = -np.inf
        self.best_iteration: int = 0
        self._wait: int = 0

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        tr_pred = model.predict(self.dtrain)
        iv_pred = model.predict(self.dval)

        tr_mse = _mse(tr_pred, self.y_tr)
        iv_mse = _mse(iv_pred, self.y_iv)
        tr_sh = _annualized_sharpe(tr_pred, self.y_tr, self.dates_tr)
        iv_sh = _annualized_sharpe(iv_pred, self.y_iv, self.dates_iv)

        self.train_losses.append(tr_mse)
        self.val_losses.append(iv_mse)
        self.train_sharpes.append(tr_sh)
        self.val_sharpes.append(iv_sh)

        if self.verbose:
            print(
                f'[{epoch + 1:4d}]  train_mse={tr_mse:.6f}  iv_mse={iv_mse:.6f}'
                f'  train_sharpe={tr_sh:.4f}  iv_sharpe={iv_sh:.4f}',
                flush=True,
            )

        if iv_sh > self.best_sharpe:
            self.best_sharpe = iv_sh
            self.best_iteration = epoch
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.rounds:
                print(
                    f'Early stopping at round {epoch + 1}; '
                    f'best round {self.best_iteration + 1} '
                    f'(iv_sharpe={self.best_sharpe:.4f})',
                    flush=True,
                )
                return True

        return False


def train_predict_fixed_split_xgboost(
    df: pd.DataFrame,
    x_cols: List[str],
    cond_cols: List[str],
    train_end_year: int = 2021,
    train_end_date: Optional[str] = None,
    val_years: Optional[List[int]] = None,
    val_start: Optional[str] = None,
    val_end: Optional[str] = None,
    val_end_date: Optional[str] = None,
    target_col: str = 'ret_fopen',
    date_col: str = 'msgStamp',
    trade_date_col: str = 'trade_date',
    xgb_params: Optional[dict] = None,
    x_clip: float = 3.0,
    y_clip_quantile: Optional[float] = 0.01,
    early_stopping_rounds: int = 25,
    random_state: int = 42,
    verbose: bool = False,
) -> pd.DataFrame:
    """Train once on historical data; predict a held-out validation/test window.

    Intended for fast HP search. Training can end by whole year
    (train_end_year) or explicit inclusive date (train_end_date). The
    prediction window can be selected by explicit inclusive date range
    (val_start/val_end) or by val_years. Inner val = last 25% of the
    training rows chronologically.
    Early stopping on inner val Sharpe (not MSE) via _SharpeStopping callback.
    Predictions at best inner-val-Sharpe round via iteration_range.
    Predictions normalised by train_pred.std().
    Returns DataFrame with .attrs['eval_results', 'inner_val_preds', 'best_iteration'].
    """
    if val_years is None:
        val_years = [2022, 2023]
    if xgb_params is None:
        xgb_params = _DEFAULT_XGB_PARAMS.copy()

    print(f'XGB params: {xgb_params}', flush=True)

    feature_cols = x_cols + cond_cols

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)
    data['_year'] = data[date_col].dt.year
    if trade_date_col not in data.columns:
        data[trade_date_col] = data[date_col].dt.date

    data_dates = pd.to_datetime(data[date_col])
    if data_dates.dt.tz is not None:
        data_dates = data_dates.dt.tz_convert(None)
    if train_end_date is not None:
        train_cutoff = pd.Timestamp(train_end_date)
        train_mask_all = data_dates < train_cutoff + pd.Timedelta(days=1)
        train_end_label = train_cutoff.date().isoformat()
    else:
        train_mask_all = data['_year'] <= train_end_year
        train_end_label = str(train_end_year)

    train_df = data[train_mask_all].dropna(
        subset=feature_cols + [target_col]
    ).copy()
    if val_start is not None or val_end is not None:
        val_mask = pd.Series(True, index=data.index)
        if val_start is not None:
            val_mask &= data_dates >= pd.Timestamp(val_start)
        if val_end is not None:
            val_mask &= data_dates < pd.Timestamp(val_end) + pd.Timedelta(days=1)
    else:
        val_mask = data['_year'].isin(val_years)
        if val_end_date is not None:
            val_mask &= data_dates < pd.Timestamp(val_end_date)
    val_df = data[val_mask].dropna(subset=feature_cols).copy()

    if len(train_df) == 0 or len(val_df) == 0:
        return pd.DataFrame(columns=[date_col, trade_date_col, 'pred'])

    X_train, X_val, C_train, C_val, y_train = prepare_arrays(
        train_df, val_df, x_cols, cond_cols, target_col, x_clip, y_clip_quantile
    )
    XC_train = np.hstack([X_train, C_train])
    XC_val = np.hstack([X_val, C_val])

    n = len(train_df)
    split_idx = int(n * 0.75)
    iv_mask = np.zeros(n, dtype=bool)
    iv_mask[split_idx:] = True
    it_mask = ~iv_mask
    dates_all = train_df[trade_date_col].values
    XC_tr, y_tr, dates_tr = XC_train[it_mask], y_train[it_mask], dates_all[it_mask]
    XC_iv, y_iv, dates_iv = XC_train[iv_mask], y_train[iv_mask], dates_all[iv_mask]
    if len(XC_tr) == 0:
        XC_tr, y_tr, dates_tr = XC_train, y_train, dates_all
        XC_iv, y_iv, dates_iv = XC_train, y_train, dates_all

    params = {**xgb_params, 'random_state': random_state}
    n_est = params.pop('n_estimators', 500)

    # Build low-level xgb.train() param dict; 'random_state' is sklearn-only → 'seed'
    xgb_low = {k: v for k, v in params.items()}
    xgb_low.pop('random_state', None)
    xgb_low['seed'] = random_state

    # Each x feature can only interact with cond features — never with other x features.
    # Low-level xgb.train expects interaction constraints as integer feature indices
    # in this XGBoost version, even when DMatrix feature_names are provided.
    feature_names = [f'f{i}' for i in range(len(x_cols) + len(cond_cols))]
    cond_indices = list(range(len(x_cols), len(x_cols) + len(cond_cols)))
    constraints = [[i] + cond_indices for i in range(len(x_cols))]
    xgb_low['interaction_constraints'] = json.dumps(constraints)
    print(f'interaction_constraints: {constraints}', flush=True)

    dtrain = xgb.DMatrix(XC_tr, label=y_tr, feature_names=feature_names)
    dval = xgb.DMatrix(XC_iv, label=y_iv, feature_names=feature_names)

    cb = _SharpeStopping(
        rounds=early_stopping_rounds,
        dtrain=dtrain,
        dval=dval,
        y_tr=y_tr,
        y_iv=y_iv,
        dates_tr=dates_tr,
        dates_iv=dates_iv,
        verbose=verbose,
    )

    booster = xgb.train(
        xgb_low,
        dtrain,
        num_boost_round=n_est,
        callbacks=[cb],
        verbose_eval=False,
    )

    best_round = cb.best_iteration + 1
    print(
        f'Training done. Best round: {best_round}/{n_est}  '
        f'iv_sharpe={cb.best_sharpe:.4f}',
        flush=True,
    )

    d_full = xgb.DMatrix(XC_train, feature_names=feature_names)
    d_val_out = xgb.DMatrix(XC_val, feature_names=feature_names)
    d_iv_out = xgb.DMatrix(XC_iv, feature_names=feature_names)

    train_pred = booster.predict(d_full, iteration_range=(0, best_round))
    pred_val = booster.predict(d_val_out, iteration_range=(0, best_round)) / (train_pred.std() + 1e-12)

    iv_df = train_df[iv_mask][[date_col, trade_date_col]].copy()
    iv_df['pred'] = booster.predict(d_iv_out, iteration_range=(0, best_round)) / (train_pred.std() + 1e-12)
    iv_df['train_end_year'] = train_end_year
    iv_df['train_end_date'] = train_end_label
    inner_val_preds = iv_df.sort_values(date_col).reset_index(drop=True)

    out = val_df[[date_col, trade_date_col]].copy()
    out['pred'] = pred_val
    out['train_end_year'] = train_end_year
    out['train_end_date'] = train_end_label

    result = out.sort_values(date_col).reset_index(drop=True)
    result.attrs['eval_results'] = {
        'train': {'mse': cb.train_losses, 'sharpe': cb.train_sharpes},
        'inner_val': {'mse': cb.val_losses, 'sharpe': cb.val_sharpes},
    }
    result.attrs['inner_val_preds'] = inner_val_preds
    result.attrs['best_iteration'] = cb.best_iteration
    result.attrs['best_val_sharpe'] = cb.best_sharpe
    result.attrs['train_end_year'] = train_end_year
    result.attrs['train_end_date'] = train_end_label
    result.attrs['val_start'] = result[date_col].min() if len(result) else None
    result.attrs['val_end'] = result[date_col].max() if len(result) else None
    result.attrs['val_years'] = sorted(result[date_col].dt.year.unique().tolist()) if len(result) else []
    return result


def walk_forward_xgboost(
    df: pd.DataFrame,
    x_cols: List[str],
    cond_cols: List[str],
    target_col: str = 'ret_fopen',
    date_col: str = 'msgStamp',
    trade_date_col: str = 'trade_date',
    window_spec: Optional[WindowSpec] = None,
    xgb_params: Optional[dict] = None,
    x_clip: float = 3.0,
    y_clip_quantile: Optional[float] = 0.01,
    early_stopping_rounds: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Walk-forward XGBoost with cond cols included as features.

    Cond cols are appended to x_cols — trees learn regime splits implicitly.
    Early stopping uses the last training year as inner validation.
    Predictions are normalized by train_pred.std() for cross-period stability.
    """
    if window_spec is None:
        window_spec = WindowSpec(mode='expanding')
    if xgb_params is None:
        xgb_params = _DEFAULT_XGB_PARAMS.copy()

    feature_cols = x_cols + cond_cols

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)
    data['year'] = data[date_col].dt.to_period('Y')
    if trade_date_col not in data.columns:
        data[trade_date_col] = data[date_col].dt.date

    schedule = build_yearly_schedule(data, date_col, window_spec)
    outputs = []
    last_eval_results: dict = {}

    for step in schedule:
        train_years, pred_year = step['train_years'], step['pred_year']

        train_df = data.loc[data['year'].isin(train_years)].dropna(
            subset=feature_cols + [target_col]
        ).copy()
        test_df = data.loc[data['year'] == pred_year].dropna(subset=feature_cols).copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train, X_test, C_train, C_test, y_train = prepare_arrays(
            train_df, test_df, x_cols, cond_cols, target_col, x_clip, y_clip_quantile
        )
        XC_train = np.hstack([X_train, C_train])
        XC_test = np.hstack([X_test, C_test])

        iv_mask = (train_df['year'] == train_years[-1]).values
        it_mask = ~iv_mask
        XC_tr, y_tr = XC_train[it_mask], y_train[it_mask]
        XC_iv, y_iv = XC_train[iv_mask], y_train[iv_mask]
        if len(XC_tr) == 0:
            XC_tr, y_tr = XC_train, y_train
            XC_iv, y_iv = XC_train, y_train

        params = {**xgb_params, 'random_state': random_state}
        n_est = params.pop('n_estimators', 500)

        model = XGBRegressor(
            n_estimators=n_est,
            early_stopping_rounds=early_stopping_rounds,
            **params,
        )
        model.fit(XC_tr, y_tr, eval_set=[(XC_iv, y_iv)], verbose=False)
        last_eval_results = model.evals_result()

        train_pred = model.predict(XC_train)
        pred_test = model.predict(XC_test) / (train_pred.std() + 1e-12)

        out = test_df[[date_col, trade_date_col]].copy()
        out['pred'] = pred_test
        out['pred_year'] = str(pred_year)
        out['train_start_year'] = str(train_years[0])
        out['train_end_year'] = str(train_years[-1])
        outputs.append(out)

    if not outputs:
        return pd.DataFrame(columns=[date_col, trade_date_col, 'pred'])

    result = pd.concat(outputs).sort_values(date_col).reset_index(drop=True)
    result.attrs['eval_results'] = last_eval_results
    return result
