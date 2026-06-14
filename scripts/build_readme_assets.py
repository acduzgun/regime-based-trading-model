from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, Ridge

from src.backtest import backtest
from src.models.xgboost_model import train_predict_fixed_split_xgboost


DATE_COL = "msgStamp"
TRADE_DATE_COL = "trade_date"
TARGET_COL = "ret_fopen"


@dataclass
class WindowSpec:
    mode: str = "expanding"
    train_years: int = 4
    min_train_years: int = 4


def _sort_model_cols(cols: Iterable[str]) -> list[str]:
    def key(col: str) -> tuple[str, int | str]:
        prefix = "".join(ch for ch in col if not ch.isdigit())
        suffix = col[len(prefix):]
        return prefix, int(suffix) if suffix.isdigit() else suffix

    return sorted(cols, key=key)


def _load_data(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    df = pd.read_parquet(path)
    if DATE_COL not in df.columns:
        raise ValueError(f"Missing required timestamp column: {DATE_COL}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_COL}")

    x_cols = _sort_model_cols([c for c in df.columns if c.startswith("x")])
    cond_cols = _sort_model_cols([c for c in df.columns if c.startswith("cond")])
    if not x_cols or not cond_cols:
        raise ValueError("Expected at least one x* feature and one cond* regime column.")

    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    if TRADE_DATE_COL not in df.columns:
        df[TRADE_DATE_COL] = df[DATE_COL].dt.date
    else:
        df[TRADE_DATE_COL] = pd.to_datetime(df[TRADE_DATE_COL]).dt.date

    for col in ("cond2", "cond3"):
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    return df, x_cols, cond_cols


def _build_yearly_schedule(df: pd.DataFrame, spec: WindowSpec) -> list[dict]:
    years = np.array(sorted(df[DATE_COL].dt.to_period("Y").unique()))
    schedule = []
    for i in range(len(years) - 1):
        pred_year = years[i + 1]
        if spec.mode == "rolling":
            start_idx = i - spec.train_years + 1
            if start_idx < 0:
                continue
            train_years = years[start_idx:i + 1]
        elif spec.mode == "expanding":
            if i + 1 < spec.min_train_years:
                continue
            train_years = years[:i + 1]
        else:
            raise ValueError("WindowSpec.mode must be 'rolling' or 'expanding'.")
        schedule.append({"train_years": train_years, "pred_year": pred_year})
    return schedule


def _fit_scale_cond(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cond_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    c_train = train_df[cond_cols].to_numpy(dtype=float)
    c_test = test_df[cond_cols].to_numpy(dtype=float)
    lower = np.quantile(c_train, 0.01, axis=0)
    upper = np.quantile(c_train, 0.99, axis=0)
    c_train = np.clip(c_train, lower, upper)
    c_test = np.clip(c_test, lower, upper)
    mean = c_train.mean(axis=0)
    std = np.where(c_train.std(axis=0) < 1e-12, 1.0, c_train.std(axis=0))
    return (
        np.clip((c_train - mean) / std, -5.0, 5.0),
        np.clip((c_test - mean) / std, -5.0, 5.0),
    )


def _prepare_xy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_cols: list[str],
    cond_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.clip(train_df[x_cols].to_numpy(dtype=float), -3.0, 3.0)
    x_test = np.clip(test_df[x_cols].to_numpy(dtype=float), -3.0, 3.0)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    lo, hi = np.quantile(y_train, [0.01, 0.99])
    y_train = np.clip(y_train, lo, hi)
    c_train, c_test = _fit_scale_cond(train_df, test_df, cond_cols)
    return x_train, x_test, c_train, c_test, y_train


def _walk_forward_global_ridge(
    df: pd.DataFrame,
    x_cols: list[str],
    cond_cols: list[str],
    alpha: float = 0.0,
) -> pd.DataFrame:
    data = df.copy()
    data["year"] = data[DATE_COL].dt.to_period("Y")
    outputs = []
    for step in _build_yearly_schedule(data, WindowSpec()):
        train_df = data.loc[data["year"].isin(step["train_years"])].dropna(
            subset=x_cols + cond_cols + [TARGET_COL]
        )
        test_df = data.loc[data["year"] == step["pred_year"]].dropna(subset=x_cols + cond_cols)
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        x_train, x_test, _, _, y_train = _prepare_xy(train_df, test_df, x_cols, cond_cols)
        model = Ridge(alpha=alpha * len(train_df), fit_intercept=False)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        coef_norm = np.abs(model.coef_).sum()
        if coef_norm > 0:
            pred = pred / coef_norm
        out = test_df[[DATE_COL, TRADE_DATE_COL]].copy()
        out["pred_global"] = pred
        outputs.append(out)
    return pd.concat(outputs).sort_values(DATE_COL).reset_index(drop=True)


def _make_group_borders(values: np.ndarray, n_groups: int) -> np.ndarray:
    borders = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_groups + 1)))
    if len(borders) < 2:
        return np.array([-np.inf, np.inf])
    borders[0] = -np.inf
    borders[-1] = np.inf
    return borders


def _assign_groups(values: np.ndarray, borders: np.ndarray) -> np.ndarray:
    return np.searchsorted(borders[1:-1], values, side="right").astype(int)


def _fit_partial_pooling(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alpha_shared: float,
    alpha_group: float,
) -> tuple[np.ndarray, np.ndarray]:
    n, p = x.shape
    n_groups = int(groups.max()) + 1
    z_parts = [x]
    for group in range(n_groups):
        z_parts.append(x * (groups == group).astype(float).reshape(-1, 1))
    z = np.hstack(z_parts)
    a = (z.T @ z) / n
    b = (z.T @ y) / n
    penalty = np.zeros(p * (1 + n_groups))
    penalty[:p] = alpha_shared
    counts = np.bincount(groups, minlength=n_groups).astype(float)
    for group in range(n_groups):
        start = p * (1 + group)
        penalty[start:start + p] = alpha_group * (counts[group] / n)
    a.flat[::a.shape[0] + 1] += penalty + 1e-10
    theta = np.linalg.solve(a, b)
    return theta[:p], theta[p:].reshape(n_groups, p)


def _predict_partial_pooling(
    x: np.ndarray,
    groups: np.ndarray,
    beta_shared: np.ndarray,
    beta_group: np.ndarray,
) -> np.ndarray:
    row_beta = beta_shared + beta_group[groups]
    norm = np.abs(row_beta).sum(axis=1) + 1e-8
    return np.sum(x * row_beta, axis=1) / norm


def _walk_forward_partial_pooling(
    df: pd.DataFrame,
    x_cols: list[str],
    cond_col: str = "cond3",
    alpha_shared: float = 0.001,
    alpha_group: float = 0.1,
) -> pd.DataFrame:
    data = df.copy()
    data["year"] = data[DATE_COL].dt.to_period("Y")
    outputs = []
    for step in _build_yearly_schedule(data, WindowSpec()):
        train_df = data.loc[data["year"].isin(step["train_years"])].dropna(
            subset=x_cols + [cond_col] + [TARGET_COL]
        )
        test_df = data.loc[data["year"] == step["pred_year"]].dropna(subset=x_cols + [cond_col])
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        x_train = np.clip(train_df[x_cols].to_numpy(dtype=float), -3.0, 3.0)
        x_test = np.clip(test_df[x_cols].to_numpy(dtype=float), -3.0, 3.0)
        y_train = train_df[TARGET_COL].to_numpy(dtype=float)
        lo, hi = np.quantile(y_train, [0.01, 0.99])
        y_train = np.clip(y_train, lo, hi)
        borders = _make_group_borders(train_df[cond_col].to_numpy(dtype=float), n_groups=3)
        train_groups = _assign_groups(train_df[cond_col].to_numpy(dtype=float), borders)
        test_groups = _assign_groups(test_df[cond_col].to_numpy(dtype=float), borders)
        beta_shared, beta_group = _fit_partial_pooling(
            x_train, y_train, train_groups, alpha_shared, alpha_group
        )
        out = test_df[[DATE_COL, TRADE_DATE_COL]].copy()
        out["pred_pooled"] = _predict_partial_pooling(x_test, test_groups, beta_shared, beta_group)
        outputs.append(out)
    return pd.concat(outputs).sort_values(DATE_COL).reset_index(drop=True)


class LogisticGateRidgeMoE:
    def __init__(
        self,
        n_experts: int = 5,
        ridge_alpha: float = 0.1,
        gate_c: float = 0.1,
        n_em_iters: int = 3,
        random_state: int = 42,
    ):
        self.n_experts = n_experts
        self.ridge_alpha = ridge_alpha
        self.gate_c = gate_c
        self.n_em_iters = n_em_iters
        self.random_state = random_state
        self.experts: list[Ridge] = []
        self.gate: LogisticRegression | None = None

    def _expert_preds(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([model.predict(x) for model in self.experts])

    def fit(self, x: np.ndarray, c: np.ndarray, y: np.ndarray) -> "LogisticGateRidgeMoE":
        labels = KMeans(n_clusters=self.n_experts, n_init=20, random_state=self.random_state).fit_predict(c)
        self.experts = []
        for idx in range(self.n_experts):
            mask = labels == idx
            if mask.sum() < max(20, x.shape[1]):
                mask = np.ones(len(y), dtype=bool)
            model = Ridge(alpha=self.ridge_alpha * mask.sum(), fit_intercept=False)
            model.fit(x[mask], y[mask])
            self.experts.append(model)
        self.gate = LogisticRegression(
            C=self.gate_c,
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )
        self.gate.fit(c, labels)

        tau = max(float(np.var(y)), 1e-12)
        for _ in range(self.n_em_iters):
            expert_preds = self._expert_preds(x)
            gate_probs = self.gate.predict_proba(c)
            likelihood = np.exp(-((y[:, None] - expert_preds) ** 2) / tau)
            resp = gate_probs * likelihood
            resp = resp / np.clip(resp.sum(axis=1, keepdims=True), 1e-12, None)
            self.experts = []
            for idx in range(self.n_experts):
                n_eff = resp[:, idx].sum()
                model = Ridge(alpha=self.ridge_alpha * n_eff, fit_intercept=False)
                model.fit(x, y, sample_weight=resp[:, idx])
                self.experts.append(model)
            self.gate.fit(c, resp.argmax(axis=1))
        return self

    def predict(self, x: np.ndarray, c: np.ndarray) -> np.ndarray:
        if self.gate is None:
            raise RuntimeError("Model has not been fitted.")
        return (self._expert_preds(x) * self.gate.predict_proba(c)).sum(axis=1)


def _walk_forward_moe(df: pd.DataFrame, x_cols: list[str], cond_cols: list[str]) -> pd.DataFrame:
    data = df.copy()
    data["year"] = data[DATE_COL].dt.to_period("Y")
    outputs = []
    for step in _build_yearly_schedule(data, WindowSpec()):
        train_df = data.loc[data["year"].isin(step["train_years"])].dropna(
            subset=x_cols + cond_cols + [TARGET_COL]
        )
        test_df = data.loc[data["year"] == step["pred_year"]].dropna(subset=x_cols + cond_cols)
        if len(train_df) == 0 or len(test_df) == 0:
            continue
        x_train, x_test, c_train, c_test, y_train = _prepare_xy(train_df, test_df, x_cols, cond_cols)
        model = LogisticGateRidgeMoE().fit(x_train, c_train, y_train)
        train_pred = model.predict(x_train, c_train)
        pred = model.predict(x_test, c_test) / (train_pred.std() + 1e-12)
        out = test_df[[DATE_COL, TRADE_DATE_COL]].copy()
        out["pred_moe"] = pred
        outputs.append(out)
    return pd.concat(outputs).sort_values(DATE_COL).reset_index(drop=True)


def _merge_predictions(preds: list[pd.DataFrame]) -> pd.DataFrame:
    merged = preds[0]
    for pred in preds[1:]:
        merged = merged.merge(pred, on=[DATE_COL, TRADE_DATE_COL], how="inner")
    return merged


def _daily_curves(
    df: pd.DataFrame,
    signal_cols: list[str],
    start: str | None,
    end: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    dates = pd.to_datetime(data[TRADE_DATE_COL])
    if start is not None:
        data = data.loc[dates >= pd.Timestamp(start)]
        dates = pd.to_datetime(data[TRADE_DATE_COL])
    if end is not None:
        data = data.loc[dates <= pd.Timestamp(end)]
    curves = {}
    rows = []
    for col in signal_cols:
        bt = backtest(data, feature_col=col, ret_col=TARGET_COL, date_col=TRADE_DATE_COL)
        curves[col] = bt["cumulative_returns"]
        rows.append(
            {
                "signal": col,
                "sharpe": bt["sharpe_ratio"],
                "average_return": bt["average_return"],
                "start": start or "",
                "end": end or "",
            }
        )
    return pd.DataFrame(curves), pd.DataFrame(rows)


def _plot_two_panel(
    validation_curves: pd.DataFrame,
    test_curves: pd.DataFrame,
    labels: dict[str, str],
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, curves, subtitle in [
        (axes[0], validation_curves, "Validation / selection period"),
        (axes[1], test_curves, "Final test period"),
    ]:
        for col in curves.columns:
            ax.plot(curves.index, curves[col], label=labels.get(col, col), linewidth=1.8)
        ax.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.5)
        ax.set_title(subtitle)
        ax.set_xlabel("Trade date")
        ax.set_ylabel("Cumulative PnL")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="png", dpi=160)
    plt.close(fig)


def _write_placeholder(path: Path, title: str, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.8, 3.6), constrained_layout=True)
    ax.set_axis_off()
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.50, message, ha="center", va="center", fontsize=11)
    ax.text(
        0.5,
        0.38,
        "Run: python scripts/build_readme_assets.py --data data/model_data.parquet --out figures",
        ha="center",
        va="center",
        fontsize=10,
    )
    ax.text(
        0.02,
        0.08,
        "PnL curves require the local confidential parquet file.",
        ha="left",
        va="center",
        fontsize=9,
        color="#555555",
    )
    fig.savefig(path, format="png", dpi=160)
    plt.close(fig)


def _read_prediction_csv(path: Path, pred_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {DATE_COL, TRADE_DATE_COL, "pred"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    out = df[[DATE_COL, TRADE_DATE_COL, "pred"]].copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out[TRADE_DATE_COL] = pd.to_datetime(out[TRADE_DATE_COL]).dt.date
    return out.rename(columns={"pred": pred_col})


def _average_seed_predictions(frames: list[pd.DataFrame], pred_col: str) -> pd.DataFrame:
    renamed = []
    for idx, frame in enumerate(frames):
        col = f"{pred_col}_seed_{idx}"
        tmp = frame[[DATE_COL, TRADE_DATE_COL, "pred"]].rename(columns={"pred": col})
        renamed.append(tmp)

    merged = renamed[0]
    for frame in renamed[1:]:
        merged = merged.merge(frame, on=[DATE_COL, TRADE_DATE_COL], how="inner")

    seed_cols = [c for c in merged.columns if c.startswith(f"{pred_col}_seed_")]
    merged[pred_col] = merged[seed_cols].mean(axis=1)
    return merged[[DATE_COL, TRADE_DATE_COL, pred_col]]


def _train_transformer_ensemble(
    df: pd.DataFrame,
    x_cols: list[str],
    cond_cols: list[str],
    train_end_year: int | None,
    train_end_date: str | None,
    val_start: str,
    val_end: str | None,
) -> pd.DataFrame:
    from src.models.transformer import DirectedRegimeAttentionTransformer, train_predict_fixed_split

    frames = []
    for seed in (7, 99, 2026):
        pred = train_predict_fixed_split(
            df,
            x_cols,
            cond_cols,
            DirectedRegimeAttentionTransformer,
            {
                "d_model": 32,
                "n_heads": 4,
                "n_layers": 2,
                "ffn_dim": 128,
                "dropout": 0.3,
            },
            train_end_year=train_end_year or 2021,
            train_end_date=train_end_date,
            val_start=val_start,
            val_end=val_end,
            lr=1e-4,
            weight_decay=0.1,
            epochs=100,
            batch_size=1024,
            patience=6,
            lr_scheduler_patience=3,
            warmup_fraction=0.01,
            grad_clip=1.0,
            inner_val_mode="tail_fraction",
            inner_val_fraction=0.15,
            random_state=seed,
        )
        frames.append(pred)

    return _average_seed_predictions(frames, "pred_transformer")


def build_assets(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    data_path = Path(args.data)
    if not data_path.exists():
        _write_placeholder(
            out_dir / "readme_initial_models_pnl.png",
            "Initial regime model PnL",
            "Data file not found, so this placeholder was written.",
        )
        _write_placeholder(
            out_dir / "readme_later_models_pnl.png",
            "Later nonlinear model PnL",
            "Data file not found, so this placeholder was written.",
        )
        raise FileNotFoundError(
            f"{data_path} does not exist. Place the parquet file there and rerun this script."
        )

    df, x_cols, cond_cols = _load_data(data_path)
    initial_preds = _merge_predictions(
        [
            _walk_forward_global_ridge(df, x_cols, cond_cols),
            _walk_forward_moe(df, x_cols, cond_cols),
            _walk_forward_partial_pooling(df, x_cols),
        ]
    ).merge(df[[DATE_COL, TARGET_COL]], on=DATE_COL, how="left")

    val_curves, val_metrics = _daily_curves(
        initial_preds, ["pred_global", "pred_moe", "pred_pooled"], None, "2022-12-31"
    )
    test_curves, test_metrics = _daily_curves(
        initial_preds, ["pred_global", "pred_moe", "pred_pooled"], "2023-01-01", "2024-12-31"
    )
    _plot_two_panel(
        val_curves,
        test_curves,
        {
            "pred_global": "Global linear ridge",
            "pred_moe": "Mixture of experts",
            "pred_pooled": "Partial-pooling ridge",
        },
        "Initial regime model PnL",
        out_dir / "readme_initial_models_pnl.png",
    )
    pd.concat([val_metrics.assign(period="validation"), test_metrics.assign(period="test")]).to_csv(
        out_dir / "readme_initial_models_metrics.csv",
        index=False,
    )

    xgb_params = {
        "max_depth": 6,
        "n_estimators": 500,
        "learning_rate": 0.1,
        "subsample": 0.6,
        "colsample_bytree": 0.8,
        "min_child_weight": 100,
        "reg_alpha": 0.1,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "verbosity": 0,
    }
    xgb_val = train_predict_fixed_split_xgboost(
        df,
        x_cols,
        cond_cols,
        train_end_year=2021,
        val_start="2022-01-01",
        val_end="2023-06-30",
        xgb_params=xgb_params,
    ).rename(columns={"pred": "pred_xgb"})
    xgb_test = train_predict_fixed_split_xgboost(
        df,
        x_cols,
        cond_cols,
        train_end_date="2023-06-30",
        val_start="2023-07-01",
        xgb_params=xgb_params,
    ).rename(columns={"pred": "pred_xgb"})
    later = pd.concat([xgb_val, xgb_test], ignore_index=True, sort=False)

    if args.transformer_validation_preds and args.transformer_test_preds:
        transformer_val = _read_prediction_csv(Path(args.transformer_validation_preds), "pred_transformer")
        transformer_test = _read_prediction_csv(Path(args.transformer_test_preds), "pred_transformer")
        later = later.merge(
            pd.concat([transformer_val, transformer_test], ignore_index=True, sort=False),
            on=[DATE_COL, TRADE_DATE_COL],
            how="outer",
        )
    else:
        transformer_val = _train_transformer_ensemble(
            df,
            x_cols,
            cond_cols,
            train_end_year=2021,
            train_end_date=None,
            val_start="2022-01-01",
            val_end="2023-06-30",
        )
        transformer_test = _train_transformer_ensemble(
            df,
            x_cols,
            cond_cols,
            train_end_year=None,
            train_end_date="2023-06-30",
            val_start="2023-07-01",
            val_end=None,
        )
        later = later.merge(
            pd.concat([transformer_val, transformer_test], ignore_index=True, sort=False),
            on=[DATE_COL, TRADE_DATE_COL],
            how="outer",
        )

    later = later.drop(columns=[TARGET_COL], errors="ignore")
    later = later.merge(df[[DATE_COL, TARGET_COL]], on=DATE_COL, how="left")
    signal_cols = [col for col in ["pred_xgb", "pred_transformer"] if col in later.columns]
    later_val_curves, later_val_metrics = _daily_curves(later, signal_cols, "2022-01-01", "2023-06-30")
    later_test_curves, later_test_metrics = _daily_curves(later, signal_cols, "2023-07-01", None)
    _plot_two_panel(
        later_val_curves,
        later_test_curves,
        {"pred_xgb": "Interaction-constrained XGBoost", "pred_transformer": "Directed-regime transformer"},
        "Later nonlinear model PnL",
        out_dir / "readme_later_models_pnl.png",
    )
    pd.concat(
        [later_val_metrics.assign(period="validation"), later_test_metrics.assign(period="test")]
    ).to_csv(out_dir / "readme_later_models_metrics.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build README PnL comparison assets.")
    parser.add_argument("--data", default="data/model_data.parquet", help="Path to the confidential parquet file.")
    parser.add_argument("--out", default="figures", help="Directory for generated README assets.")
    parser.add_argument("--transformer-validation-preds", help="Optional CSV with msgStamp, trade_date, pred.")
    parser.add_argument("--transformer-test-preds", help="Optional CSV with msgStamp, trade_date, pred.")
    return parser.parse_args()


if __name__ == "__main__":
    build_assets(parse_args())
