"""
Fatigue prediction pipeline for simulated EMG/IMU data.

This module implements the multi-stage processing and baseline models described
by the user:
    1. RMS 去力化
    2. 累積疲勞分數與燈號決策
    3. EWMA 與 5 分鐘趨勢
    4. 產生 30 分鐘未來疲勞標籤與 baseline 特徵
    5. 訓練 Logistic 與 HistGradientBoosting 模型
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ========= 可調參數 =========
SAMPLE_EVERY_S = 30.0
ALPHA_EWMA = 0.30
THETA_PER_MIN = 0.02
THETA_PER_SEC = THETA_PER_MIN / 60.0
LAMBDA_REST = 0.0003
POSE_GAMMA = 0.10
POSE_CAP = 0.50
WINDOW_SLOPE_S = 300.0
ELEVEL_BINS = [-1, 0.33, 0.66, 10]
ELEVEL_LABELS = ["low", "mid", "high"]
HORIZON_S = 1800.0


# ========= A. RMS 去力化 =========
def ensure_emg_rms(df: pd.DataFrame) -> pd.DataFrame:
    """Synthesise EMG_RMS if the raw dataset does not provide it."""
    if "EMG_RMS" in df.columns:
        return df
    out = df.copy()
    mvc = out["mvc_percent"].astype(float)
    rula = out["rula_score"].astype(float)
    out["EMG_RMS"] = 0.02 + 0.005 * mvc + 0.0005 * rula
    return out


def fit_force_rms_model(df_fresh: pd.DataFrame) -> Tuple[float, float]:
    X = df_fresh[["mvc_percent"]].values
    X = np.c_[np.ones(len(X)), X]
    y = df_fresh["EMG_RMS"].values
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


def compute_rms_fe(df: pd.DataFrame, alpha: float, beta: float) -> pd.Series:
    return df["EMG_RMS"] - (alpha + beta * df["mvc_percent"])


# ========= B. 累積疲勞分數 =========
def deltaE_double_threshold(mvc: np.ndarray) -> np.ndarray:
    over20 = np.maximum(0.0, mvc - 20.0)
    over40 = np.maximum(0.0, mvc - 40.0)
    return over20 + 2.0 * over40


def pose_multiplier(rula: np.ndarray) -> np.ndarray:
    add = POSE_GAMMA * np.maximum(0.0, rula - 3.0)
    add = np.minimum(POSE_CAP, add)
    return 1.0 + add


def accumulate_with_recovery(deltaE: np.ndarray, mvc: np.ndarray) -> np.ndarray:
    dt = SAMPLE_EVERY_S
    E = np.zeros_like(deltaE, dtype=float)
    for i in range(len(deltaE)):
        lam_eff = LAMBDA_REST * max(0.0, 1.0 - min(1.0, mvc[i] / 20.0))
        decay = 1.0 - lam_eff * dt
        prev = 0.0 if i == 0 else E[i - 1]
        E[i] = max(0.0, prev * decay + deltaE[i] * dt)
    return E


def normalize_E(E: np.ndarray, session_seconds: float = 2 * 3600.0) -> np.ndarray:
    Emax_per_sec = ((80 - 20) * 1 + (80 - 40) * 2)
    return np.clip(E / (Emax_per_sec * session_seconds), 0.0, 1.0)


# ========= C. 平滑與斜率 =========
def ewma(x: np.ndarray, alpha: float = ALPHA_EWMA) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    for i, v in enumerate(x):
        out[i] = v if i == 0 else alpha * v + (1 - alpha) * out[i - 1]
    return out


def slope_5min(series: np.ndarray) -> np.ndarray:
    dt = SAMPLE_EVERY_S
    w = int(WINDOW_SLOPE_S / dt)
    if w < 2:
        raise ValueError("WINDOW_SLOPE_S 太小")
    x = np.arange(w) * dt
    x = x - x.mean()
    denom = np.sum(x**2)
    slope = np.full_like(series, np.nan, dtype=float)
    for i in range(w - 1, len(series)):
        y = series[i - w + 1 : i + 1]
        y = y - y.mean()
        slope[i] = float((x @ y) / denom)
    return slope


def trend_from_slope(s_per_sec: float) -> str:
    if np.isnan(s_per_sec):
        return "NA"
    if s_per_sec > THETA_PER_SEC:
        return "up"
    if s_per_sec < -THETA_PER_SEC:
        return "down"
    return "flat"


# ========= D. 燈號決策 =========
def level_from_Enorm(e: float) -> str:
    bins = ELEVEL_BINS
    if e < bins[1]:
        return "low"
    if e < bins[2]:
        return "mid"
    return "high"


def led_logic(level: str, trend: str) -> Dict[str, str]:
    if level == "high" and trend == "up":
        return {"color": "red", "blink": "fast"}
    if level == "high" and trend in ("flat", "down"):
        return {"color": "red", "blink": "slow"}
    if level == "mid" and trend == "up":
        return {"color": "amber", "blink": "fast"}
    return {"color": "green", "blink": "none" if trend != "up" else "slow"}


# ========= E. 標籤與特徵 =========
def future_high_label(E_norm: np.ndarray) -> np.ndarray:
    horizon_pts = int(HORIZON_S / SAMPLE_EVERY_S)
    y = np.zeros_like(E_norm, dtype=int)
    for i in range(len(E_norm)):
        j = min(len(E_norm), i + horizon_pts)
        y[i] = 1 if np.any(E_norm[i:j] >= 0.66) else 0
    return y


def future_level_label(E_norm: np.ndarray) -> np.ndarray:
    horizon_pts = int(HORIZON_S / SAMPLE_EVERY_S)
    y = np.empty(len(E_norm), dtype=object)
    for i in range(len(E_norm)):
        j = min(len(E_norm), i + horizon_pts)
        m = np.nanmax(E_norm[i:j]) if j > i else E_norm[i]
        y[i] = level_from_Enorm(m)
    return y


def rolling_feature(arr: np.ndarray, w_pts: int, fn: str) -> np.ndarray:
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(w_pts - 1, len(arr)):
        win = arr[i - w_pts + 1 : i + 1]
        if fn == "mean":
            out[i] = np.nanmean(win)
        elif fn == "max":
            out[i] = np.nanmax(win)
        elif fn == "slope":
            dt = SAMPLE_EVERY_S
            x = np.arange(w_pts) * dt
            x = x - x.mean()
            denom = np.sum(x**2)
            win2 = win - np.nanmean(win)
            out[i] = float((x @ win2) / denom)
        else:
            raise ValueError("unknown fn")
    return out


def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = SAMPLE_EVERY_S
    w3m = int(180.0 / dt)
    w5m = int(300.0 / dt)

    feat = pd.DataFrame(index=df.index)

    feat["mvc_mean_3m"] = rolling_feature(df["mvc_percent"].values, w3m, "mean")
    feat["mvc_max_5m"] = rolling_feature(df["mvc_percent"].values, w5m, "max")
    feat["mvc_slope_5m"] = rolling_feature(df["mvc_percent"].values, w5m, "slope")

    feat["E_norm"] = df["E_norm"].values
    feat["E_ewma"] = df["E_smooth"].values
    feat["E_slope5m"] = df["slope5m"].values

    r = df["rula_score"].values.astype(float)
    feat["rula_avg_5m"] = rolling_feature(r, w5m, "mean")
    feat["rula_max_5m"] = rolling_feature(r, w5m, "max")

    frac = np.full(len(r), np.nan)
    for i in range(w5m - 1, len(r)):
        win = r[i - w5m + 1 : i + 1]
        frac[i] = np.mean(win >= 5)
    feat["rula_frac_ge5_5m"] = frac

    feat["rms_fe_last"] = df["RMS_fe"].values

    var5 = np.full(len(r), np.nan)
    rms_vals = df["RMS_fe"].values
    for i in range(w5m - 1, len(r)):
        win = rms_vals[i - w5m + 1 : i + 1]
        var5[i] = np.nanvar(win)
    feat["rms_fe_var_5m"] = var5

    tod = pd.to_datetime(df["timestamp"]).dt.hour.values
    feat["tod_morning"] = ((tod >= 8) & (tod < 12)).astype(int)
    feat["tod_afternoon"] = ((tod >= 12) & (tod < 18)).astype(int)
    feat["tod_evening"] = (((tod >= 18) | (tod < 8))).astype(int)

    return feat


# ========= 主流程 =========
def _run_pipeline_single(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_emg_rms(df)
    df = df.sort_values("timestamp").reset_index(drop=True).copy()

    t0 = pd.to_datetime(df["timestamp"]).min()
    fresh_mask = pd.to_datetime(df["timestamp"]) <= t0 + pd.Timedelta(minutes=20)
    fresh = df.loc[fresh_mask]
    alpha, beta = fit_force_rms_model(fresh)

    df["RMS_fe"] = compute_rms_fe(df, alpha, beta)

    base = deltaE_double_threshold(df["mvc_percent"].values)
    mult = pose_multiplier(df["rula_score"].values.astype(float))
    deltaE = base * mult

    E = accumulate_with_recovery(deltaE, df["mvc_percent"].values)
    df["E"] = E
    session_seconds = df.shape[0] * SAMPLE_EVERY_S
    df["E_norm"] = normalize_E(E, session_seconds=session_seconds)
    df["level"] = [level_from_Enorm(v) for v in df["E_norm"].values]

    df["E_smooth"] = ewma(df["E_norm"].values, ALPHA_EWMA)
    df["slope5m"] = slope_5min(df["E_smooth"].values)
    df["trend"] = [trend_from_slope(s) for s in df["slope5m"].values]

    out = [led_logic(lv, tr) for lv, tr in zip(df["level"], df["trend"])]
    df["color"] = [o["color"] for o in out]
    df["blink"] = [o["blink"] for o in out]

    return df


def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("資料需包含 timestamp 欄位")

    if "session_id" in df.columns:
        processed = []
        for _, grp in df.groupby("session_id", sort=False):
            processed.append(_run_pipeline_single(grp))
        df_out = pd.concat(processed, ignore_index=True)
    else:
        df_out = _run_pipeline_single(df)
    return df_out


def build_and_train_baselines(df_proc: pd.DataFrame, return_reports: bool = False):
    y_bin = future_high_label(df_proc["E_norm"].values)
    y_tri = future_level_label(df_proc["E_norm"].values)

    X = build_baseline_features(df_proc)

    valid = ~X.isna().any(axis=1)
    X = X.loc[valid]
    yb = y_bin[valid.values]
    yt = y_tri[valid.values]

    models = {}
    reports: Dict[str, Dict[str, Dict[str, float]]] = {}

    if len(np.unique(yb)) > 1:
        X_train, X_test, yb_tr, yb_te = train_test_split(
            X, yb, test_size=0.3, random_state=42, stratify=yb
        )
        clf_log = LogisticRegression(max_iter=200)
        clf_log.fit(X_train, yb_tr)
        y_pred_log = clf_log.predict(X_test)
        print("\n[Binary: future 30m HIGH?] Logistic")
        print(classification_report(yb_te, y_pred_log, zero_division=0))
        if return_reports:
            reports["binary_logistic"] = classification_report(
                yb_te, y_pred_log, zero_division=0, output_dict=True
            )

        clf_hgb = HistGradientBoostingClassifier(random_state=42)
        clf_hgb.fit(X_train, yb_tr)
        y_pred_hgb = clf_hgb.predict(X_test)
        print("[Binary: future 30m HIGH?] HGB")
        print(classification_report(yb_te, y_pred_hgb, zero_division=0))
        if return_reports:
            reports["binary_hgb"] = classification_report(
                yb_te, y_pred_hgb, zero_division=0, output_dict=True
            )

        models["bin_log"] = clf_log
        models["bin_hgb"] = clf_hgb
    else:
        print("Binary label lacks class diversity; skipping binary models.")

    if len(np.unique(yt)) > 1:
        X_train, X_test, yt_tr, yt_te = train_test_split(
            X, yt, test_size=0.3, random_state=42, stratify=yt
        )
        clf_hgb3 = HistGradientBoostingClassifier(random_state=42)
        clf_hgb3.fit(X_train, yt_tr)
        y_pred_hgb3 = clf_hgb3.predict(X_test)
        print("\n[Trinary: future level (low/mid/high)] HGB")
        print(classification_report(yt_te, y_pred_hgb3, zero_division=0))
        if return_reports:
            reports["trinary_hgb"] = classification_report(
                yt_te, y_pred_hgb3, zero_division=0, output_dict=True
            )
        models["tri_hgb"] = clf_hgb3
    else:
        print("Trinary label lacks class diversity; skipping multi-class model.")

    if return_reports:
        return models, reports
    return models


def augment_with_high_fatigue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append simulated sessions containing sustained moderate/high負荷，使得
    未來 30 分鐘標籤具備多個等級（含 high）。
    """
    dt = SAMPLE_EVERY_S
    columns = [
        "session_id",
        "timestamp",
        "t_sec",
        "intensity",
        "mvc_percent",
        "rula_score",
        "deltaE_work",
        "E_fixed",
        "E_recover",
        "E_norm_fixed",
        "E_norm_recover",
        "fatigue_fixed",
        "fatigue_recover",
        "slope_5min_per_s",
        "is_rest_like",
        "subject",
    ]

    if not set(columns).issubset(df.columns):
        missing = set(columns) - set(df.columns)
        raise ValueError(f"資料集缺少欄位，無法模擬：{missing}")

    t_max = pd.to_datetime(df["timestamp"]).max()
    current_start = t_max + pd.Timedelta(seconds=dt)

    scenarios = [
        (
            "sim_midfatigue",
            "simulated_mid",
            [
                (15, 12.0, 2.0, "warmup"),
                (25, 40.0, 3.0, "moderate"),
                (45, 55.0, 4.0, "mid"),
                (20, 20.0, 2.5, "recover"),
                (30, 50.0, 3.5, "mid_repeat"),
            ],
        ),
        (
            "sim_highfatigue",
            "simulated_high",
            [
                (10, 15.0, 2.0, "prep"),
                (20, 45.0, 3.5, "build"),
                (120, 85.0, 5.0, "high"),
                (15, 25.0, 2.5, "short_rest"),
                (50, 90.0, 5.5, "extreme"),
                (20, 18.0, 2.0, "cooldown"),
            ],
        ),
    ]

    all_rows = []

    for session_id, subject, segments in scenarios:
        t_sec = 0.0
        rows = []
        for minutes, mvc, rula, intensity in segments:
            steps = int(round((minutes * 60.0) / dt))
            for _ in range(steps):
                ts = current_start + pd.Timedelta(seconds=t_sec)
                rows.append(
                    {
                        "session_id": session_id,
                        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "t_sec": t_sec,
                        "intensity": intensity,
                        "mvc_percent": mvc,
                        "rula_score": rula,
                        "deltaE_work": np.nan,
                        "E_fixed": np.nan,
                        "E_recover": np.nan,
                        "E_norm_fixed": np.nan,
                        "E_norm_recover": np.nan,
                        "fatigue_fixed": np.nan,
                        "fatigue_recover": np.nan,
                        "slope_5min_per_s": np.nan,
                        "is_rest_like": 0 if mvc >= 20 else 1,
                        "subject": subject,
                    }
                )
                t_sec += dt

        all_rows.extend(rows)
        total_seconds = t_sec if rows else 0.0
        current_start = current_start + pd.Timedelta(seconds=total_seconds + dt)

    df_sim = pd.DataFrame(all_rows, columns=columns)
    return pd.concat([df, df_sim], ignore_index=True)


def main():
    input_csv = "fatigue_simulated_with_recovery.csv"
    df_raw = pd.read_csv(input_csv)

    required = {"timestamp", "mvc_percent", "rula_score"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"缺少必備欄位：{missing}")

    df_proc = run_pipeline(df_raw)
    df_proc.to_csv("fatigue_processed.csv", index=False)
    print("Saved fatigue_processed.csv")

    result = build_and_train_baselines(df_proc, return_reports=True)
    if isinstance(result, tuple):
        models, reports = result
    else:
        models, reports = result, {}

    if models:
        summary = pd.DataFrame(
            [(name, type(model).__name__) for name, model in models.items()],
            columns=["name", "sklearn_class"],
        )
        summary.to_csv("trained_models_summary.csv", index=False)
        print("Saved trained_models_summary.csv")

    if reports:
        pd.DataFrame(reports).to_json("trained_model_reports.json", orient="index")
        print("Saved trained_model_reports.json")


if __name__ == "__main__":
    main()
