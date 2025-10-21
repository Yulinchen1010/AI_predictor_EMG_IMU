# main.py
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==================== 基本設定 ====================
APP_TITLE = "疲勞預測系統"
DB_PATH = os.environ.get("DB_PATH", "fatigue_data.db")

# 建置版本：Render/Heroku 常見環境變數或退回 ISO 時間
APP_BUILD = (
    os.environ.get("RENDER_GIT_COMMIT")
    or os.environ.get("SOURCE_VERSION")
    or os.environ.get("APP_BUILD")
    or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
)

# 模型檔案
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_classifier.pkl")
# 🔮 LSTM 與縮放器（若存在就載入，不存在就用後援法）
LSTM_MODEL_PATH = os.path.join(MODEL_DIR, "lstm_predictor.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# 預測相關常數
TIME_INTERVAL_MINUTES = int(os.environ.get("TIME_INTERVAL_MINUTES", "1"))  # 取樣粒度（分）
HISTORY_SEQUENCE_LENGTH = int(os.environ.get("HISTORY_SEQUENCE_LENGTH", "60"))  # LSTM 歷史長度
PREDICTION_HORIZON = int(os.environ.get("PREDICTION_HORIZON", "30"))  # 未來幾步（分）

# 使用者（如果 App 沒傳 worker_id，就用這個）
DEFAULT_WORKER_ID = os.environ.get("DEFAULT_WORKER_ID", "user001")

# 風險層級 → 顏色與標籤（改成前端慣用色票）
RISK_LABELS = ["低度", "中度", "高度"]
RISK_COLORS = ["#22C55E", "#F59E0B", "#EF4444"]  # 綠、橘、紅

# 感測器輸出最大值提示（若無設定，稍後會依資料自動推斷）
try:
    _sensor_max_env = os.environ.get("MVC_SENSOR_MAX")
    MVC_SENSOR_MAX_HINT: Optional[float] = None
    if _sensor_max_env is not None:
        candidate = float(_sensor_max_env)
        if candidate > 0:
            MVC_SENSOR_MAX_HINT = candidate
except Exception:
    MVC_SENSOR_MAX_HINT = None

# 台灣時區
TZ_TAIWAN = timezone(timedelta(hours=8))

# 常數：型別名稱與篩選門檻
HEARTBEAT_TYPE = "heartbeat"
MIN_EFFECTIVE_PCT = 0.1  # 低於此值視為近似 0，不納入特徵
# 若收到異常小的 UNIX 秒（例如 56.545 秒被誤當 epoch），以 2000-01-01 作為「合理界線」
UNREASONABLE_EPOCH_SEC = datetime(2000, 1, 1, tzinfo=timezone.utc).timestamp()


def now_taiwan_iso() -> str:
    return datetime.now(TZ_TAIWAN).isoformat()


# ==================== FastAPI ====================
app = FastAPI(title=APP_TITLE, version="v1.3")

# CORS 設定：預設全開（支援以逗號分隔的允許來源）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== DB ====================
logger = logging.getLogger("uvicorn.error")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=5.0, check_same_thread=False)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass
    return conn


def init_db() -> None:
    conn = get_conn()
    c = conn.cursor()
    # 擴充欄位但相容：必填 worker_id / timestamp / percent_mvc
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            percent_mvc REAL NOT NULL,
            ts REAL,               -- UNIX seconds，可空
            rms REAL,              -- 預留 EMG RMS，非必填
            type TEXT              -- 來源/類型（例如 'heartbeat' / 'mvc' / 'emg' / 'imu'）
        )
        """
    )
    # 以防舊表缺欄位（忽略錯誤即可）
    for ddl in (
        "ALTER TABLE sensor_data ADD COLUMN rms REAL",
        "ALTER TABLE sensor_data ADD COLUMN ts REAL",
        "ALTER TABLE sensor_data ADD COLUMN type TEXT",
    ):
        try:
            c.execute(ddl)
        except Exception:
            pass

    c.execute("CREATE INDEX IF NOT EXISTS idx_worker_ts ON sensor_data(worker_id, timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_type_ts ON sensor_data(type, timestamp)")
    conn.commit()
    conn.close()


init_db()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    logger.info(
        f"[REQ] {request.method} {request.url.path} q={dict(request.query_params)} "
        f"len={len(body)} ct={request.headers.get('content-type')}"
    )
    try:
        response = await call_next(request)
        return response
    finally:
        try:
            status_code = getattr(response, "status_code", "n/a")
        except UnboundLocalError:
            status_code = "n/a"
        logger.info(f"[RESP] {request.method} {request.url.path} -> {status_code}")


@app.exception_handler(Exception)
async def unhandled_ex_handler(request: Request, exc: Exception):
    import traceback

    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(f"[UNHANDLED] {request.url.path} {exc}\n{tb}")
    if isinstance(exc, HTTPException):
        detail = exc.detail
        if not isinstance(detail, dict):
            detail = {"detail": detail}
        return JSONResponse(
            status_code=exc.status_code,
            content=detail,
            headers=getattr(exc, "headers", None),
        )
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": str(exc)})


# ==================== 請求模型 ====================
class SensorData(BaseModel):
    percent_mvc: float = Field(ge=0, le=120)  # 正規化後允許些微超 100
    timestamp: Optional[str] = None  # ISO-8601


# ==================== 工具：正規化上傳列 ====================
def _to_ts_sec(v: Any) -> float:
    if v is None:
        return datetime.utcnow().timestamp()
    if isinstance(v, (int, float)):
        return float(v)
    # 可能是 ISO 字串
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return datetime.utcnow().timestamp()


def _safe_iso_from_ts(ts_sec: Optional[float]) -> str:
    """
    將 ts（UNIX 秒或 None）轉為 ISO-8601 UTC。
    若 ts 秒數明顯不合理（如 56.545），改用現在時間防呆。
    """
    if ts_sec is None:
        ts_sec = datetime.utcnow().timestamp()
    try:
        tsf = float(ts_sec)
    except Exception:
        tsf = datetime.utcnow().timestamp()

    if tsf < UNREASONABLE_EPOCH_SEC:  # 早於 2000-01-01 視為不合理
        tsf = datetime.utcnow().timestamp()

    return datetime.utcfromtimestamp(tsf).replace(tzinfo=timezone.utc).isoformat()


def _clean_numeric(v: Any) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _infer_mvc_scale(values: List[Optional[float]]) -> float:
    """根據資料決定原始 MVC 的最大刻度（推斷 0..1 / 0..20 / 0..100 這種）。"""
    if MVC_SENSOR_MAX_HINT:
        return MVC_SENSOR_MAX_HINT

    cleaned = [float(x) for x in values if x is not None]
    if not cleaned:
        return 100.0

    max_val = max(cleaned)
    if max_val <= 1.0:
        return 1.0
    if max_val <= 20.0:
        return 20.0
    if max_val <= 120.0:
        return 100.0
    return max(100.0, max_val)


def _mvc_to_percent(raw_value: Optional[float], scale: float) -> float:
    if raw_value is None:
        return 0.0
    try:
        value = float(raw_value)
    except Exception:
        return 0.0

    scale = float(scale) if scale and scale > 0 else 100.0
    if scale == 1.0:
        percent = value * 100.0
    else:
        percent = (value / scale) * 100.0
    if percent < 0:
        percent = 0.0
    return float(min(percent, 120.0))  # 允許些微超 100，避免過度截斷


def _to_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _ensure_json_array(rows: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if rows is None:
        return []
    if isinstance(rows, list):
        return [dict(r) for r in rows]
    if isinstance(rows, dict):
        return [dict(rows)]
    raise HTTPException(400, detail="rows 必須是物件或物件陣列")


def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    容錯鍵名並轉成 DB 欄位：
    - worker_id：預設 DEFAULT_WORKER_ID
    - percent_mvc：容忍 MVC(0..1/0..100)、percent_mvc、mvc、emg_pct（自動推斷刻度後換算為 0..100）
    - timestamp：ISO；若沒給，用 ts（秒）或現在；並防呆修正「不合理的 epoch」
    - ts：UNIX 秒
    - rms：容忍 RMS/emg_rms/rms/emg
    - type：原樣保留（heartbeat / mvc / emg / imu）
    """
    out: List[Dict[str, Any]] = []
    raw_mvc_values: List[Optional[float]] = []

    # 先掃一次：蒐集原始 mvc 值以推斷刻度
    for m in rows:
        mm = dict(m)
        raw_mvc = None
        for k in ("percent_mvc", "MVC", "mvc", "emg_pct"):
            if k in mm and raw_mvc is None:
                raw_mvc = _clean_numeric(mm.get(k))
        raw_mvc_values.append(raw_mvc)

    scale = _infer_mvc_scale(raw_mvc_values)

    # 再建最終列
    for m, raw_mvc in zip(rows, raw_mvc_values):
        mm = dict(m)
        worker_id = str((mm.get("worker_id") or DEFAULT_WORKER_ID)).strip()

        # 時間
        if mm.get("timestamp"):
            ts_sec = _to_ts_sec(mm["timestamp"])
        else:
            ts_sec = _to_ts_sec(mm.get("ts"))
        iso = _safe_iso_from_ts(ts_sec)

        # RMS（可選）
        rms = None
        for k in ("RMS", "emg_rms", "rms", "emg"):
            if k in mm and rms is None:
                rms = _to_float(mm.get(k))

        typ = str(mm.get("type")) if mm.get("type") is not None else None

        out.append(
            {
                "worker_id": worker_id,
                "timestamp": iso,
                "percent_mvc": _mvc_to_percent(raw_mvc, scale) if raw_mvc is not None else 0.0,
                "ts": float(ts_sec) if ts_sec is not None else None,
                "rms": rms,
                "type": typ,
            }
        )

    return out


# ==================== 模型（載入/訓練） ====================
def _train_rf_classifier() -> RandomForestClassifier:
    # 合成一份可區分 Δ%MVC 與變化速率的資料，訓練快速
    n = 3000
    rng = np.random.RandomState(42)
    current_mvc = rng.uniform(20, 95, size=n)
    total_change = rng.uniform(-10, 60, size=n)
    change_rate = rng.uniform(-1.0, 3.0, size=n)
    avg_mvc = rng.uniform(25, 85, size=n)
    std_mvc = rng.uniform(0.5, 15, size=n)

    X = np.vstack([current_mvc, total_change, change_rate, avg_mvc, std_mvc]).T
    y = np.zeros(n, dtype=int)
    y[(total_change >= 20)] = 1
    y[(total_change >= 40)] = 2
    y[(change_rate > 2.0) & (y < 2)] = 2
    y[(change_rate > 1.2) & (y < 1)] = 1

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=150, max_depth=14, min_samples_split=8, min_samples_leaf=4, random_state=42
    )
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"✅ RandomForest 訓練完成（val acc = {score:.3f}）")
    joblib.dump(model, RF_MODEL_PATH)
    return model


def _load_or_train_rf() -> RandomForestClassifier:
    if os.path.exists(RF_MODEL_PATH):
        try:
            print("✅ 載入 RandomForest")
            return joblib.load(RF_MODEL_PATH)
        except Exception:
            pass
    print("⚙️ 未找到模型，開始訓練 RandomForest ...")
    return _train_rf_classifier()


RF_MODEL: RandomForestClassifier = _load_or_train_rf()

# ====== 嘗試載入 LSTM（可選）======
_TF_AVAILABLE = False
_LSTM_MODEL = None
_SCALER = None
_LSTM_SOURCE = "naive"

try:
    # 延遲匯入，避免環境沒裝 TF 就崩潰
    import tensorflow as tf  # type: ignore
    from tensorflow.keras.models import load_model  # type: ignore

    if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(SCALER_PATH):
        _LSTM_MODEL = load_model(LSTM_MODEL_PATH)
        _SCALER = joblib.load(SCALER_PATH)
        _TF_AVAILABLE = True
        _LSTM_SOURCE = "lstm"
        print("✅ LSTM 與 SCALER 載入成功")
    else:
        print("ℹ️ 找不到 LSTM/SCALER，將使用後援預測（naive linear）")
except Exception as _e:
    print(f"ℹ️ TensorFlow 無法載入或模型缺失（{_e}），使用後援預測（naive linear）")


# ==================== 特徵計算 & 風險評分 ====================
def _get_df(worker_id: str, limit: int = 600, exclude_heartbeat: bool = True) -> pd.DataFrame:
    """
    拉資料（預設排除 heartbeat），取最新 limit 筆，再轉回時間正序。
    """
    conn = get_conn()
    if exclude_heartbeat:
        sql = """
        SELECT * FROM sensor_data
        WHERE worker_id = ? AND (type IS NULL OR type != ?)
        ORDER BY timestamp DESC LIMIT ?
        """
        params = (worker_id, HEARTBEAT_TYPE, int(limit))
    else:
        sql = """
        SELECT * FROM sensor_data
        WHERE worker_id = ?
        ORDER BY timestamp DESC LIMIT ?
        """
        params = (worker_id, int(limit))

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _extract_features(df: pd.DataFrame) -> Optional[np.ndarray]:
    if len(df) < 2:
        return None

    # 整理 & 保險
    df = df.dropna(subset=["timestamp", "percent_mvc"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["ts_sec"] = df["timestamp"].astype("int64") / 1e9  # UNIX 秒

    # --- 1) 用較長的時間窗估變化速率（預設 5 分鐘；不足則退回最多 30 筆）---
    t_end = df["timestamp"].iloc[-1]
    window = df[df["timestamp"] >= t_end - pd.Timedelta(minutes=5)]
    if len(window) < 5:
        window = df.tail(min(30, len(df)))

    # --- 2) 其他特徵 ---
    first_mvc = float(df.iloc[0]["percent_mvc"])
    last_mvc = float(df.iloc[-1]["percent_mvc"])
    initial_mvc = max(first_mvc, float(df["percent_mvc"].max()))
    current_mvc = last_mvc
    total_change = float(initial_mvc - current_mvc)

    # --- 3) 用最小二乘回歸算斜率（%/min），並做穩定化 ---
    if len(window) >= 2:
        x = (window["ts_sec"] - window["ts_sec"].iloc[0]).to_numpy()  # 秒差（從 0 開始）
        y = window["percent_mvc"].to_numpy()

        span_min = (x[-1] - x[0]) / 60.0  # 窗口涵蓋的分鐘數
        if span_min <= 0:
            change_rate = 0.0
        else:
            # 中心化後的一元線性回歸斜率（%/秒）
            x_c = x - x.mean()
            y_c = y - y.mean()
            denom = float((x_c * x_c).sum())
            slope_per_sec = float((x_c * y_c).sum() / denom) if denom > 0 else 0.0
            change_rate = slope_per_sec * 60.0  # 換成 %/min
            change_rate = float(np.clip(change_rate, -2.0, 2.0))
    else:
        change_rate = 0.0

    # --- 4) 均值/波動 ---
    avg_mvc = float(window["percent_mvc"].mean())
    std_mvc = float(window["percent_mvc"].std(ddof=0)) if len(window) > 1 else 0.0

    # 和訓練時的特徵順序一致
    X = np.array([[current_mvc, total_change, change_rate, avg_mvc, std_mvc]])
    return X


def _predict_risk(features: np.ndarray) -> Tuple[str, int, str, np.ndarray]:
    """
    回傳：(label, level012, color, proba)
    level012: 0=低 1=中 2=高
    """
    level = int(RF_MODEL.predict(features)[0])
    proba = RF_MODEL.predict_proba(features)[0]
    return RISK_LABELS[level], level, RISK_COLORS[level], proba


def _level13_from_012(level012: int) -> int:
    # 對外 1/2/3
    return int(level012) + 1


# ==================== LSTM / 後援 預測 ====================
def _resample_minutely(df: pd.DataFrame, step_min: int) -> pd.DataFrame:
    """把 percent_mvc 依分鐘重採樣（前向填補），避免 LSTM/外推被不等間隔干擾。"""
    s = df.set_index("timestamp")["percent_mvc"].astype(float)
    s = s.resample(f"{step_min}T").ffill().dropna()
    out = s.to_frame(name="percent_mvc").reset_index()
    return out


def _prepare_seq_for_lstm(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    取最近 HISTORY_SEQUENCE_LENGTH 筆（分鐘取樣），回傳 shape=(1, L, 1)
    """
    if df.empty:
        return None
    rs = _resample_minutely(df, TIME_INTERVAL_MINUTES)
    if len(rs) < HISTORY_SEQUENCE_LENGTH:
        return None
    window = rs.tail(HISTORY_SEQUENCE_LENGTH)["percent_mvc"].to_numpy().astype("float32").reshape(-1, 1)
    if _TF_AVAILABLE and _SCALER is not None:
        x_scaled = _SCALER.transform(window)  # (L, 1)
    else:
        mean = window.mean()
        std = window.std() if window.std() > 1e-6 else 1.0
        x_scaled = (window - mean) / std
    return x_scaled.reshape(1, HISTORY_SEQUENCE_LENGTH, 1)


def _forecast_with_lstm(df: pd.DataFrame) -> Tuple[List[float], str]:
    """
    回傳（未來 PREDICTION_HORIZON 個 %MVC、來源）
    """
    X = _prepare_seq_for_lstm(df)
    if X is None:
        raise ValueError("資料不足以做序列預測")
    if _TF_AVAILABLE and _LSTM_MODEL is not None:
        y_hat = _LSTM_MODEL.predict(X, verbose=0).reshape(-1).astype(float)
        forecast = [float(np.clip(v, 0.0, 120.0)) for v in y_hat[:PREDICTION_HORIZON]]
        return forecast, "lstm"
    # ---- 後援：線性趨勢外推 ----
    rs = _resample_minutely(df, TIME_INTERVAL_MINUTES)
    y = rs.tail(HISTORY_SEQUENCE_LENGTH)["percent_mvc"].to_numpy().astype(float)
    x = np.arange(len(y), dtype=float)
    x_c = x - x.mean()
    denom = float((x_c * x_c).sum()) or 1.0
    slope = float(((x_c * (y - y.mean())).sum()) / denom)
    intercept = float(y.mean() - slope * x.mean())
    future_x = np.arange(len(y), len(y) + PREDICTION_HORIZON, dtype=float)
    y_future = intercept + slope * future_x
    y_future = np.clip(y_future, 0.0, 120.0)
    return [float(v) for v in y_future], "naive"


def _risk_for_future_points(df_hist: pd.DataFrame, future_vals: List[float]) -> Dict[int, Dict[str, Any]]:
    """
    把預測的未來點接到歷史序列後，挑選 5/15/30 分鐘位置做 RF 風險分級。
    回傳：{5: {...}, 15: {...}, 30: {...}}
    """
    rs = _resample_minutely(df_hist, TIME_INTERVAL_MINUTES)
    base_ts = rs["timestamp"].iloc[-1]
    results: Dict[int, Dict[str, Any]] = {}
    for m in [5, 15, 30]:
        if m <= 0 or m > len(future_vals):
            continue
        fut = pd.DataFrame(
            {
                "timestamp": [base_ts + pd.Timedelta(minutes=i + 1) for i in range(m)],
                "percent_mvc": future_vals[:m],
            }
        )
        df_tmp = pd.concat([rs[["timestamp", "percent_mvc"]], fut], ignore_index=True)
        feats = _extract_features(df_tmp)
        if feats is None:
            continue
        label, level012, color, proba = _predict_risk(feats)
        results[m] = {
            "label": label,
            "level": _level13_from_012(level012),
            "color": color,
            "prob": [float(x) for x in proba],
        }
    return results


# ==================== 路由 ====================
@app.get("/")
def home():
    return {
        "service": APP_TITLE,
        "version": "v1.3",
        "build": APP_BUILD,
        "endpoints": {
            "健康檢查": "GET /healthz",
            "總覽": "GET /health",
            "App 專用": "GET /app_data",
            "未來預測": "GET /forecast",
            "單筆上傳": "POST /upload",
            "批次上傳": "POST /process_json",
            "查詢狀態": "GET /status/{worker_id}",
            "清空資料": "DELETE /clear/{worker_id}",
        },
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok", "build": APP_BUILD}


@app.get("/health")
def health():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM sensor_data")
    total = c.fetchone()[0]
    conn.close()
    return {
        "status": "healthy",
        "db": {"path": DB_PATH, "total_records": total},
        "models": {
            "rf_loaded": RF_MODEL is not None,
            "rf_path": RF_MODEL_PATH,
            "lstm_loaded": bool(_TF_AVAILABLE and _LSTM_MODEL is not None and _SCALER is not None),
            "lstm_path": LSTM_MODEL_PATH if os.path.exists(LSTM_MODEL_PATH) else None,
        },
        "forecast": {
            "history_len": HISTORY_SEQUENCE_LENGTH,
            "horizon_min": PREDICTION_HORIZON,
            "step_min": TIME_INTERVAL_MINUTES,
            "source": _LSTM_SOURCE,
        },
        "version": "v1.3",
        "build": APP_BUILD,
    }


# ---- 單筆上傳（你原本的 /upload） ----
@app.post("/upload")
def upload(item: SensorData):
    worker_id = DEFAULT_WORKER_ID
    ts_iso = item.timestamp or now_taiwan_iso()

    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO sensor_data(worker_id, timestamp, percent_mvc, ts, rms, type) VALUES(?,?,?,?,?,?)",
        (worker_id, ts_iso, float(item.percent_mvc), None, None, "mvc"),
    )
    conn.commit()
    conn.close()

    return {"status": "success", "worker_id": worker_id, "timestamp": ts_iso, "mvc": float(item.percent_mvc)}


# ---- App 簡化端點（多回 forecast）----
@app.get("/app_data")
def get_app_data():
    df = _get_df(DEFAULT_WORKER_ID, limit=120, exclude_heartbeat=True)
    if df.empty:
        return {
            "user_id": DEFAULT_WORKER_ID,
            "last_update": None,
            "risk_level": "無資料",
            "forecast": None,
        }

    feats = _extract_features(df)
    if feats is None:
        forecast_vals = None
        source = _LSTM_SOURCE
        try:
            vals, source = _forecast_with_lstm(df)
            forecast_vals = vals
        except Exception:
            forecast_vals = None
        return {
            "user_id": DEFAULT_WORKER_ID,
            "last_update": str(df.iloc[-1]["timestamp"]),
            "risk_level": "資料不足",
            "forecast": None
            if forecast_vals is None
            else {
                "horizon_min": min(PREDICTION_HORIZON, len(forecast_vals)),
                "step_min": TIME_INTERVAL_MINUTES,
                "values": forecast_vals,
                "source": source,
            },
        }

    label, level, color, _ = _predict_risk(feats)

    forecast_block = None
    try:
        vals, source = _forecast_with_lstm(df)
        risks = _risk_for_future_points(df, vals)
        forecast_block = {
            "horizon_min": min(PREDICTION_HORIZON, len(vals)),
            "step_min": TIME_INTERVAL_MINUTES,
            "values": vals,
            "source": source,
            "risk_at_min": risks,  # {5:{...},15:{...},30:{...}}
        }
    except Exception:
        forecast_block = None

    return {
        "user_id": DEFAULT_WORKER_ID,
        "last_update": str(df.iloc[-1]["timestamp"]),
        "risk_level": label,
        "forecast": forecast_block,
    }


# ---- 獨立的未來預測端點 ----
@app.get("/forecast")
def forecast(worker_id: Optional[str] = None):
    wid = str((worker_id or DEFAULT_WORKER_ID)).strip()
    df = _get_df(wid, limit=HISTORY_SEQUENCE_LENGTH * 4, exclude_heartbeat=True)
    if df.empty or len(df) < max(5, HISTORY_SEQUENCE_LENGTH // 4):
        raise HTTPException(422, detail="資料不足，無法進行未來預測")

    vals, source = _forecast_with_lstm(df)
    risks = _risk_for_future_points(df, vals)
    last_ts = str(df.iloc[-1]["timestamp"])
    return {
        "worker_id": wid,
        "last_update": last_ts,
        "horizon_min": min(PREDICTION_HORIZON, len(vals)),
        "step_min": TIME_INTERVAL_MINUTES,
        "values": vals,
        "source": source,  # "lstm" 或 "naive"
        "risk_at_min": risks,  # {5:{label,level,color,prob}, 15:{...}, 30:{...}}
    }


# ---- 批次上傳（Flutter App 用）----
@app.post("/process_json")
def process_json(rows: Union[List[Dict[str, Any]], Dict[str, Any]]):
    payload = _ensure_json_array(rows)

    # 正規化 + 刻度換算 + 時間修正（防 1970）
    normalized = _normalize_rows(payload)
    if not normalized:
        raise HTTPException(400, detail="空的上傳資料")

    conn = get_conn()
    c = conn.cursor()
    inserted = 0
    last_worker = None

    try:
        for r in normalized:
            worker_id = str((r.get("worker_id") or DEFAULT_WORKER_ID)).strip()
            ts_iso = r.get("timestamp") or now_taiwan_iso()
            pmv = r.get("percent_mvc")
            pmv = float(pmv) if pmv is not None else 0.0
            ts_sec = r.get("ts")
            ts_sec = float(ts_sec) if ts_sec is not None else None
            rms = r.get("rms")
            rms = float(rms) if rms is not None else None
            typ = r.get("type")

            c.execute(
                "INSERT INTO sensor_data(worker_id, timestamp, percent_mvc, ts, rms, type) VALUES(?,?,?,?,?,?)",
                (worker_id, ts_iso, pmv, ts_sec, rms, typ),
            )
            inserted += 1
            last_worker = worker_id
        conn.commit()
    except sqlite3.OperationalError as e:
        raise HTTPException(status_code=503, detail=f"db_error: {e}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # 回傳同時附上最新風險等級（方便 App 立即更新）
    if last_worker is None:
        last_worker = DEFAULT_WORKER_ID
    df = _get_df(last_worker, limit=120, exclude_heartbeat=True)
    if df.empty:
        resp = {"status": "ok", "inserted": inserted, "worker_id": last_worker}
    else:
        feats = _extract_features(df)
        if feats is None:
            resp = {
                "status": "ok",
                "inserted": inserted,
                "worker_id": last_worker,
                "risk": {"label": "資料不足", "level": 1, "color": "#95a5a6"},
            }
        else:
            label012, level012, color, proba = _predict_risk(feats)
            resp = {
                "status": "ok",
                "inserted": inserted,
                "worker_id": last_worker,
                "risk": {
                    "label": label012,
                    "level": _level13_from_012(level012),  # 1/2/3
                    "color": color,
                    "prob": [float(x) for x in proba],
                },
            }
    return resp


# ---- 查詢狀態（App 拉心跳/狀態用）----
@app.get("/status/{worker_id}")
def status(worker_id: str):
    wid = str((worker_id or DEFAULT_WORKER_ID)).strip()
    df = _get_df(wid, limit=120, exclude_heartbeat=True)
    if df.empty:
        return {
            "worker_id": wid,
            "status": "無資料",
            "risk_label": "低度",
            "risk_level": 1,
            "risk_color": RISK_COLORS[0],
            "last": None,
        }

    feats = _extract_features(df)
    if feats is None:
        last = df.iloc[-1]
        return {
            "worker_id": wid,
            "status": "資料不足",
            "risk_label": "低度",
            "risk_level": 1,
            "risk_color": RISK_COLORS[0],
            "last": {
                "timestamp": str(last["timestamp"]),
                "percent_mvc": float(last["percent_mvc"]),
                "rms": float(last["rms"]) if pd.notna(last.get("rms", np.nan)) else None,
            },
        }

    label012, level012, color, proba = _predict_risk(feats)
    last = df.iloc[-1]
    return {
        "worker_id": wid,
        "status": "ok",
        "risk_label": label012,
        "risk_level": _level13_from_012(level012),  # 1/2/3
        "risk_color": color,
        "prob": [float(x) for x in proba],
        "last": {
            "timestamp": str(last["timestamp"]),
            "percent_mvc": float(last["percent_mvc"]),
            "rms": float(last["rms"]) if pd.notna(last.get("rms", np.nan)) else None,
        },
    }


# ---- 清空資料（測試用）----
@app.delete("/clear/{worker_id}")
def clear(worker_id: str):
    wid = str((worker_id or DEFAULT_WORKER_ID)).strip()
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM sensor_data WHERE worker_id = ?", (wid,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return {"status": "success", "deleted": deleted, "worker_id": wid}


# ==================== 本機啟動提示 ====================
if __name__ == "__main__":
    print("🚀 疲勞預測系統啟動")
    print(f"📦 DB: {DB_PATH}")
    print(f"👤 預設使用者: {DEFAULT_WORKER_ID}")
    print(f"🧱 Build: {APP_BUILD}")
    print("📍 本機: http://localhost:8000")
    print("➡️  健康檢查:   GET http://localhost:8000/healthz")
    print("➡️  批次上傳:   POST http://localhost:8000/process_json")
    print("➡️  查詢狀態:   GET  http://localhost:8000/status/user001")
    print("➡️  App 簡化:   GET  http://localhost:8000/app_data")
    print("➡️  未來預測:   GET  http://localhost:8000/forecast")
    print("\n啟動：uvicorn main:app --reload --host 0.0.0.0 --port 8000")
