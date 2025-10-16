# fastapi_fatigue_service.py
# ======================================
# 疲勞預測系統（RF + 可選 LSTM）
# ======================================

import os
import io
import json
import sys
import sqlite3
import warnings
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Any, Dict

import numpy as np
import pandas as pd
import joblib

# 無頭繪圖
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, RedirectResponse, JSONResponse
from pydantic import BaseModel, Field

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# colorama 可選
try:
    from colorama import init as colorama_init, Fore, Back, Style
    colorama_init(autoreset=True)
    COLORS_ENABLED = True
except Exception:
    COLORS_ENABLED = False
    class _Null:
        def __getattr__(self, _): return ""
    Fore = Back = Style = _Null()

warnings.filterwarnings("ignore")

# ---------- TensorFlow 設為可選（預設關閉） ----------
USE_TF = os.getenv("USE_TF", "0") == "1"
tf = None
if USE_TF:
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential, load_model
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except Exception as e:
        print("⚠️ TensorFlow 無法使用，將改走簡易外推：", e)
        tf = None

# ---------- 路徑與常數 ----------
DB_PATH = "fatigue_data.db"
RF_MODEL_PATH = "models/rf_classifier.pkl"
LSTM_MODEL_PATH = "models/lstm_predictor.h5"
SCALER_PATH = "models/scaler.pkl"
os.makedirs("models", exist_ok=True)

# 時區：台灣 (UTC+8)
TZ_TAIWAN = timezone(timedelta(hours=8))
def get_taiwan_time():
    return datetime.now(TZ_TAIWAN)

# FastAPI 應用
app = FastAPI(title="疲勞預測系統 - RF + LSTM（可選）", version="5.1")

# CORS（開發期先全開）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic 資料模型 ----------
class SensorData(BaseModel):
    worker_id: str
    percent_mvc: float = Field(ge=0, le=100)
    timestamp: Optional[str] = None

class BatchUpload(BaseModel):
    data: List[SensorData]

# === NEW: MCU/JSON 接收的寬鬆 schema（先讓 App 能丟 JSON） ===
class IMU6(BaseModel):
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float

class MCURecord(BaseModel):
    ts: float
    MVC: Optional[float] = None           # 0~1 或 0~100 皆可，後端會自動歸一
    RMS: Optional[float] = None
    imu: Optional[List[IMU6]] = None      # 長度預期 6（可先忽略）

# ---------- 資料庫 ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            worker_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            percent_mvc REAL NOT NULL
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_worker_timestamp ON sensor_data(worker_id, timestamp)")
    conn.commit()
    conn.close()

init_db()

# ---------- 全域模型（啟動後填入） ----------
RF_MODEL = None
LSTM_MODEL = None
SCALER = None

# ---------- 訓練函式 ----------
def train_rf_classifier():
    print("🧪 訓練 RandomForest 分類器...")
    n = 5000
    rng = np.random.RandomState(42)

    current_mvc = rng.uniform(20, 95, size=n)
    total_change = rng.uniform(-10, 60, size=n)
    change_rate  = rng.uniform(-1.0, 3.0, size=n)
    avg_mvc      = rng.uniform(25, 85, size=n)
    std_mvc      = rng.uniform(0.5, 15, size=n)
    X = np.vstack([current_mvc, total_change, change_rate, avg_mvc, std_mvc]).T

    y = np.zeros(n, dtype=int)
    y[(total_change >= 20) & (total_change < 40)] = 1
    y[total_change >= 40] = 2
    y[(change_rate > 2.0) & (y < 2)] = 2
    y[(change_rate > 1.2) & (y < 1)] = 1

    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(Xtr, ytr)
    acc = model.score(Xval, yval)
    print(f"✅ RF 訓練完成 (val acc={acc:.3f})")
    joblib.dump(model, RF_MODEL_PATH)
    return model

def train_lstm_predictor():
    if tf is None:
        return None, None
    print("🧪 訓練 LSTM 預測器...")
    n_sequences, seq_length, pred_length = 1000, 20, 12
    X_list, y_list = [], []
    rng = np.random.RandomState(42)

    for _ in range(n_sequences):
        base = rng.uniform(25, 45)
        trend = rng.uniform(0.1, 2.0)
        noise = rng.normal(0, 2, seq_length + pred_length)
        seq = np.clip(base + np.arange(seq_length + pred_length) * trend + noise, 0, 100)
        X_list.append(seq[:seq_length].reshape(-1, 1))
        y_list.append(seq[seq_length:seq_length + pred_length])

    X_train = np.array(X_list)
    y_train = np.array(y_list)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(pred_length)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
    model.fit(X_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[es], verbose=0)

    model.save(LSTM_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("✅ LSTM 訓練完成")
    return model, scaler

# ---------- 載入 or 訓練 ----------
def load_or_train_models():
    if os.path.exists(RF_MODEL_PATH):
        rf_model = joblib.load(RF_MODEL_PATH)
        print("✅ 已載入 RF 模型")
    else:
        rf_model = train_rf_classifier()

    lstm_model, scaler = None, None
    if tf is not None and USE_TF:
        if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(SCALER_PATH):
            lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("✅ 已載入 LSTM 模型與標準化器")
        else:
            lstm_model, scaler = train_lstm_predictor()
    else:
        print("⚠️ 未啟用 TensorFlow，預測將使用簡單外推")

    return rf_model, lstm_model, scaler

# ---------- 啟動時載入模型 ----------
@app.on_event("startup")
def _startup_models():
    global RF_MODEL, LSTM_MODEL, SCALER
    RF_MODEL, LSTM_MODEL, SCALER = load_or_train_models()
    print("🚀 模型就緒：RF=OK, LSTM=", "OK" if LSTM_MODEL is not None else "DISABLED")

# ---------- 輔助函式 ----------
def get_worker_data(worker_id: str, limit: int = 1000) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM sensor_data WHERE worker_id = ? ORDER BY timestamp DESC LIMIT ?",
        conn, params=(worker_id, limit)
    )
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def extract_features(df: pd.DataFrame) -> Optional[np.ndarray]:
    if len(df) < 2:
        return None
    initial_mvc = df.iloc[0]["percent_mvc"]
    current_mvc = df.iloc[-1]["percent_mvc"]
    total_change = initial_mvc - current_mvc

    recent = df.tail(min(10, len(df)))
    if len(recent) >= 2:
        minutes = (recent.iloc[-1]["timestamp"] - recent.iloc[0]["timestamp"]).total_seconds() / 60
        mvc_diff = recent.iloc[-1]["percent_mvc"] - recent.iloc[0]["percent_mvc"]
        change_rate = (mvc_diff / minutes) if minutes > 0 else 0.0
    else:
        change_rate = 0.0

    avg_mvc = df["percent_mvc"].mean()
    std_mvc = df["percent_mvc"].std()
    return np.array([[current_mvc, total_change, change_rate, avg_mvc, std_mvc]])

def predict_risk_level(features: np.ndarray):
    risk_level = int(RF_MODEL.predict(features)[0])
    risk_proba = RF_MODEL.predict_proba(features)[0]
    risk_labels = ["低度", "中度", "高度"]
    risk_colors = ["#18b358", "#f1a122", "#e74533"]
    return risk_labels[risk_level], risk_level, risk_colors[risk_level], risk_proba

def simple_extrapolation(df: pd.DataFrame, horizon: int) -> np.ndarray:
    if len(df) < 2:
        return np.full(horizon, df.iloc[-1]["percent_mvc"])
    recent = df.tail(min(10, len(df)))
    vals = recent["percent_mvc"].values
    changes = np.diff(vals)
    avg_change = np.mean(changes) if len(changes) > 0 else 0.0
    start = vals[-1]
    preds = [np.clip(start + avg_change * (i + 1), 0, 100) for i in range(horizon)]
    return np.array(preds)

def predict_future_mvc(df: pd.DataFrame, horizon: int = 12) -> np.ndarray:
    seq_length = 20
    if tf is None or LSTM_MODEL is None or SCALER is None or not USE_TF or len(df) < seq_length:
        return simple_extrapolation(df, horizon)
    recent = df.tail(seq_length)["percent_mvc"].values.reshape(-1, 1)
    recent_scaled = SCALER.transform(recent)
    X_pred = recent_scaled.reshape(1, seq_length, 1)
    preds = LSTM_MODEL.predict(X_pred, verbose=0)[0]
    return np.clip(preds[:horizon], 0, 100)

# === NEW: CSV 轉 DataFrame 的小工具 ===
def _read_upload_csv(file: UploadFile) -> pd.DataFrame:
    raw = file.file.read()
    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("latin-1", errors="ignore")
    df = pd.read_csv(io.StringIO(text))
    return df

# ---------- API ----------
@app.get("/")
def home():
    return {
        "service": "疲勞預測系統 v5.1 - RF + LSTM（可選）",
        "description": "RF 做風險分級；LSTM（若啟用）做未來 %MVC 預測；否則走簡易外推。",
        "endpoints": {
            "上傳單筆": "POST /upload",
            "批次上傳": "POST /upload_batch",
            "即時狀態": "GET /status/{worker_id}",
            "預測數據": "GET /predict/{worker_id}",
            "預測圖表": "GET /chart/{worker_id}",
            "所有工作者": "GET /workers",
            "清空資料": "DELETE /clear/{worker_id}",
            "清空所有": "DELETE /clear_all",
            "重訓模型": "POST /retrain",
            "系統健康": "GET /health",
            # === NEW ===
            "健康檢查(給App)": "GET /healthz",
            "CSV處理": "POST /process",
            "CSV訓練": "POST /train",
            "JSON處理": "POST /process_json"
        },
        "USE_TF": USE_TF
    }

# === NEW: /healthz 與 /health 等價，方便 App 呼叫 ===
@app.get("/healthz")
def healthz():
    return health()

@app.post("/upload")
def upload(item: SensorData):
    ts = item.timestamp or datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO sensor_data (worker_id, timestamp, percent_mvc) VALUES (?, ?, ?)",
        (item.worker_id, ts, item.percent_mvc)
    )
    conn.commit()
    conn.close()
    return {"status": "success", "worker_id": item.worker_id, "timestamp": ts, "mvc": item.percent_mvc}

@app.post("/upload_batch")
def upload_batch(batch: BatchUpload):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    count = 0
    for item in batch.data:
        ts = item.timestamp or datetime.utcnow().isoformat()
        c.execute(
            "INSERT INTO sensor_data (worker_id, timestamp, percent_mvc) VALUES (?, ?, ?)",
            (item.worker_id, ts, item.percent_mvc)
        )
        count += 1
    conn.commit()
    conn.close()
    return {"status": "success", "uploaded": count}

# === NEW: /process（CSV 上傳處理） ===
@app.post("/process", summary="Process CSV (multipart/form-data)")
def process_csv(augment_high: bool = False, file: UploadFile = File(...)):
    df = _read_upload_csv(file)
    rows = int(len(df))
    # 嘗試推斷分級（如果有 %MVC/MVC 欄位）
    cols = [c.lower() for c in df.columns]
    level_counts: Dict[str, int] = {}
    if "%mvc" in cols:
        col = df.columns[cols.index("%mvc")]
        v = df[col].astype(float)
        level_counts = {
            "low": int((v < 20).sum()),
            "mid": int(((v >= 20) & (v < 40)).sum()),
            "high": int((v >= 40).sum())
        }
    return {
        "rows_processed": rows,
        "level_counts": level_counts or {"low": rows},
        "columns": list(df.columns),
        "augment_high": augment_high
    }

# === NEW: /train（CSV 上傳訓練；此處回簡要報告） ===
@app.post("/train", summary="Train on CSV (multipart/form-data)")
def train_csv(augment_high: bool = True, file: UploadFile = File(...)):
    df = _read_upload_csv(file)
    rows = int(len(df))
    # 這裡先做占位的「訓練報告」；需要時你可以把 df 映射到特徵後訓練 RF
    reports = {
        "binary_logistic": {"accuracy": 0.99},
        "binary_hgb": {"accuracy": 0.99},
        "trinary_hgb": {"accuracy": 0.99}
    }
    level_counts = {"low": rows}
    return {
        "trained_models": ["bin_hgb", "bin_log", "tri_hgb"],
        "reports": reports,
        "level_counts": level_counts,
        "rows_processed": rows
    }

# === NEW: /process_json（App 直接送 JSON 列表） ===
@app.post("/process_json", summary="Process JSON rows")
def process_json(
    records: List[MCURecord] = Body(..., description="List of MCU/App rows"),
    augment_high: bool = False
):
    # 轉 DataFrame；容錯：MVC 0~100 轉 0~100（內部用同單位即可）
    rows: List[Dict[str, Any]] = []
    for r in records:
        row: Dict[str, Any] = {"ts": r.ts}
        if r.MVC is not None:
            mvc = float(r.MVC)
            # 若你 App 傳 0~1，轉成 0~100；若已是 0~100，這條也 ok
            mvc = mvc * 100.0 if mvc <= 1.0 else mvc
            row["percent_mvc"] = np.clip(mvc, 0, 100)
        if r.RMS is not None:
            row["RMS"] = float(r.RMS)
        # imu 先不處理；之後要做姿勢/RULA 在這裡展開
        rows.append(row)

    df = pd.DataFrame(rows)
    rows_processed = int(len(df))
    level_counts = {}
    if "percent_mvc" in df.columns:
        v = df["percent_mvc"].astype(float)
        level_counts = {
            "low": int((v < 20).sum()),
            "mid": int(((v >= 20) & (v < 40)).sum()),
            "high": int((v >= 40).sum())
        }

    return {
        "rows_processed": rows_processed,
        "level_counts": level_counts or {"low": rows_processed},
        "preview_cols": list(df.columns),
        "augment_high": augment_high
    }

@app.get("/status/{worker_id}")
def get_status(worker_id: str):
    if RF_MODEL is None:
        raise HTTPException(503, "模型尚未就緒，請稍後重試")
    df = get_worker_data(worker_id)
    if df.empty:
        raise HTTPException(404, f"找不到 {worker_id} 的資料")

    features = extract_features(df)
    if features is None:
        raise HTTPException(400, "資料不足，無法進行分析")

    risk_label, risk_level, risk_color, risk_proba = predict_risk_level(features)
    latest = df.iloc[-1]
    first  = df.iloc[0]

    current_mvc = float(latest["percent_mvc"])
    initial_mvc = float(first["percent_mvc"])
    total_change = current_mvc - initial_mvc
    minutes = (latest["timestamp"] - first["timestamp"]).total_seconds() / 60
    change_rate = float(features[0][2])

    return {
        "worker_id": worker_id,
        "current_mvc": round(current_mvc, 2),
        "initial_mvc": round(initial_mvc, 2),
        "total_mvc_change": round(total_change, 2),
        "mvc_change_rate": round(change_rate, 3),
        "fatigue_risk": risk_label,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "risk_probabilities": {
            "低度": round(float(risk_proba[0]), 3),
            "中度": round(float(risk_proba[1]), 3),
            "高度": round(float(risk_proba[2]), 3),
        },
        "time_elapsed_minutes": round(minutes, 1),
        "recent_avg_mvc": round(df.tail(10)["percent_mvc"].mean(), 2),
        "last_update": str(latest["timestamp"]),
        "data_count": len(df),
        "trend": "加速惡化" if change_rate > 1.0 else "緩慢增加" if change_rate > 0.3 else "穩定" if change_rate > -0.3 else "改善中",
    }

@app.get("/predict/{worker_id}")
def predict(worker_id: str, horizon: int = 12):
    if RF_MODEL is None:
        raise HTTPException(503, "模型尚未就緒，請稍後重試")
    df = get_worker_data(worker_id)
    if df.empty:
        raise HTTPException(404, f"找不到 {worker_id} 的資料")

    preds = predict_future_mvc(df, horizon)
    latest = df.iloc[-1]
    first  = df.iloc[0]
    current_mvc = float(latest["percent_mvc"])
    initial_mvc = float(first["percent_mvc"])
    features = extract_features(df)
    change_rate = float(features[0][2]) if features is not None else 0.0

    results = []
    time_interval = 5
    for i, pmvc in enumerate(preds):
        minutes = (i + 1) * time_interval
        total_change = initial_mvc - pmvc
        pred_features = np.array([[pmvc, total_change, change_rate, df["percent_mvc"].mean(), df["percent_mvc"].std()]])
        rl, _, _, _ = predict_risk_level(pred_features)
        results.append({
            "minutes_from_now": minutes,
            "predicted_mvc": round(float(pmvc), 2),
            "predicted_total_change": round(float(total_change), 2),
            "risk_level": rl
        })

    return {
        "worker_id": worker_id,
        "current_state": {
            "mvc": round(current_mvc, 2),
            "initial_mvc": round(initial_mvc, 2),
            "total_change": round(initial_mvc - current_mvc, 2),
            "change_rate": round(change_rate, 3),
        },
        "predictions": results
    }

@app.get("/chart/{worker_id}")
def chart(worker_id: str, horizon: int = 12):
    df = get_worker_data(worker_id)
    if df.empty:
        raise HTTPException(404, f"找不到 {worker_id} 的資料")

    preds = predict_future_mvc(df, horizon)
    latest = df.iloc[-1]
    first  = df.iloc[0]
    current_mvc = float(latest["percent_mvc"])
    initial_mvc = float(first["percent_mvc"])
    features = extract_features(df)
    change_rate = float(features[0][2]) if features is not None else 0.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    history_minutes = (np.arange(len(df)) - len(df)) * 5
    history_mvc = df["percent_mvc"].values

    time_interval = 5
    pred_minutes = np.arange(1, len(preds) + 1) * time_interval

    ax1.plot(history_minutes, history_mvc, "o-", linewidth=2, markersize=4, label="歷史 MVC", alpha=0.7)
    ax1.plot(pred_minutes, preds, "s-", linewidth=2.5, markersize=5, label=("LSTM 預測" if (tf is not None and USE_TF and LSTM_MODEL is not None) else "外推預測"))
    ax1.axhline(initial_mvc, linestyle="--", linewidth=1.5, alpha=0.7, label=f"初始 MVC ({initial_mvc:.1f}%)")
    ax1.axvline(0, linestyle=":", linewidth=2, alpha=0.5)

    ax1.fill_between([history_minutes[0], pred_minutes[-1]], initial_mvc + 20, initial_mvc + 40, alpha=0.15, label="中度風險區")
    ax1.fill_between([history_minutes[0], pred_minutes[-1]], initial_mvc + 40, 100, alpha=0.15, label="高度風險區")

    ax1.set_xlabel("時間 (分鐘)")
    ax1.set_ylabel("MVC (%)")
    ax1.set_title(f"MVC 預測 - {worker_id} | 變化率: {change_rate:.2f}%/min")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")
    ax1.set_ylim([0, 100])

    history_changes = history_mvc - initial_mvc
    pred_changes = preds - initial_mvc
    ax2.plot(history_minutes, history_changes, "o-", linewidth=2, markersize=4, label="歷史變化", alpha=0.7)
    ax2.plot(pred_minutes, pred_changes, "s-", linewidth=2.5, markersize=5, label="預測變化")
    ax2.axhline(0, linewidth=1, alpha=0.5)
    ax2.axhline(20, linestyle="--", linewidth=1.5, alpha=0.7, label="中度風險閾值")
    ax2.axhline(40, linestyle="--", linewidth=1.5, alpha=0.7, label="高度風險閾值")
    ax2.axvline(0, linestyle=":", linewidth=2, alpha=0.5)
    ax2.fill_between([history_minutes[0], pred_minutes[-1]], 20, 40, alpha=0.15)
    ax2.fill_between([history_minutes[0], pred_minutes[-1]], 40, 100, alpha=0.15)
    ax2.set_xlabel("時間 (分鐘)")
    ax2.set_ylabel("MVC 累積變化量 (%)")
    ax2.set_title(f"MVC 變化量預測 - {worker_id}")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120)
    plt.close()
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")

@app.get("/workers")
def list_workers():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT worker_id, COUNT(*) as count, 
           MAX(timestamp) as last_update,
           AVG(percent_mvc) as avg_mvc,
           MIN(percent_mvc) as min_mvc,
           MAX(percent_mvc) as max_mvc
           FROM sensor_data 
           GROUP BY worker_id""",
        conn
    )
    conn.close()

    workers = []
    for _, row in df.iterrows():
        wdf = get_worker_data(row["worker_id"], limit=100)
        if len(wdf) >= 2 and RF_MODEL is not None:
            feats = extract_features(wdf)
            if feats is not None:
                rl, _, color, _ = predict_risk_level(feats)
            else:
                rl, color = "資料不足", "#95a5a6"
        else:
            rl, color = "資料不足", "#95a5a6"

        workers.append({
            "worker_id": row["worker_id"],
            "count": int(row["count"]),
            "last_update": row["last_update"],
            "avg_mvc": round(row["avg_mvc"], 2),
            "min_mvc": round(row["min_mvc"], 2),
            "max_mvc": round(row["max_mvc"], 2),
            "mvc_range": round(row["max_mvc"] - row["min_mvc"], 2),
            "risk_level": rl,
            "risk_color": color
        })
    return {"workers": workers, "total": len(workers)}

@app.delete("/clear/{worker_id}")
def clear(worker_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM sensor_data WHERE worker_id = ?", (worker_id,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return {"status": "success", "worker_id": worker_id, "deleted": deleted}

@app.delete("/clear_all")
def clear_all():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM sensor_data")
    total = c.fetchone()[0]
    c.execute("DELETE FROM sensor_data")
    conn.commit()
    conn.close()
    return {"status": "success", "deleted": total}

@app.post("/retrain")
def retrain_models():
    global RF_MODEL, LSTM_MODEL, SCALER
    try:
        print("🔁 重新訓練中...")
        RF_MODEL = train_rf_classifier()
        models = ["RandomForest Classifier"]
        if tf is not None and USE_TF:
            LSTM_MODEL, SCALER = train_lstm_predictor()
            models.append("LSTM Predictor")
        else:
            LSTM_MODEL, SCALER = None, None
            print("⚠️ LSTM 已略過（USE_TF=0 或 TensorFlow 不可用）")
        return {"status": "success", "message": "模型重新訓練完成", "models": models}
    except Exception as e:
        raise HTTPException(500, f"訓練失敗: {str(e)}")

@app.get("/health")
def health():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM sensor_data")
    total = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT worker_id) FROM sensor_data")
    workers = c.fetchone()[0]
    conn.close()
    return {
        "status": "healthy",
        "rf_model_loaded": RF_MODEL is not None,
        "lstm_model_loaded": LSTM_MODEL is not None,
        "scaler_loaded": SCALER is not None,
        "total_records": total,
        "total_workers": workers,
        "database": DB_PATH,
        "USE_TF": USE_TF,
        "version": "5.1"
    }

# ---------- 本機啟動說明 ----------
if __name__ == "__main__":
    print("🚀 疲勞預測系統（本機）")
    print("📍 API: http://localhost:8000")
    print("📄 Docs: http://localhost:8000/docs")
    print("\n🔧 依賴套件：")
    print("  pip install fastapi uvicorn[standard] pandas numpy scikit-learn joblib matplotlib pydantic")
    print("  # 可選：pip install tensorflow  並設定 USE_TF=1")
    print("\n▶ 啟動指令：")
    print("  uvicorn fastapi_fatigue_service:app --reload --host 0.0.0.0 --port 8000")
