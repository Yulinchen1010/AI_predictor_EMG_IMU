from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import sqlite3
import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta, timezone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# ==================== 初始設定 ====================
DB_PATH = "fatigue_data.db"
RF_MODEL_PATH = "models/rf_classifier.pkl"
LSTM_MODEL_PATH = "models/lstm_predictor.keras"
SCALER_PATH = "models/scaler.pkl"
os.makedirs("models", exist_ok=True)

# 固定使用者 ID
SINGLE_USER_ID = "user001"

# 預測參數
TIME_INTERVAL_MINUTES = 1
HISTORY_SEQUENCE_LENGTH = 60
PREDICTION_HORIZON = 30
MAX_DATA_LIMIT = 1000

# 台灣時區
TZ_TAIWAN = timezone(timedelta(hours=8))

def get_taiwan_time():
    return datetime.now(TZ_TAIWAN)

app = FastAPI(title="疲勞預測系統")

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 資料模型 ====================
class SensorData(BaseModel):
    percent_mvc: float = Field(ge=0, le=100)
    timestamp: Optional[str] = None

# ==================== 資料庫初始化 ====================
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

# ==================== 模型訓練 ====================
def train_rf_classifier():
    print("訓練 RandomForest 分類器...")
    n = 5000
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
    
    model = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10, 
                                   min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)
    
    score = model.score(X_val, y_val)
    print(f"RandomForest 訓練完成 (準確率 = {score:.3f})")
    joblib.dump(model, RF_MODEL_PATH)
    return model

def train_lstm_predictor():
    print("訓練 LSTM 預測器...")
    n_sequences = 1000
    seq_length = HISTORY_SEQUENCE_LENGTH
    pred_length = PREDICTION_HORIZON
    
    X_train_list = []
    y_train_list = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_sequences):
        base = rng.uniform(60, 90)
        trend = rng.uniform(-2.0, -0.1)
        noise = rng.normal(0, 2, seq_length + pred_length)
        sequence = base + np.arange(seq_length + pred_length) * trend + noise
        sequence = np.clip(sequence, 0, 100)
        
        X_train_list.append(sequence[:seq_length].reshape(-1, 1))
        y_train_list.append(sequence[seq_length:seq_length + pred_length])
    
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(pred_length)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, 
                       validation_split=0.2, callbacks=[early_stop], verbose=0)
    
    final_mae = history.history.get('mean_absolute_error', [0])[-1]
    print(f"LSTM 訓練完成 (MAE = {final_mae:.3f})")
    
    model.save(LSTM_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler

def load_or_train_models():
    if os.path.exists(RF_MODEL_PATH):
        print("✅ 載入 RandomForest")
        rf_model = joblib.load(RF_MODEL_PATH)
    else:
        rf_model = train_rf_classifier()
    
    if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("✅ 載入 LSTM")
        lstm_model = load_model(LSTM_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        lstm_model, scaler = train_lstm_predictor()
    
    return rf_model, lstm_model, scaler

RF_MODEL, LSTM_MODEL, SCALER = load_or_train_models()

# ==================== 輔助函數 ====================
def get_worker_data(limit: int = None) -> pd.DataFrame:
    if limit is None:
        limit = MAX_DATA_LIMIT
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM sensor_data WHERE worker_id = ? ORDER BY timestamp DESC LIMIT {limit}",
        conn, params=(SINGLE_USER_ID,)
    )
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
    return df

def extract_features(df: pd.DataFrame) -> np.ndarray:
    if len(df) < 2:
        return None
    
    first_mvc = df.iloc[0]['percent_mvc']
    last_mvc = df.iloc[-1]['percent_mvc']
    
    if first_mvc > last_mvc:
        initial_mvc = first_mvc
        current_mvc = last_mvc
    else:
        initial_mvc = df['percent_mvc'].max()
        current_mvc = last_mvc
    
    total_change = initial_mvc - current_mvc
    
    recent_df = df.tail(min(10, len(df)))
    if len(recent_df) >= 2:
        time_diff = (recent_df.iloc[-1]['timestamp'] - recent_df.iloc[0]['timestamp']).total_seconds() / 60
        if time_diff > 0:
            mvc_diff = recent_df.iloc[0]['percent_mvc'] - recent_df.iloc[-1]['percent_mvc']
            change_rate = mvc_diff / time_diff
        else:
            change_rate = 0.0
    else:
        change_rate = 0.0
    
    avg_mvc = df['percent_mvc'].mean()
    std_mvc = df['percent_mvc'].std()
    
    features = np.array([[current_mvc, total_change, change_rate, avg_mvc, std_mvc]])
    return features

def predict_risk_level(features: np.ndarray) -> tuple:
    risk_level = int(RF_MODEL.predict(features)[0])
    risk_proba = RF_MODEL.predict_proba(features)[0]
    risk_labels = ["低度", "中度", "高度"]
    risk_colors = ["#18b358", "#f1a122", "#e74533"]
    return risk_labels[risk_level], risk_level, risk_colors[risk_level], risk_proba

# ==================== API 端點（簡化版）====================

@app.get("/")
def home():
    return {
        "service": "疲勞預測系統",
        "version": "簡化版 v1.0",
        "user_id": SINGLE_USER_ID,
        "endpoints": {
            "app使用": "GET /app_data - App 專用端點",
            "上傳資料": "POST /upload",
            "健康檢查": "GET /health"
        }
    }

# ==================== 🎯 App 專用端點（只返回三個資料）====================
@app.get("/app_data")
def get_app_data():
    """
    App 專用端點
    只返回：user_id, last_update, risk_level
    """
    try:
        df = get_worker_data(limit=100)
        
        # 如果沒有資料
        if df.empty:
            return {
                "user_id": SINGLE_USER_ID,
                "last_update": None,
                "risk_level": "無資料"
            }
        
        # 如果資料不足
        features = extract_features(df)
        if features is None:
            return {
                "user_id": SINGLE_USER_ID,
                "last_update": str(df.iloc[-1]['timestamp']),
                "risk_level": "資料不足"
            }
        
        # 正常情況：預測風險等級
        risk_label, risk_level, risk_color, _ = predict_risk_level(features)
        
        return {
            "user_id": SINGLE_USER_ID,
            "last_update": str(df.iloc[-1]['timestamp']),
            "risk_level": risk_label
        }
        
    except Exception as e:
        raise HTTPException(500, detail=f"查詢失敗: {str(e)}")

# ==================== 上傳資料端點 ====================
@app.post('/upload')
def upload(item: SensorData):
    """
    上傳 MVC 資料
    不需要提供 user_id，系統會自動使用固定的使用者
    """
    ts = item.timestamp or get_taiwan_time().isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO sensor_data (worker_id, timestamp, percent_mvc) VALUES (?, ?, ?)",
        (SINGLE_USER_ID, ts, item.percent_mvc)
    )
    conn.commit()
    conn.close()
    
    print(f"✅ 上傳成功: MVC={item.percent_mvc}%")
    
    return {
        "status": "success",
        "user_id": SINGLE_USER_ID,
        "timestamp": ts,
        "mvc": item.percent_mvc
    }

# ==================== 健康檢查 ====================
@app.get("/healthz")
def healthz():
    # 回用 /health 的資訊也可以：
    # return health()
    return {"status": "ok"}

@app.get('/health')
def health():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM sensor_data WHERE worker_id = ?", (SINGLE_USER_ID,))
    total = c.fetchone()[0]
    conn.close()
    
    return {
        "status": "healthy",
        "user_id": SINGLE_USER_ID,
        "total_records": total,
        "models_loaded": RF_MODEL is not None and LSTM_MODEL is not None,
        "version": "簡化版 v1.0"
    }

# ==================== 清空資料（測試用）====================
@app.delete('/clear')
def clear_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM sensor_data WHERE worker_id = ?", (SINGLE_USER_ID,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    print(f"🗑️ 已刪除 {deleted} 筆資料")
    return {"status": "success", "deleted": deleted}

if __name__ == '__main__':
    print("🚀 疲勞預測系統啟動（簡化版）")
    print(f"👤 使用者: {SINGLE_USER_ID}")
    print("📍 本機: http://localhost:8000")
    print("📍 App 端點: http://localhost:8000/app_data")
    print("\n🎯 啟動命令:")

    print("  uvicorn main:app --reload --host 0.0.0.0 --port 8000")
