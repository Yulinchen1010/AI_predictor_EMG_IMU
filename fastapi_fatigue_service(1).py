from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional, List
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
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# 終端機顯示設定
import sys
from colorama import init, Fore, Back, Style
try:
    init(autoreset=True)
    COLORS_ENABLED = True
except:
    COLORS_ENABLED = False
    print("⚠️ colorama 未安裝，使用純文字顯示 (可選: pip install colorama)")

# ==================== 初始設定 ====================
DB_PATH = "fatigue_data.db"
RF_MODEL_PATH = "models/rf_classifier.pkl"
LSTM_MODEL_PATH = "models/lstm_predictor.h5"
SCALER_PATH = "models/scaler.pkl"
os.makedirs("models", exist_ok=True)

# 設定時區：台灣時間 (UTC+8)
TZ_TAIWAN = timezone(timedelta(hours=8))

def get_taiwan_time():
    """取得台灣當前時間"""
    return datetime.now(TZ_TAIWAN)

app = FastAPI(title="疲勞預測系統 - RF + LSTM")

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
    worker_id: str
    percent_mvc: float = Field(ge=0, le=100)
    timestamp: Optional[str] = None

class BatchUpload(BaseModel):
    data: List[SensorData]

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

# ==================== 模型訓練函數 ====================
def train_rf_classifier():
    """訓練 RandomForest 風險分類器"""
    print("訓練 RandomForest 分類器...")
    
    # 生成訓練資料
    n = 5000
    rng = np.random.RandomState(42)
    
    # 特徵：當前MVC、累積變化量、變化速率、平均MVC、標準差
    current_mvc = rng.uniform(20, 95, size=n)
    total_change = rng.uniform(-10, 60, size=n)
    change_rate = rng.uniform(-1.0, 3.0, size=n)
    avg_mvc = rng.uniform(25, 85, size=n)
    std_mvc = rng.uniform(0.5, 15, size=n)
    
    X = np.vstack([current_mvc, total_change, change_rate, avg_mvc, std_mvc]).T
    
    # 目標：風險等級 (0: 低度, 1: 中度, 2: 高度)
    y = np.zeros(n, dtype=int)
    y[(total_change >= 20) & (total_change < 40)] = 1
    y[total_change >= 40] = 2
    # 考慮變化速率的影響
    y[(change_rate > 2.0) & (y < 2)] = 2
    y[(change_rate > 1.2) & (y < 1)] = 1
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    score = model.score(X_val, y_val)
    print(f"RandomForest 分類器訓練完成 (準確率 = {score:.3f})")
    
    joblib.dump(model, RF_MODEL_PATH)
    return model

def train_lstm_predictor():
    """訓練 LSTM 時間序列預測器"""
    print("訓練 LSTM 預測器...")
    
    # 生成訓練資料
    n_sequences = 1000
    seq_length = 20
    pred_length = 12  # 預測未來 12 個時間點
    
    X_train_list = []
    y_train_list = []
    
    rng = np.random.RandomState(42)
    
    for _ in range(n_sequences):
        # 生成一個時間序列
        base = rng.uniform(25, 45)
        trend = rng.uniform(0.1, 2.0)
        noise = rng.normal(0, 2, seq_length + pred_length)
        
        sequence = base + np.arange(seq_length + pred_length) * trend + noise
        sequence = np.clip(sequence, 0, 100)
        
        X_train_list.append(sequence[:seq_length].reshape(-1, 1))
        y_train_list.append(sequence[seq_length:seq_length + pred_length])
    
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    
    # 建立 LSTM 模型
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(32, activation='relu'),
        Dropout(0.2),
        Dense(pred_length)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    print(f"LSTM 預測器訓練完成 (MAE = {history.history['mae'][-1]:.3f})")
    
    model.save(LSTM_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    return model, scaler

# ==================== 載入/訓練模型 ====================
def load_or_train_models():
    """載入或訓練模型"""
    # RandomForest 分類器
    if os.path.exists(RF_MODEL_PATH):
        print("✅ 載入 RandomForest 分類器")
        rf_model = joblib.load(RF_MODEL_PATH)
    else:
        rf_model = train_rf_classifier()
    
    # LSTM 預測器
    if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("✅ 載入 LSTM 預測器")
        lstm_model = load_model(LSTM_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        lstm_model, scaler = train_lstm_predictor()
    
    return rf_model, lstm_model, scaler

# 初始化模型
RF_MODEL, LSTM_MODEL, SCALER = load_or_train_models()

# ==================== 輔助函數 ====================
def get_worker_data(worker_id: str, limit: int = 1000) -> pd.DataFrame:
    """取得工作者資料"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM sensor_data WHERE worker_id = ? ORDER BY timestamp DESC LIMIT {limit}",
        conn, params=(worker_id,)
    )
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def extract_features(df: pd.DataFrame) -> np.ndarray:
    """從資料中提取特徵用於 RF 分類"""
    if len(df) < 2:
        return None
    
    initial_mvc = df.iloc[-1]['percent_mvc']
    current_mvc = df.iloc[0]['percent_mvc']
    total_change = initial_mvc - current_mvc
    
    # 計算變化速率（最近10筆）
    recent_df = df.tail(min(10, len(df)))
    if len(recent_df) >= 2:
        time_diff = (recent_df.iloc[-1]['timestamp'] - recent_df.iloc[0]['timestamp']).total_seconds() / 60
        if time_diff > 0:
            mvc_diff = recent_df.iloc[-1]['percent_mvc'] - recent_df.iloc[0]['percent_mvc']
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
    """使用 RF 預測風險等級"""
    risk_level = int(RF_MODEL.predict(features)[0])
    risk_proba = RF_MODEL.predict_proba(features)[0]
    
    risk_labels = ["低度", "中度", "高度"]
    risk_colors = ["#18b358", "#f1a122", "#e74533"]
    
    return risk_labels[risk_level], risk_level, risk_colors[risk_level], risk_proba

def predict_future_mvc(df: pd.DataFrame, horizon: int = 12) -> np.ndarray:
    """使用 LSTM 預測未來 MVC 值"""
    seq_length = 20
    
    if len(df) < seq_length:
        # 資料不足，使用簡單線性外推
        return simple_extrapolation(df, horizon)
    
    # 取最近的序列
    recent_mvc = df.tail(seq_length)['percent_mvc'].values.reshape(-1, 1)
    
    # 標準化
    recent_scaled = SCALER.transform(recent_mvc)
    
    # 預測
    X_pred = recent_scaled.reshape(1, seq_length, 1)
    predictions = LSTM_MODEL.predict(X_pred, verbose=0)[0]
    
    # 限制範圍
    predictions = np.clip(predictions[:horizon], 0, 100)
    
    return predictions

def simple_extrapolation(df: pd.DataFrame, horizon: int) -> np.ndarray:
    """簡單線性外推（資料不足時使用）"""
    if len(df) < 2:
        return np.full(horizon, df.iloc[-1]['percent_mvc'])
    
    recent = df.tail(min(10, len(df)))
    mvc_values = recent['percent_mvc'].values
    
    # 計算平均變化率
    changes = np.diff(mvc_values)
    avg_change = np.mean(changes) if len(changes) > 0 else 0
    
    current_mvc = mvc_values[0]
    predictions = []
    
    for i in range(1, horizon + 1):
        pred = current_mvc + avg_change * i
        predictions.append(np.clip(pred, 0, 100))
    
    return np.array(predictions)

def get_recommendation(risk_level: int, change_rate: float) -> str:
    """根據風險等級和變化趨勢提供建議"""
    if risk_level == 2 or change_rate > 2.0:
        return "⚠️ 高風險！建議立即休息 15-20 分鐘"
    elif risk_level == 1 or change_rate > 1.2:
        return "⚡ 中度風險，建議調整工作姿勢或短暫休息"
    elif change_rate < 0:
        return "✅ 狀態改善中，繼續保持"
    else:
        return "✅ 狀態良好，繼續保持"

# ==================== API 端點 ====================

@app.get("/")
def home():
    return {
        "service": "疲勞預測系統 v5.0 - RF + LSTM",
        "description": "使用 RandomForest 分類風險等級，LSTM 預測時間序列",
        "features": [
            "✅ RandomForest 多類別風險分類",
            "✅ LSTM 時間序列預測",
            "✅ 實時風險評估與預測",
            "✅ 視覺化預測曲線"
        ],
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
            "系統健康": "GET /health"
        }
    }

@app.post('/upload')
def upload(item: SensorData):
    """上傳單筆感測器資料"""
    ts = item.timestamp or datetime.utcnow().isoformat()
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO sensor_data (worker_id, timestamp, percent_mvc) VALUES (?, ?, ?)",
        (item.worker_id, ts, item.percent_mvc)
    )
    conn.commit()
    conn.close()
    
    print(f"✅ 上傳成功: {item.worker_id} | MVC={item.percent_mvc}% | 時間={ts}")
    
    return {
        "status": "success",
        "worker_id": item.worker_id,
        "timestamp": ts,
        "mvc": item.percent_mvc
    }

@app.post('/upload_batch')
def upload_batch(batch: BatchUpload):
    """批次上傳多筆資料"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    for item in batch.data:
        ts = item.timestamp or datetime.utcnow().isoformat()
        c.execute(
            "INSERT INTO sensor_data (worker_id, timestamp, percent_mvc) VALUES (?, ?, ?)",
            (item.worker_id, ts, item.percent_mvc)
        )
    
    conn.commit()
    conn.close()
    
    print(f"✅ 批次上傳成功: {len(batch.data)} 筆")
    
    return {
        "status": "success",
        "uploaded": len(batch.data)
    }

@app.get('/status/{worker_id}')
def get_status(worker_id: str):
    """取得工作者即時狀態（使用 RF 分類）"""
    df = get_worker_data(worker_id)
    if df.empty:
        raise HTTPException(404, f"找不到 {worker_id} 的資料")
    
    # 提取特徵
    features = extract_features(df)
    if features is None:
        raise HTTPException(400, "資料不足，無法進行分析")
    
    # RF 分類預測
    risk_label, risk_level, risk_color, risk_proba = predict_risk_level(features)
    
    # 基本統計
    latest = df.iloc[-1]
    initial = df.iloc[0]
    current_mvc = float(latest['percent_mvc'])
    initial_mvc = float(initial['percent_mvc'])
    total_change = current_mvc - initial_mvc
    
    time_diff = (latest['timestamp'] - initial['timestamp']).total_seconds() / 60
    
    # 變化速率
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
            "高度": round(float(risk_proba[2]), 3)
        },
        "time_elapsed_minutes": round(time_diff, 1),
        "recent_avg_mvc": round(df.tail(10)['percent_mvc'].mean(), 2),
        "last_update": str(latest['timestamp']),
        "data_count": len(df),
        "trend": "加速惡化" if change_rate > 1.0 else "緩慢增加" if change_rate > 0.3 else "穩定" if change_rate > -0.3 else "改善中",
        "recommendation": get_recommendation(risk_level, change_rate)
    }

@app.get('/predict/{worker_id}')
def predict(worker_id: str, horizon: int = 12):
    """取得未來MVC變化預測（使用 LSTM）"""
    df = get_worker_data(worker_id)
    if df.empty:
        raise HTTPException(404, f"找不到 {worker_id} 的資料")
    
    # LSTM 預測
    predictions = predict_future_mvc(df, horizon)
    
    # 當前狀態
    latest = df.iloc[-1]
    initial = df.iloc[0]
    current_mvc = float(latest['percent_mvc'])
    initial_mvc = float(initial['percent_mvc'])
    
    features = extract_features(df)
    change_rate = float(features[0][2])
    
    # 生成預測結果
    results = []
    time_interval = 5  # 假設每個預測點間隔5分鐘
    
    for i, pred_mvc in enumerate(predictions):
        minutes = (i + 1) * time_interval
        total_change = initial_mvc - pred_mvc 
        
        # 預測該時刻的風險等級
        pred_features = np.array([[pred_mvc, total_change, change_rate, 
                                  df['percent_mvc'].mean(), df['percent_mvc'].std()]])
        risk_label, risk_level, _, _ = predict_risk_level(pred_features)
        
        results.append({
            "minutes_from_now": minutes,
            "predicted_mvc": round(float(pred_mvc), 2),
            "predicted_total_change": round(float(total_change), 2),
            "risk_level": risk_label
        })
    
    return {
        "worker_id": worker_id,
        "current_state": {
            "mvc": current_mvc,
            "initial_mvc": initial_mvc,
            "total_change": round(initial_mvc-current_mvc, 2),
            "change_rate": round(change_rate, 3),
            "trend": "上升" if change_rate > 0.1 else "下降" if change_rate < -0.1 else "穩定"
        },
        "predictions": results
    }

@app.get('/chart/{worker_id}')
def chart(worker_id: str, horizon: int = 12):
    """取得MVC變化預測曲線圖"""
    df = get_worker_data(worker_id)
    if df.empty:
        raise HTTPException(404, f"找不到 {worker_id} 的資料")
    
    # LSTM 預測
    predictions = predict_future_mvc(df, horizon)
    
    # 當前狀態
    latest = df.iloc[-1]
    initial = df.iloc[0]
    current_mvc = float(latest['percent_mvc'])
    initial_mvc = float(initial['percent_mvc'])
    
    features = extract_features(df)
    change_rate = float(features[0][2])
    
    # 繪圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 歷史資料
    history_minutes = np.arange(-len(df), 0) * 5
    history_mvc = df['percent_mvc'].values
    
    # 預測資料
    time_interval = 5
    pred_minutes = np.arange(1, len(predictions) + 1) * time_interval
    
    # 圖1: MVC絕對值
    ax1.plot(history_minutes, history_mvc, 'o-', color='#3498db', linewidth=2, 
             markersize=4, label='歷史 MVC', alpha=0.7)
    ax1.plot(pred_minutes, predictions, 's-', color='#e74c3c', linewidth=2.5, 
             markersize=5, label='LSTM 預測')
    ax1.axhline(initial_mvc, color='green', linestyle='--', linewidth=1.5, 
                alpha=0.7, label=f'初始 MVC ({initial_mvc:.1f}%)')
    ax1.axvline(0, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    
    # 風險區域
    ax1.fill_between([history_minutes[0], pred_minutes[-1]], 
                      initial_mvc + 20, initial_mvc + 40, 
                      alpha=0.15, color='orange', label='中度風險區')
    ax1.fill_between([history_minutes[0], pred_minutes[-1]], 
                      initial_mvc + 40, 100, 
                      alpha=0.15, color='red', label='高度風險區')
    
    ax1.set_xlabel('時間 (分鐘)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MVC (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'MVC 預測 - {worker_id} | 變化率: {change_rate:.2f}%/min | LSTM預測', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_ylim([0, 100])
    
    # 圖2: MVC變化量
    history_changes = history_mvc - initial_mvc
    pred_changes = predictions - initial_mvc
    
    ax2.plot(history_minutes, history_changes, 'o-', color='#3498db', 
             linewidth=2, markersize=4, label='歷史變化', alpha=0.7)
    ax2.plot(pred_minutes, pred_changes, 's-', color='#e74c3c', 
             linewidth=2.5, markersize=5, label='預測變化')
    ax2.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axhline(20, color='orange', linestyle='--', linewidth=1.5, 
                alpha=0.7, label='中度風險閾值')
    ax2.axhline(40, color='red', linestyle='--', linewidth=1.5, 
                alpha=0.7, label='高度風險閾值')
    ax2.axvline(0, color='gray', linestyle=':', linewidth=2, alpha=0.5)
    
    # 風險區域
    ax2.fill_between([history_minutes[0], pred_minutes[-1]], 
                      20, 40, alpha=0.15, color='orange')
    ax2.fill_between([history_minutes[0], pred_minutes[-1]], 
                      40, 100, alpha=0.15, color='red')
    
    ax2.set_xlabel('時間 (分鐘)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MVC 累積變化量 (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'MVC 變化量預測 - {worker_id}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=120)
    plt.close()
    buf.seek(0)
    
    return Response(content=buf.getvalue(), media_type='image/png')

@app.get('/workers')
def list_workers():
    """列出所有工作者"""
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
        worker_df = get_worker_data(row['worker_id'], limit=100)
        
        if len(worker_df) >= 2:
            features = extract_features(worker_df)
            risk_label, risk_level, risk_color, _ = predict_risk_level(features)
        else:
            risk_label = "資料不足"
            risk_color = "#95a5a6"
        
        workers.append({
            "worker_id": row['worker_id'],
            "count": int(row['count']),
            "last_update": row['last_update'],
            "avg_mvc": round(row['avg_mvc'], 2),
            "min_mvc": round(row['min_mvc'], 2),
            "max_mvc": round(row['max_mvc'], 2),
            "mvc_range": round(row['max_mvc'] - row['min_mvc'], 2),
            "risk_level": risk_label,
            "risk_color": risk_color
        })
    
    return {"workers": workers, "total": len(workers)}

@app.delete('/clear/{worker_id}')
def clear(worker_id: str):
    """清空工作者資料"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM sensor_data WHERE worker_id = ?", (worker_id,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    
    print(f"🗑️ 已刪除 {worker_id} 的 {deleted} 筆資料")
    
    return {
        "status": "success",
        "worker_id": worker_id,
        "deleted": deleted
    }

@app.delete('/clear_all')
def clear_all():
    """清空所有資料"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM sensor_data")
    total = c.fetchone()[0]
    c.execute("DELETE FROM sensor_data")
    conn.commit()
    conn.close()
    
    print(f"🗑️ 已清空資料庫，刪除 {total} 筆資料")
    
    return {
        "status": "success",
        "deleted": total
    }

@app.post('/retrain')
def retrain_models():
    """重新訓練模型"""
    global RF_MODEL, LSTM_MODEL, SCALER
    
    try:
        print("🔄 開始重新訓練模型...")
        
        # 訓練 RandomForest
        RF_MODEL = train_rf_classifier()
        
        # 訓練 LSTM
        LSTM_MODEL, SCALER = train_lstm_predictor()
        
        return {
            "status": "success",
            "message": "模型重新訓練完成",
            "models": ["RandomForest Classifier", "LSTM Predictor"]
        }
    except Exception as e:
        raise HTTPException(500, f"訓練失敗: {str(e)}")

@app.get('/health')
def health():
    """系統健康檢查"""
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
        "version": "5.0 - RF + LSTM"
    }

if __name__ == '__main__':
    print("🚀 疲勞預測系統已啟動")
    print("📍 本機: http://localhost:8000")
    print("📍 文件: http://localhost:8000/docs")
    print("\n✨ v5.0 架構:")
    print("  ✅ RandomForest: 多類別風險分類 (低/中/高度)")
    print("  ✅ LSTM: 時間序列預測未來 MVC 趨勢")
    print("  ✅ 即時風險評估與機率輸出")
    print("  ✅ 完整的歷史+預測視覺化")
    print("\n⚙️ 依賴套件:")
    print("  pip install fastapi uvicorn pandas numpy scikit-learn")
    print("  pip install tensorflow matplotlib joblib")
    print("\n🎯 啟動命令:")
    print("  uvicorn main:app --reload --host 0.0.0.0 --port 8000")