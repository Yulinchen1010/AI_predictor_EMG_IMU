from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

# ==================== 預測參數設定 ====================
TIME_INTERVAL_MINUTES = 1  # 每筆資料的時間間隔（分鐘）
HISTORY_SEQUENCE_LENGTH = 60  # LSTM 使用的歷史序列長度（筆數）
HISTORY_TIME_WINDOW = HISTORY_SEQUENCE_LENGTH * TIME_INTERVAL_MINUTES  # 歷史時間窗口（分鐘）
PREDICTION_HORIZON = 30  # 預測未來的時間點數量（筆數）
PREDICTION_TIME_WINDOW = PREDICTION_HORIZON * TIME_INTERVAL_MINUTES  # 預測時間窗口（分鐘）
MAX_DATA_LIMIT = 1000  # 最多查詢的資料筆數

print(f"⚙️ 預測參數設定:")
print(f"  📊 時間間隔: {TIME_INTERVAL_MINUTES} 分鐘/筆")
print(f"  📈 歷史資料: 最近 {HISTORY_SEQUENCE_LENGTH} 筆 (約 {HISTORY_TIME_WINDOW} 分鐘)")
print(f"  🔮 預測範圍: 未來 {PREDICTION_HORIZON} 筆 (約 {PREDICTION_TIME_WINDOW} 分鐘)")

# 設定時區：台灣時間 (UTC+8)
TZ_TAIWAN = timezone(timedelta(hours=8))

def get_taiwan_time():
    return datetime.now(TZ_TAIWAN)

app = FastAPI(title="疲勞預測系統")

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
    print(f"RandomForest 分類器訓練完成 (準確率 = {score:.3f})")
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
    print(f"LSTM 預測器訓練完成 (MAE = {final_mae:.3f})")
    
    model.save(LSTM_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return model, scaler

def load_or_train_models():
    if os.path.exists(RF_MODEL_PATH):
        print("✅ 載入 RandomForest 分類器")
        rf_model = joblib.load(RF_MODEL_PATH)
    else:
        rf_model = train_rf_classifier()
    
    if os.path.exists(LSTM_MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("✅ 載入 LSTM 預測器")
        lstm_model = load_model(LSTM_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        lstm_model, scaler = train_lstm_predictor()
    
    return rf_model, lstm_model, scaler

RF_MODEL, LSTM_MODEL, SCALER = load_or_train_models()

# ==================== 輔助函數 ====================
def get_worker_data(worker_id: str, limit: int = None) -> pd.DataFrame:
    if limit is None:
        limit = MAX_DATA_LIMIT
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM sensor_data WHERE worker_id = ? ORDER BY timestamp DESC LIMIT {limit}",
        conn, params=(worker_id,)
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

def predict_future_mvc(df: pd.DataFrame, horizon: int = None) -> np.ndarray:
    if horizon is None:
        horizon = PREDICTION_HORIZON
    
    seq_length = HISTORY_SEQUENCE_LENGTH
    
    if len(df) < seq_length:
        return simple_extrapolation(df, horizon)
    
    recent_mvc = df.tail(seq_length)['percent_mvc'].values.reshape(-1, 1)
    recent_scaled = SCALER.transform(recent_mvc)
    X_pred = recent_scaled.reshape(1, seq_length, 1)
    predictions = LSTM_MODEL.predict(X_pred, verbose=0)[0]
    predictions = np.clip(predictions[:horizon], 0, 100)
    return predictions

def simple_extrapolation(df: pd.DataFrame, horizon: int) -> np.ndarray:
    if len(df) < 2:
        return np.full(horizon, df.iloc[-1]['percent_mvc'])
    
    recent = df.tail(min(10, len(df)))
    mvc_values = recent['percent_mvc'].values
    changes = np.diff(mvc_values)
    avg_change = np.mean(changes) if len(changes) > 0 else 0
    current_mvc = mvc_values[-1]
    predictions = []
    
    for i in range(1, horizon + 1):
        pred = current_mvc + avg_change * i
        predictions.append(np.clip(pred, 0, 100))
    
    return np.array(predictions)

def calculate_decline_rate(predictions: np.ndarray, time_interval: int = None) -> dict:
    if time_interval is None:
        time_interval = TIME_INTERVAL_MINUTES
    
    decline_rates = []
    for i in range(len(predictions) - 1):
        rate = (predictions[i] - predictions[i + 1]) / time_interval
        decline_rates.append(rate)
    
    accelerations = []
    if len(decline_rates) >= 2:
        for i in range(len(decline_rates) - 1):
            accel = decline_rates[i + 1] - decline_rates[i]
            accelerations.append(accel)
    
    avg_rate = np.mean(decline_rates) if decline_rates else 0
    avg_accel = np.mean(accelerations) if accelerations else 0
    
    if avg_accel > 0.05:
        trend = "加速惡化"
        trend_warning = "⚠️ 疲勞正在加速累積"
    elif avg_accel < -0.05:
        trend = "減速改善"
        trend_warning = "✅ 疲勞累積速度緩慢"
    else:
        if avg_rate > 0.5:
            trend = "穩定惡化"
            trend_warning = " 疲勞持續累積中"
        elif avg_rate < -0.1:
            trend = "穩定恢復"
            trend_warning = " 持續恢復中"
        else:
            trend = "維持穩定"
            trend_warning = " 狀態穩定"
    
    return {
        "decline_rates": decline_rates,
        "accelerations": accelerations,
        "avg_decline_rate": round(float(avg_rate), 3),
        "avg_acceleration": round(float(avg_accel), 4),
        "trend": trend,
        "trend_warning": trend_warning
    }

def find_critical_timepoints(predictions: np.ndarray, initial_mvc: float, 
                             time_interval: int = None) -> dict:
    if time_interval is None:
        time_interval = TIME_INTERVAL_MINUTES
    
    warnings = []
    moderate_threshold = 20
    high_threshold = 40
    moderate_time = None
    high_time = None
    
    for i, pred_mvc in enumerate(predictions):
        minutes = (i + 1) * time_interval
        total_change = initial_mvc - pred_mvc
        
        if moderate_time is None and total_change >= moderate_threshold:
            moderate_time = minutes
            warnings.append({
                "time_minutes": minutes,
                "risk_level": "中度",
                "predicted_mvc": round(float(pred_mvc), 2),
                "total_change": round(float(total_change), 2),
                "warning": f"⚡ 預計 {minutes} 分鐘後達到中度風險"
            })
        
        if high_time is None and total_change >= high_threshold:
            high_time = minutes
            warnings.append({
                "time_minutes": minutes,
                "risk_level": "高度",
                "predicted_mvc": round(float(pred_mvc), 2),
                "total_change": round(float(total_change), 2),
                "warning": f"⚠️ 預計 {minutes} 分鐘後達到高度風險"
            })
    
    if not warnings:
        return {
            "has_warnings": False,
            "message": "✅ 預測期間內風險可控",
            "warnings": []
        }
    
    return {
        "has_warnings": True,
        "moderate_risk_at": moderate_time,
        "high_risk_at": high_time,
        "warnings": warnings
    }

def get_recommendation(risk_level: int, change_rate: float) -> str:
    if risk_level == 2 or change_rate > 2.0:
        return "⚠️ 高風險！建議立即休息 15-20 分鐘"
    elif risk_level == 1 or change_rate > 1.2:
        return "⚡ 中度風險，建議調整工作姿勢或短暫休息"
    elif change_rate < -0.3:
        return "✅ 狀態改善中，繼續保持"
    else:
        return "✅ 狀態良好，繼續保持"

# ==================== API 端點 ====================

@app.get("/")
def home():
    return {
        "service": "疲勞預測系統 v5.1 - RF + LSTM",
        "endpoints": {
            "上傳單筆": "POST /upload",
            "批次上傳": "POST /upload_batch",
            "即時狀態": "GET /status/{worker_id}",
            "預測數據": "GET /predict/{worker_id}",
            "所有工作者": "GET /workers",
            "清空資料": "DELETE /clear/{worker_id}",
            "清空所有": "DELETE /clear_all",
            "重訓模型": "POST /retrain",
            "系統健康": "GET /health"
        }
    }

@app.post('/upload')
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
    
    print(f"✅ 上傳成功: {item.worker_id} | MVC={item.percent_mvc}%")
    
    return {"status": "success", "worker_id": item.worker_id, "timestamp": ts, "mvc": item.percent_mvc}

@app.post('/upload_batch')
def upload_batch(batch: BatchUpload):
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
    return {"status": "success", "uploaded": len(batch.data)}

@app.get('/status/{worker_id}')
def get_status(worker_id: str):
    df = get_worker_data(worker_id)
    if df.empty:
        raise HTTPException(404, f"找不到 {worker_id} 的資料")
    
    features = extract_features(df)
    if features is None:
        raise HTTPException(400, "資料不足，無法進行分析")
    
    risk_label, risk_level, risk_color, risk_proba = predict_risk_level(features)
    
    initial = df.iloc[0]
    latest = df.iloc[-1]
    initial_mvc = float(initial['percent_mvc'])
    current_mvc = float(latest['percent_mvc'])
    total_change = initial_mvc - current_mvc
    time_diff = (latest['timestamp'] - initial['timestamp']).total_seconds() / 60
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
        "trend": "加速惡化" if change_rate > 1.0 else "緩慢惡化" if change_rate > 0.3 else "穩定" if change_rate > -0.3 else "改善中",
        "recommendation": get_recommendation(risk_level, change_rate)
    }

@app.get('/predict/{worker_id}')
def predict(worker_id: str, horizon: int = None):
    if horizon is None:
        horizon = PREDICTION_HORIZON
    
    df = get_worker_data(worker_id)
    if df.empty:
        raise HTTPException(404, f"找不到 {worker_id} 的資料")
    
    predictions = predict_future_mvc(df, horizon)
    
    initial = df.iloc[0]
    latest = df.iloc[-1]
    initial_mvc = float(initial['percent_mvc'])
    current_mvc = float(latest['percent_mvc'])
    
    features = extract_features(df)
    current_change_rate = float(features[0][2])
    
    rate_analysis = calculate_decline_rate(predictions)
    critical_points = find_critical_timepoints(predictions, initial_mvc)
    
    detailed_predictions = []
    
    for i, pred_mvc in enumerate(predictions):
        minutes = (i + 1) * TIME_INTERVAL_MINUTES
        total_change = initial_mvc - pred_mvc
        
        if i < len(rate_analysis['decline_rates']):
            point_decline_rate = rate_analysis['decline_rates'][i]
        else:
            point_decline_rate = rate_analysis['decline_rates'][-1] if rate_analysis['decline_rates'] else 0
        
        pred_features = np.array([[pred_mvc, total_change, point_decline_rate, 
                                  df['percent_mvc'].mean(), df['percent_mvc'].std()]])
        risk_label, risk_level, risk_color, _ = predict_risk_level(pred_features)
        
        detailed_predictions.append({
            "minutes_from_now": minutes,
            "predicted_mvc": round(float(pred_mvc), 2),
            "mvc_decline_from_initial": round(float(total_change), 2),
            "decline_rate_per_min": round(float(point_decline_rate), 3),
            "risk_level": risk_label,
            "risk_color": risk_color
        })
    
    current_features = extract_features(df)
    current_risk_label, current_risk_level, current_risk_color, risk_proba = predict_risk_level(current_features)
    
    actual_history_used = min(len(df), HISTORY_SEQUENCE_LENGTH)
    actual_history_minutes = actual_history_used * TIME_INTERVAL_MINUTES
    
    return {
        "worker_id": worker_id,
        "timestamp": str(latest['timestamp']),
        "prediction_config": {
            "time_interval_minutes": TIME_INTERVAL_MINUTES,
            "history_data_points": actual_history_used,
            "history_time_window_minutes": actual_history_minutes,
            "prediction_horizon_points": len(predictions),
            "prediction_time_window_minutes": len(predictions) * TIME_INTERVAL_MINUTES
        },
        "current_state": {
            "mvc": round(current_mvc, 2),
            "initial_mvc": round(initial_mvc, 2),
            "total_change": round(initial_mvc - current_mvc, 2),
            "current_decline_rate": round(current_change_rate, 3),
            "risk_level": current_risk_label,
            "risk_color": current_risk_color,
            "risk_probabilities": {
                "低度": round(float(risk_proba[0]), 3),
                "中度": round(float(risk_proba[1]), 3),
                "高度": round(float(risk_proba[2]), 3)
            }
        },
        "trend_analysis": {
            "avg_decline_rate": rate_analysis['avg_decline_rate'],
            "avg_acceleration": rate_analysis['avg_acceleration'],
            "trend": rate_analysis['trend'],
            "warning": rate_analysis['trend_warning']
        },
        "critical_timepoints": critical_points,
        "predictions": detailed_predictions,
        "recommendation": get_recommendation(current_risk_level, current_change_rate),
        "prediction_summary": {
            "total_predictions": len(detailed_predictions),
            "time_horizon_minutes": horizon * TIME_INTERVAL_MINUTES,
            "final_predicted_mvc": round(float(predictions[-1]), 2),
            "total_predicted_decline": round(float(initial_mvc - predictions[-1]), 2)
        }
    }

@app.get('/workers')
def list_workers():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT worker_id, COUNT(*) as count, MAX(timestamp) as last_update,
           AVG(percent_mvc) as avg_mvc, MIN(percent_mvc) as min_mvc, MAX(percent_mvc) as max_mvc
           FROM sensor_data GROUP BY worker_id""", conn)
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM sensor_data WHERE worker_id = ?", (worker_id,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    print(f"🗑️ 已刪除 {worker_id} 的 {deleted} 筆資料")
    return {"status": "success", "worker_id": worker_id, "deleted": deleted}

@app.delete('/clear_all')
def clear_all():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM sensor_data")
    total = c.fetchone()[0]
    c.execute("DELETE FROM sensor_data")
    conn.commit()
    conn.close()
    print(f"🗑️ 已清空資料庫，刪除 {total} 筆資料")
    return {"status": "success", "deleted": total}

@app.post('/retrain')
def retrain_models():
    global RF_MODEL, LSTM_MODEL, SCALER
    
    try:
        print("🔄 開始重新訓練模型...")
        RF_MODEL = train_rf_classifier()
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
        "version": "5.1 - Enhanced Prediction"
    }

if __name__ == '__main__':
    print("🚀 疲勞預測系統 v5.1 已啟動")
    print("📍 本機: http://localhost:8000")
    print("📍 文件: http://localhost:8000/docs")
    print("\n✨ 新功能:")
    print("  ✅ 下降速率分析: 每個時間區間的 MVC 變化速率")
    print("  ✅ 趨勢加速度判斷: 疲勞是加速還是減速")
    print("  ✅ 關鍵時間點預警: 預測何時達到風險閾值")
    print("  ✅ 全域參數統一設定: 方便調整時間參數")
    print("\n🎯 啟動命令:")
    print("  uvicorn main:app --reload --host 0.0.0.0 --port 8000")
