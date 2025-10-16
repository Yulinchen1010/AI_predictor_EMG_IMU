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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 設定中文字體
def setup_chinese_font():
    """設定 matplotlib 中文字體"""
    try:
        # 嘗試使用系統中文字體
        fonts = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'STHeiti', 'PingFang TC', 'Noto Sans CJK TC']
        for font in fonts:
            if any(font.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✅ 使用字體: {font}")
                return
        # 如果都找不到，使用 DejaVu Sans（顯示英文）
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("⚠️ 未找到中文字體，使用英文顯示")
    except:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("⚠️ 字體設定失敗，使用預設字體")

setup_chinese_font()

# ==================== 初始設定 ====================
DB_PATH = "fatigue_data.db"
MODEL_PATH = "models/fatigue_regressor.pkl"
os.makedirs("models", exist_ok=True)

# 設定時區：台灣時間 (UTC+8)
TZ_TAIWAN = timezone(timedelta(hours=8))

def get_taiwan_time():
    """取得台灣當前時間"""
    return datetime.now(TZ_TAIWAN)

app = FastAPI(title="疲勞預測系統", version="4.0")

# CORS 設定（允許 APP 連接）
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

# ==================== 載入/訓練模型 ====================
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print("✅ 載入已訓練模型")
        return joblib.load(MODEL_PATH)
    
    print("🔧 訓練新模型...")
    n = 3000
    rng = np.random.RandomState(42)
    percent_mvc = rng.uniform(0, 100, size=n)
    mvc_change = rng.uniform(-20, 30, size=n)  # MVC變化量
    time_elapsed = rng.uniform(0, 240, size=n)  # 經過時間(分鐘)
    
    X = np.vstack([percent_mvc, mvc_change, time_elapsed]).T
    # 風險分數: MVC增加越多，風險越高
    y = mvc_change * 2 + time_elapsed * 0.1 + rng.normal(0, 5, n)
    y = np.clip(y, -20, 100)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, MODEL_PATH)
    print("✅ 模型訓練完成")
    return model

# 初始化模型
MODEL = load_or_train_model()

# ==================== 輔助函數 ====================
def get_worker_data(worker_id: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM sensor_data WHERE worker_id = ? ORDER BY timestamp DESC LIMIT 1000",
        conn, params=(worker_id,)
    )
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def calculate_risk_level(mvc_change: float) -> tuple:
    """根據MVC變化量計算風險等級"""
    if mvc_change >= 30:
        return "高度", 3, "#e74c3c"
    elif mvc_change >= 15:
        return "中度", 2, "#f39c12"
    else:
        return "低度", 1, "#27ae60"

def predict_fatigue_risk(current_mvc: float, initial_mvc: float, time_elapsed: float) -> float:
    """預測未來的MVC變化(風險分數)"""
    mvc_change = current_mvc - initial_mvc
    X = np.array([[current_mvc, mvc_change, time_elapsed]])
    return float(MODEL.predict(X)[0])

# ==================== API 端點 ====================

@app.get("/")
def home():
    return {
        "service": "疲勞預測系統",
        "version": "4.0",
        "description": "基於 MVC 百分比的疲勞風險監測",
        "endpoints": {
            "上傳單筆": "POST /upload",
            "批次上傳": "POST /upload_batch",
            "即時狀態": "GET /status/{worker_id}",
            "預測數據": "GET /predict/{worker_id}",
            "預測圖表": "GET /chart/{worker_id}",
            "所有工作者": "GET /workers",
            "生成測試": "POST /test/generate/{worker_id}?count=10",
            "清空資料": "DELETE /clear/{worker_id}",
            "系統健康": "GET /health",
            "API文件": "GET /docs"
        }
    }

@app.post('/upload')
def upload(item: SensorData):
    """上傳單筆感測器資料 (由 APP 自動上傳)"""
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
    """取得工作者即時狀態"""
    df = get_worker_data(worker_id)
    if df.empty:
        raise HTTPException(404, f"找不到 {worker_id} 的資料")
    
    # 最新數據
    latest = df.iloc[0]
    current_mvc = float(latest['percent_mvc'])
    
    # 初始數據 (第一筆資料)
    initial = df.iloc[-1]
    initial_mvc = float(initial['percent_mvc'])
    
    # 計算MVC變化量
    mvc_change = current_mvc - initial_mvc
    
    # 計算經過時間(分鐘)
    time_diff = (latest['timestamp'] - initial['timestamp']).total_seconds() / 60
    
    # 風險等級評估
    risk_label, risk_level, risk_color = calculate_risk_level(mvc_change)
    
    # 計算平均MVC (最近10筆)
    recent_avg = df.head(10)['percent_mvc'].mean()
    
    return {
        "worker_id": worker_id,
        "current_mvc": round(current_mvc, 2),
        "initial_mvc": round(initial_mvc, 2),
        "mvc_change": round(mvc_change, 2),
        "mvc_change_rate": round(mvc_change / time_diff if time_diff > 0 else 0, 3),
        "fatigue_risk": risk_label,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "time_elapsed_minutes": round(time_diff, 1),
        "recent_avg_mvc": round(recent_avg, 2),
        "last_update": str(latest['timestamp']),
        "data_count": len(df),
        "recommendation": get_recommendation(risk_level, mvc_change)
    }

def get_recommendation(risk_level: int, mvc_change: float) -> str:
    """根據風險等級提供建議"""
    if risk_level == 3:
        return "⚠️ 高風險！建議立即休息 15-20 分鐘"
    elif risk_level == 2:
        return "⚡ 中度風險，建議調整工作姿勢或短暫休息"
    else:
        return "✅ 狀態良好，繼續保持"

@app.get('/predict/{worker_id}')
def predict(worker_id: str, horizon: int = 120):
    """取得未來MVC變化預測 (JSON)"""
    df = get_worker_data(worker_id)
    if df.empty:
        raise HTTPException(404, f"找不到 {worker_id} 的資料")
    
    latest = df.iloc[0]
    initial = df.iloc[-1]
    
    current_mvc = float(latest['percent_mvc'])
    initial_mvc = float(initial['percent_mvc'])
    current_time = (latest['timestamp'] - initial['timestamp']).total_seconds() / 60
    
    # 生成未來預測
    predictions = []
    for t in range(0, min(horizon, 240) + 1, 10):
        future_time = current_time + t
        predicted_risk = predict_fatigue_risk(current_mvc, initial_mvc, future_time)
        predicted_mvc = initial_mvc + predicted_risk
        
        risk_label, risk_level, _ = calculate_risk_level(predicted_risk)
        
        predictions.append({
            "minutes_from_now": t,
            "predicted_mvc": round(predicted_mvc, 2),
            "mvc_change": round(predicted_risk, 2),
            "risk_level": risk_label
        })
    
    return {
        "worker_id": worker_id,
        "current_state": {
            "mvc": current_mvc,
            "initial_mvc": initial_mvc,
            "current_change": round(current_mvc - initial_mvc, 2)
        },
        "predictions": predictions
    }

@app.get('/chart/{worker_id}')
def chart(worker_id: str, horizon: int = 120):
    """取得MVC變化預測曲線圖"""
    df = get_worker_data(worker_id)
    if df.empty:
        raise HTTPException(404, f"找不到 {worker_id} 的資料")
    
    latest = df.iloc[0]
    initial = df.iloc[-1]
    
    current_mvc = float(latest['percent_mvc'])
    initial_mvc = float(initial['percent_mvc'])
    current_time = (latest['timestamp'] - initial['timestamp']).total_seconds() / 60
    
    # 生成預測
    times = np.arange(0, min(horizon, 240) + 1, 5)
    mvc_values = []
    
    for t in times:
        future_time = current_time + t
        predicted_risk = predict_fatigue_risk(current_mvc, initial_mvc, future_time)
        predicted_mvc = initial_mvc + predicted_risk
        mvc_values.append(predicted_mvc)
    
    # 繪圖
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 圖1: MVC絕對值預測
    ax1.plot(times, mvc_values, 'o-', color='#667eea', linewidth=2.5, markersize=5)
    ax1.axhline(initial_mvc, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='初始 MVC')
    ax1.axhline(current_mvc, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='當前 MVC')
    ax1.fill_between(times, initial_mvc + 10, initial_mvc + 20, alpha=0.2, color='orange', label='中度風險區')
    ax1.fill_between(times, initial_mvc + 20, 100, alpha=0.2, color='red', label='高度風險區')
    ax1.set_xlabel('未來分鐘數', fontsize=12, fontweight='bold')
    ax1.set_ylabel('預測 MVC (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'MVC 百分比預測 - {worker_id}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    ax1.set_ylim([0, 100])
    
    # 圖2: MVC變化量
    mvc_changes = np.array(mvc_values) - initial_mvc
    ax2.plot(times, mvc_changes, 'o-', color='#f5576c', linewidth=2.5, markersize=5)
    ax2.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax2.axhline(10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='中度風險 (+10%)')
    ax2.axhline(20, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='高度風險 (+20%)')
    ax2.fill_between(times, 10, 20, alpha=0.2, color='orange')
    ax2.fill_between(times, 20, 100, alpha=0.2, color='red')
    ax2.set_xlabel('未來分鐘數', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MVC 變化量 (%)', fontsize=12, fontweight='bold')
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
    
    # 計算每個工作者的風險狀態
    workers = []
    for _, row in df.iterrows():
        mvc_range = row['max_mvc'] - row['min_mvc']
        risk_label, risk_level, risk_color = calculate_risk_level(mvc_range)
        
        workers.append({
            "worker_id": row['worker_id'],
            "count": int(row['count']),
            "last_update": row['last_update'],
            "avg_mvc": round(row['avg_mvc'], 2),
            "min_mvc": round(row['min_mvc'], 2),
            "max_mvc": round(row['max_mvc'], 2),
            "mvc_range": round(mvc_range, 2),
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
        "deleted": deleted,
        "message": f"已成功刪除 {worker_id} 的所有記錄"
    }

@app.delete('/clear_all')
def clear_all():
    """清空所有資料（慎用）"""
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
        "deleted": total,
        "message": f"已清空所有資料，共刪除 {total} 筆記錄"
    }

@app.post('/test/generate/{worker_id}')
def generate_test(worker_id: str, count: int = 20):
    """生成測試資料 (模擬每分鐘上傳)"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 從30%開始，隨機遞增模擬疲勞累積
    base_mvc = 30.0
    
    for i in range(count):
        ts = (datetime.utcnow() - timedelta(minutes=count-i)).isoformat()
        # MVC隨時間增加，模擬疲勞累積
        mvc = base_mvc + i * np.random.uniform(0.5, 2.0) + np.random.normal(0, 2)
        mvc = np.clip(mvc, 0, 100)
        
        c.execute(
            "INSERT INTO sensor_data (worker_id, timestamp, percent_mvc) VALUES (?, ?, ?)",
            (worker_id, ts, float(mvc))
        )
    
    conn.commit()
    conn.close()
    
    print(f"✅ 已生成 {count} 筆測試資料 (模擬每分鐘上傳)")
    
    return {
        "status": "success",
        "worker_id": worker_id,
        "generated": count,
        "message": f"已生成 {count} 筆測試資料，模擬 {count} 分鐘的工作記錄"
    }

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
        "model_loaded": MODEL is not None,
        "total_records": total,
        "total_workers": workers,
        "database": DB_PATH,
        "version": "4.0"
    }

if __name__ == '__main__':
    print("🚀 疲勞預測系統已啟動 (v4.0)")
    print("📍 本機: http://localhost:8000")
    print("📍 文件: http://localhost:8000/docs")
    print("💡 系統特點: 基於 MVC 變化量的疲勞風險評估")
    print("⏱️  建議 APP 每分鐘自動上傳 MVC 數據")
    print("\n啟動命令: uvicorn main:app --reload --host 0.0.0.0 --port 8000")