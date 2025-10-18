import requests
import random
from datetime import datetime, timedelta
import time

API_URL = "http://localhost:8000"

# 終端機顏色輸出（可選）
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORS_ENABLED = True
except:
    COLORS_ENABLED = False


def print_colored(text: str, color: str = "white"):
    """彩色終端機輸出"""
    if not COLORS_ENABLED:
        print(text)
        return
    
    color_map = {
        "red": Fore.RED,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "blue": Fore.BLUE,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
    }
    print(color_map.get(color, Fore.WHITE) + text + Style.RESET_ALL)


# ✅ 新增：確保 MVC 單調遞減
def ensure_monotonic_decrease(data):
    """確保 MVC 單調遞減（初始值 ≥ 之後所有值）"""
    if not data:
        return data
    
    max_mvc = data[0]["percent_mvc"]
    for item in data:
        if item["percent_mvc"] > max_mvc:
            item["percent_mvc"] = max_mvc
        else:
            max_mvc = item["percent_mvc"]
    return data


# ---------------------------
# 模擬情境 1：正常疲勞累積
# ---------------------------
def generate_normal_fatigue(worker_id: str, duration_minutes: int = 60):
    print_colored(f"📊 生成情境1: 正常疲勞累積 ({duration_minutes}分鐘)", "cyan")
    
    base_mvc = random.uniform(25, 35)
    data = []
    now = datetime.now()
    
    for i in range(duration_minutes):
        progress = i / duration_minutes
        mvc = base_mvc + i * 0.6 + (progress ** 2) * 15
        mvc += random.gauss(0, 2.0)
        mvc = min(max(mvc, 0), 100)
        
        timestamp = (now - timedelta(minutes=duration_minutes - i)).isoformat()
        data.append({
            "worker_id": worker_id,
            "percent_mvc": round(mvc, 2),
            "timestamp": timestamp
        })

    # 確保單調遞減
    data = ensure_monotonic_decrease(data)
    
    print(f"   ✓ 起始: {data[-1]['percent_mvc']:.1f}% → 結束: {data[0]['percent_mvc']:.1f}%")
    return data


# ---------------------------
# 模擬情境 2：工作後休息恢復
# ---------------------------
def generate_rest_recovery(worker_id: str, work_minutes: int = 40, rest_minutes: int = 20):
    print_colored(f"📊 生成情境2: 工作{work_minutes}分鐘 → 休息{rest_minutes}分鐘", "cyan")
    
    base_mvc = random.uniform(28, 35)
    data = []
    now = datetime.now()
    
    # 工作階段
    for i in range(work_minutes):
        mvc = base_mvc + i * 0.9 + random.gauss(0, 2.5)
        mvc = min(mvc, 100)
        timestamp = (now - timedelta(minutes=work_minutes + rest_minutes - i)).isoformat()
        data.append({
            "worker_id": worker_id,
            "percent_mvc": round(mvc, 2),
            "timestamp": timestamp
        })
    
    # 休息階段
    peak_mvc = data[-1]['percent_mvc']
    for i in range(rest_minutes):
        mvc = peak_mvc - i * 0.5 + random.gauss(0, 1.5)
        mvc = max(mvc, base_mvc)
        timestamp = (now - timedelta(minutes=rest_minutes - i)).isoformat()
        data.append({
            "worker_id": worker_id,
            "percent_mvc": round(mvc, 2),
            "timestamp": timestamp
        })
    
    # 確保單調遞減
    data = ensure_monotonic_decrease(data)
    
    print(f"   ✓ 起始: {data[-1]['percent_mvc']:.1f}% → 峰值: {peak_mvc:.1f}% → 結束: {data[0]['percent_mvc']:.1f}%")
    return data


# ---------------------------
# 模擬情境 3：間歇性工作
# ---------------------------
def generate_intermittent_work(worker_id: str, cycles: int = 4, work_min: int = 15, rest_min: int = 5):
    print_colored(f"📊 生成情境3: {cycles}個循環 (工作{work_min}分/休息{rest_min}分)", "cyan")
    
    base_mvc = random.uniform(25, 30)
    data = []
    now = datetime.now()
    total_minutes = cycles * (work_min + rest_min)
    accumulated_fatigue = 0
    
    for cycle in range(cycles):
        # 工作階段
        for i in range(work_min):
            time_index = cycle * (work_min + rest_min) + i
            mvc = base_mvc + accumulated_fatigue + i * 0.8 + random.gauss(0, 2.0)
            mvc = min(mvc, 100)
            timestamp = (now - timedelta(minutes=total_minutes - time_index)).isoformat()
            data.append({
                "worker_id": worker_id,
                "percent_mvc": round(mvc, 2),
                "timestamp": timestamp
            })
        
        accumulated_fatigue += work_min * 0.3
        
        # 休息階段
        rest_start_mvc = data[-1]['percent_mvc']
        for i in range(rest_min):
            time_index = cycle * (work_min + rest_min) + work_min + i
            mvc = rest_start_mvc - i * 0.6 + random.gauss(0, 1.5)
            mvc = max(mvc, base_mvc)
            timestamp = (now - timedelta(minutes=total_minutes - time_index)).isoformat()
            data.append({
                "worker_id": worker_id,
                "percent_mvc": round(mvc, 2),
                "timestamp": timestamp
            })
    
    # 確保單調遞減
    data = ensure_monotonic_decrease(data)
    
    print(f"   ✓ 起始: {data[-1]['percent_mvc']:.1f}% → 結束: {data[0]['percent_mvc']:.1f}% (殘留疲勞: -{accumulated_fatigue:.1f}%)")
    return data


# ---------------------------
# 模擬情境 4：高風險場景
# ---------------------------
def generate_high_risk_scenario(worker_id: str, duration_minutes: int = 50):
    print_colored(f"📊 生成情境4: 高風險場景 ({duration_minutes}分鐘)", "red")
    
    base_mvc = random.uniform(30, 40)
    data = []
    now = datetime.now()
    
    for i in range(duration_minutes):
        mvc = base_mvc + i * 1.2 + (i / 10) ** 2 + random.gauss(0, 3.0)
        mvc = min(mvc, 100)
        timestamp = (now - timedelta(minutes=duration_minutes - i)).isoformat()
        data.append({
            "worker_id": worker_id,
            "percent_mvc": round(mvc, 2),
            "timestamp": timestamp
        })
    
    # 確保單調遞減
    data = ensure_monotonic_decrease(data)
    
    change = data[-1]['percent_mvc'] - data[0]['percent_mvc']
    print(f"   ⚠️  起始: {data[-1]['percent_mvc']:.1f}% → 結束: {data[0]['percent_mvc']:.1f}% (變化: -{change:.1f}%)")
    return data


# ---------------------------
# API 上傳與狀態檢查
# ---------------------------
def upload_data(data: list):
    try:
        response = requests.post(
            f"{API_URL}/upload_batch",
            json={"data": data},
            timeout=10
        )
        if response.status_code == 200:
            print_colored(f"✅ 上傳成功: {len(data)} 筆資料", "green")
            return True
        else:
            print_colored(f"❌ 上傳失敗: {response.status_code}", "red")
            print(response.text)
            return False
    except Exception as e:
        print_colored(f"❌ 連線錯誤: {e}", "red")
        return False


def check_status(worker_id: str):
    try:
        response = requests.get(f"{API_URL}/status/{worker_id}", timeout=5)
        if response.status_code == 200:
            return True
        return False
    except:
        return False


# ---------------------------
# 主程式
# ---------------------------
def main():
    print("\n" + "="*70)
    print_colored("🚀 MVC 模擬資料生成器", "cyan")
    print("="*70 + "\n")
    
    scenarios = [
        ("worker_001", generate_normal_fatigue, {"duration_minutes": 60}),
        ("worker_002", generate_rest_recovery, {"work_minutes": 40, "rest_minutes": 20}),
        ("worker_003", generate_intermittent_work, {"cycles": 4, "work_min": 15, "rest_min": 5}),
        ("worker_004", generate_high_risk_scenario, {"duration_minutes": 50}),
    ]
    
    for worker_id, generator, params in scenarios:
        print(f"\n{'='*70}")
        print_colored(f"👷 處理工作者: {worker_id}", "yellow")
        print('='*70)
        
        data = generator(worker_id, **params)
        
        if upload_data(data):
            time.sleep(0.5)
            print("📊 正在分析狀態...")
            time.sleep(0.5)
            if check_status(worker_id):
                print_colored(f"✓ 可查看狀態: {API_URL}/status/{worker_id}", "green")
            else:
                print("⚠️  狀態尚未就緒")
    
    print(f"\n{'='*70}")
    print_colored("✅ 所有模擬資料生成完成！", "green")
    print("="*70)
    print(f"\n📊 查看結果:")
    print(f"  • 所有工作者: {API_URL}/workers")
    print(f"  • 個別狀態:   {API_URL}/status/worker_001")
    print(f"  • 預測結果:   {API_URL}/predict/worker_001")
    print(f"  • 圖表數據:   {API_URL}/chart/worker_001")
    print("\n💡 提示:")
    print("  - 終端機會即時顯示分析結果")
    print("  - API 會將數據傳回你的 app 端")
    print("  - 使用 GET 請求即可取得 JSON 格式數據")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
