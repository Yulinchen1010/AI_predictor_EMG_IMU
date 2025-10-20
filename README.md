# 部署與除錯檢查清單（中文）

若在雲端部署的後端遇到上傳失敗或 500 錯誤，可依照以下流程逐項排查：

1. **確認啟動指令**  
   `Procfile` 應設定為 `web: uvicorn main:app --host=0.0.0.0 --port=${PORT:-8000}`，並確認專案的進入點檔案為 `main.py` 且內含 `app = FastAPI(...)`。

2. **顯示版本資訊**  
   在 `/healthz` 或 `/health` 回應中加入 `commit_sha` 或自訂常數 `APP_BUILD="2025-10-21-01"`，方便檢查部署版本是否更新。

3. **檢查匯入來源**  
   若將工具函式移到 `utils.py` 等模組，請記得在 `main.py` 以 `from utils import _looks_like_plain_avg, _to_mvc_0_100` 匯入，並避免造成循環匯入。

4. **清除快取並重新部署**  
   在 Render 後台執行「清除快取 / 重新部署」，或修改任一檔案以觸發全新 build，避免沿用舊的 `.pyc` 快取。

5. **健康檢查**  
   部署完成後於終端機測試：
   ```bash
   curl https://<你的域名>/healthz
   curl -s https://<你的域名>/health | jq
   ```
   確認回傳的版本資訊已更新。

6. **/process_json 單元測試**  
   以最小測試資料驗證 API：
   ```bash
   curl -v -X POST 'https://<你的域名>/process_json' \\
     -H 'Content-Type: application/json' \\
     --data '[{"worker_id":"user001","percent_mvc":55,"timestamp":"2025-10-21T01:31:00Z"}]'
   ```
   應得到 200 回應並包含 `inserted` 與 `risk` 等欄位。

7. **調整 App 上傳頻率**  
   若 App 每秒送出多次請求，請加入節流機制（例如 1–2 秒一次或等待上一筆完成再送），減少後端壓力。

8. **驗證錯誤訊息**  
   完成上述步驟後，檢查 `/process_json` 是否仍回傳 500；若已修復應回傳 200 或在資料格式不符時回傳 4xx。並確認記錄檔中不再出現 `name '_looks_like_plain_avg' is not defined` 或 `_to_mvc_0_100` 等匯入錯誤。

透過此檢查清單，可快速定位並解決上傳失敗的常見原因，確保 App 能順利取得預測結果。
