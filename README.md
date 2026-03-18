# 天氣報告（機車通勤族專用）

[![GitHub license](https://img.shields.io/github/license/bruce-yang-422/ai_weather_report)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/bruce-yang-422/ai_weather_report)](https://github.com/bruce-yang-422/ai_weather_report/commits/main)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

同時整合**中央氣象署（CWA）**與**Open-Meteo**兩個資料來源，交叉比對合併後產出圖表與文字報告，針對機車通勤族設計。

---

## 功能特色

- **雙資料源合併**：CWA + Open-Meteo 同時取得，溫度取平均、降雨機率取最大值，提升預報準確性
- **多區域同時查詢**：可設定多個行政區（如蘆洲區、泰山區），各自產出獨立報告
- **真實體感溫度**：日間 / 夜間體感溫度分開顯示
- **圖像化天氣報表**：未來 7 日溫度折線 + 降雨機率長條雙軸圖（`weather_report.png`）
- **文字報告**：逐區天氣日報，含今日概況、未來一週預覽、機車通勤貼心提醒
- **API 金鑰保護**：CWA 憑證存於 `cwa_api.env`，已列入 `.gitignore` 不會上傳 GitHub

---

## 專案結構

```text
ai_weather_report/
├── config/
│   └── config.json                    # 地點、字型設定
├── output/                            # 執行後自動產出（不納入版控）
│   ├── weather_report.png             # 7 日圖表
│   ├── weather_analysis_蘆洲區.txt
│   ├── weather_analysis_泰山區.txt
│   ├── weather_analysis.txt           # 主要區域的報告（向下相容）
│   ├── weather_raw_069.csv            # CWA 3天逐3小時（寬表）
│   ├── weather_raw_069_long.csv       # CWA 3天逐3小時（長表）
│   ├── weather_raw_071.csv            # CWA 1週逐12小時（寬表）
│   ├── weather_raw_071_long.csv       # CWA 1週逐12小時（長表）
│   ├── weather_raw_open_meteo.csv     # Open-Meteo 7日每日資料
│   └── weather.log
├── scripts/
│   └── weather.py                     # 主程式
├── cwa_api.env                        # CWA API 金鑰（本地，不納入版控）
├── run_weather.ps1                    # Windows 快速執行腳本
└── requirements.txt
```

---

## 環境需求

- Python 3.10+

```bash
pip install -r requirements.txt
```

---

## 設定步驟

### 1. 設定 CWA API 金鑰

在專案根目錄建立 `cwa_api.env`：

```env
CWA_AUTHORIZATION=CWA-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
CWA_SKIP_SSL_VERIFY=true   # 若遇到 SSL 驗證失敗可啟用
```

> API 金鑰可至 [CWA 開放資料平台](https://opendata.cwa.gov.tw/) 免費申請。

### 2. 設定查詢地點與字型

編輯 `config/config.json`：

```json
{
  "location": {
    "city": "新北市",
    "townships": ["蘆洲區", "泰山區"],
    "open_meteo_coords": {
      "蘆洲區": { "latitude": 25.076134881491722, "longitude": 121.47669968399556 },
      "泰山區": { "latitude": 25.04696465891518, "longitude": 121.42661936526125 }
    }
  },
  "font": {
    "path": "C:\\Windows\\Fonts\\msjh.ttc",
    "fallback": ["Microsoft JhengHei", "SimHei"]
  }
}
```

> `townships` 為 CWA 鄉鎮名稱，同時用於 Open-Meteo 座標對應，新增區域時兩處需一併設定。

---

## 使用方式

```bash
python scripts/weather.py
```

Windows 也可直接執行：

```powershell
.\run_weather.ps1
```

執行完成後，`output/` 資料夾會產出：

| 檔案 | 說明 |
|------|------|
| `weather_report.png` | 未來 7 日圖像化天氣報表（850px） |
| `weather_analysis_<區名>.txt` | 各區文字日報 |
| `weather_raw_069.csv` / `weather_raw_071.csv` | CWA 原始資料（寬表） |
| `weather_raw_069_long.csv` / `weather_raw_071_long.csv` | CWA 原始資料（長表） |
| `weather_raw_open_meteo.csv` | Open-Meteo 7 日每日資料 |
| `weather.log` | 執行紀錄（每次覆蓋） |

---

## 資料來源與合併策略

| 資料項目 | 合併方式 |
|----------|----------|
| 最高 / 最低溫 | CWA + Open-Meteo 平均 |
| 體感溫度（日 / 夜） | CWA + Open-Meteo 平均 |
| 降雨機率 | 取兩者最大值（較保守） |
| 天氣描述 | 優先使用 CWA（本地中文描述） |
| 濕度 | 僅 CWA 提供 |

- **CWA**：[中央氣象署開放資料平台](https://opendata.cwa.gov.tw/)
  - `F-D0047-069`：3 天逐 3 小時鄉鎮預報
  - `F-D0047-071`：1 週逐 12 小時鄉鎮預報
- **Open-Meteo**：[Open-Meteo Forecast API](https://open-meteo.com/en/docs)（免費、無需金鑰）

---

## 授權條款

本專案採用 MIT 授權條款，詳見 [`LICENSE`](LICENSE)。
