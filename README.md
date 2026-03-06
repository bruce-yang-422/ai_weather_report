# CWA 天氣報告（機車通勤族專用）

[![GitHub license](https://img.shields.io/github/license/bruce-yang-422/ai_weather_report)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/bruce-yang-422/ai_weather_report)](https://github.com/bruce-yang-422/ai_weather_report/commits/main)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

自動從**中央氣象署（CWA）開放資料平台**抓取天氣預報，產出圖表與文字報告，針對機車通勤族設計。當 CWA API 維護或暫時失敗時，可自動切換至 **Open-Meteo** 備援來源。

---

## 功能特色

- **多區域同時查詢**：可設定多個行政區（如五股區、泰山區），各自產出獨立報告
- **真實體感溫度**：考慮溫度、濕度、風速，計算日間 / 夜間體感溫度
- **圖像化天氣報表**：未來 7 日最高 / 最低溫、體感溫度趨勢圖（`weather_report.png`）
- **文字報告**：逐區天氣日報，含今日概況、未來一週預覽、機車通勤貼心提醒
- **API 金鑰保護**：憑證存於 `cwa_api.env`，已列入 `.gitignore` 不會上傳 GitHub
- **備援資料源**：CWA 失敗時可改用 Open-Meteo 產生一週報表與今日提醒

---

## 專案結構

```text
ai_weather_report/
├── config/
│   └── config.json          # 地點、字型設定
├── output/                  # 執行後自動產出（不納入版控）
│   ├── weather_report.png
│   ├── weather_analysis_五股區.txt
│   ├── weather_analysis_泰山區.txt
│   ├── weather_raw_069.csv
│   ├── weather_raw_071.csv
│   └── weather.log
├── scripts/
│   └── weather.py           # 主程式
├── cwa_api.env              # CWA API 金鑰（本地，不納入版控）
├── run_weather.ps1          # Windows 快速執行腳本
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
    "townships": ["五股區", "泰山區"],
    "open_meteo_coords": {
      "五股區": { "latitude": 25.0827, "longitude": 121.4381 },
      "泰山區": { "latitude": 25.0589, "longitude": 121.4316 }
    }
  },
  "font": {
    "path": "C:\\Windows\\Fonts\\msjh.ttc",
    "fallback": ["Microsoft JhengHei", "SimHei"]
  }
}
```

> `townships` 可新增或移除行政區，每個區會各自產出一份文字報告。
> `open_meteo_coords` 用於 CWA API 失敗時的 Open-Meteo 備援，新增行政區時請一併設定對應經緯度。

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
| `weather_report.png` | 未來 7 日圖像化天氣報表 |
| `weather_analysis_<區名>.txt` | 各區文字日報 |
| `weather_raw_069.csv` / `weather_raw_071.csv` | 原始資料（3天/1週） |
| `weather.log` | 執行紀錄（每次覆蓋） |

---

## 輸出範例

### 文字日報（`weather_analysis_五股區.txt`）

```text
03-05(四) 氣象日報 - 五股區

🌤️ 今日概況
氣溫：17~24°C
體感：日 25°C / 夜 17°C
降雨機率：90%
短暫陣雨。偏東風 平均風速1-2級。相對濕度91%。

📅 未來一週
- 週五(03-06)：☁️ 氣溫 16-20°C / 體感 20-15°C / 降雨 30%
- 週六(03-07)：☁️ 氣溫 15-17°C / 體感 16-15°C / 降雨 10%
...

💡 貼心提醒
1) 降雨機率高時請穿雨衣並注意視線。
2) 風速升高時，高架、橋面請降低速度。
3) 體感溫度低時，請備妥手套與保暖層。
```

---

## 資料來源

- **天氣資料**：[中央氣象署開放資料平台](https://opendata.cwa.gov.tw/)
  - `F-D0047-069`：3 天逐 3 小時鄉鎮預報
  - `F-D0047-071`：1 週逐 12 小時鄉鎮預報
- **備援資料**：[Open-Meteo Forecast API](https://open-meteo.com/en/docs)

---

## 授權條款

本專案採用 MIT 授權條款，詳見 [`LICENSE`](LICENSE)。
