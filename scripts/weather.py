"""
天氣預報腳本 - 機車通勤族專用 (LINE 純文字 + 未來一週數值化版)

資料來源：中央氣象署 CWA 開放資料
- 未來 3 天（逐 3 小時）：F-D0047-069
- 未來 1 週（逐 12 小時）：F-D0047-071

輸出：
- output/weather_report.png
- output/weather_analysis.txt
- output/weather.log

專案結構（建議）：
ai_weather_report/
├─ scripts/weather.py
├─ config/config.json
├─ cwa_api.env
└─ output/...
"""

from __future__ import annotations

import csv
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import urllib3
import json
from urllib3.exceptions import InsecureRequestWarning

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    MATPLOTLIB_IMPORT_ERROR = None
except ImportError as exc:
    plt = None
    FontProperties = None
    MATPLOTLIB_IMPORT_ERROR = exc

# ==========================================
# 路徑與日誌
# ==========================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = OUTPUT_DIR / "cache"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
ENV_PATH = PROJECT_ROOT / "cwa_api.env"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "weather.log", encoding="utf-8", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ==========================================
# 工具：讀取 CWA 環境設定
# ==========================================

def load_cwa_env_settings(env_path: Path) -> Dict[str, str]:
    """
    支援兩種檔案格式：
    1) .env 形式：CWA_AUTHORIZATION=CWA-XXXX...
    2) 純文字：CWA-XXXX...
    """
    if not env_path.exists():
        raise FileNotFoundError(f"找不到授權碼檔案：{env_path}")

    raw = env_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"授權碼檔案為空：{env_path}")

    # 去除註解行
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise ValueError(f"授權碼檔案內容無有效行：{env_path}")

    settings: Dict[str, str] = {}
    plain_values: List[str] = []

    for line in lines:
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                settings[key] = value
        else:
            plain_values.append(line.strip().strip('"').strip("'"))

    if "CWA_AUTHORIZATION" not in settings and plain_values:
        settings["CWA_AUTHORIZATION"] = plain_values[0]

    return settings


def load_cwa_api_key(env_path: Path) -> str:
    settings = load_cwa_env_settings(env_path)
    api_key = settings.get("CWA_AUTHORIZATION", "").strip()
    if not api_key:
        raise ValueError(f"授權碼內容無值：{env_path}")
    return api_key


def _parse_bool_env_value(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _cache_file_path(dataset_id: str) -> Path:
    return CACHE_DIR / f"{dataset_id}.json"


def save_dataset_cache(dataset_id: str, data: Dict[str, Any]) -> None:
    payload = {
        "cached_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": dataset_id,
        "data": data,
    }
    _cache_file_path(dataset_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("已更新快取：%s", _cache_file_path(dataset_id))


def load_dataset_cache(dataset_id: str) -> Optional[Dict[str, Any]]:
    cache_path = _cache_file_path(dataset_id)
    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("讀取快取失敗，略過 %s：%s", cache_path, exc)
        return None

    data = payload.get("data") if isinstance(payload, dict) else None
    cached_at = payload.get("cached_at") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        logger.warning("快取格式不正確，略過 %s", cache_path)
        return None

    logger.warning("已改用快取資料：%s%s", dataset_id, f"（cached_at={cached_at}）" if cached_at else "")
    return data


DEFAULT_OPEN_METEO_COORDS: Dict[str, Dict[str, float]] = {
    "五股區": {"latitude": 25.0827, "longitude": 121.4381},
    "泰山區": {"latitude": 25.0589, "longitude": 121.4316},
}


# ==========================================
# 配置
# ==========================================

@dataclass
class Config:
    # 基本：地點
    city: str = "新北市"
    townships: List[str] = None  # 例如 ["五股區", "泰山區"]
    open_meteo_coords: Dict[str, Dict[str, float]] = None

    # 字型（Windows 預設）
    font_path: str = r"C:\Windows\Fonts\msjh.ttc"
    fallback_fonts: List[str] = None

    # 資料集
    dataset_3day: str = "F-D0047-069"
    dataset_1week: str = "F-D0047-071"

    def __post_init__(self):
        if self.townships is None:
            self.townships = ["五股區", "泰山區"]
        if self.open_meteo_coords is None:
            self.open_meteo_coords = dict(DEFAULT_OPEN_METEO_COORDS)
        if self.fallback_fonts is None:
            self.fallback_fonts = ["Microsoft JhengHei", "SimHei"]

    @classmethod
    def load_from_yaml(cls, path: Path) -> "Config":
        if not path.exists():
            logger.warning(f"找不到配置檔 {path}，使用預設值")
            return cls()

        try:
            data = json.loads(path.read_text(encoding="utf-8")) or {}
            loc = data.get("location", {}) if isinstance(data, dict) else {}

            city = loc.get("city", "新北市")
            townships = loc.get("townships") or ["五股區", "泰山區"]
            open_meteo_coords = loc.get("open_meteo_coords") or dict(DEFAULT_OPEN_METEO_COORDS)

            font = data.get("font", {}) if isinstance(data, dict) else {}
            font_path = font.get("path", r"C:\Windows\Fonts\msjh.ttc")
            fallback = font.get("fallback", ["Microsoft JhengHei", "SimHei"])

            return cls(
                city=city,
                townships=townships,
                open_meteo_coords=open_meteo_coords,
                font_path=font_path,
                fallback_fonts=fallback,
            )
        except Exception as e:
            logger.error(f"讀取配置檔失敗：{e}，使用預設值")
            return cls()


def setup_font(cfg: Config) -> None:
    if plt is None or FontProperties is None:
        logger.warning(f"matplotlib 不可用，跳過字型設置：{MATPLOTLIB_IMPORT_ERROR}")
        return

    try:
        fp = FontProperties(fname=cfg.font_path)
        plt.rcParams["font.family"] = fp.get_name()
        logger.info(f"成功載入字型：{fp.get_name()}")
    except Exception as e:
        logger.warning(f"無法載入指定字型，改用備用字型：{e}")
        plt.rcParams["font.sans-serif"] = cfg.fallback_fonts
    plt.rcParams["axes.unicode_minus"] = False


# ==========================================
# 風力警告（機車族）
# ==========================================

class WindWarningSystem:
    # 以 km/h 判定（與你原本邏輯一致）
    WIND_LEVELS = [
        (39, 49, "⚠️今日有6級強風，騎經高架或路口請抓緊龍頭。"),
        (50, 61, "⚠️今日7級疾風，車身會明顯晃動,請放慢車速。"),
        (62, 88, "⛔今日8-9級烈風，極度危險！務必慢行，防範路邊倒車。"),
        (89, float("inf"), "☠️今日10級狂風，生命受威脅，強烈建議不要騎車出門。"),
    ]

    @classmethod
    def get_warning(cls, wind_kmh: float) -> str:
        for mn, mx, msg in cls.WIND_LEVELS:
            if mn <= wind_kmh <= mx:
                return msg
        return ""

    @classmethod
    def is_dangerous(cls, wind_kmh: float) -> bool:
        return wind_kmh >= 39


# ==========================================
# CWA API Client
# ==========================================

class CWAClient:
    BASE = "https://opendata.cwa.gov.tw/api/v1/rest/datastore"
    RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str,
        timeout: int = 15,
        skip_ssl_verify: bool = False,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.skip_ssl_verify = skip_ssl_verify
        self.max_retries = max(1, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    def _get_with_retry(
        self,
        url: str,
        params: Dict[str, Any],
        headers: Dict[str, str],
        verify: bool,
    ) -> requests.Response:
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                    verify=verify,
                )
                if resp.status_code in self.RETRY_STATUS_CODES:
                    raise requests.exceptions.HTTPError(
                        f"HTTP {resp.status_code}",
                        response=resp,
                    )
                return resp
            except requests.exceptions.RequestException as exc:
                last_exc = exc

                status_code = None
                if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
                    status_code = exc.response.status_code

                is_retryable_http = status_code in self.RETRY_STATUS_CODES
                is_retryable_network = isinstance(
                    exc,
                    (
                        requests.exceptions.Timeout,
                        requests.exceptions.ConnectionError,
                    ),
                )

                if attempt >= self.max_retries or not (is_retryable_http or is_retryable_network):
                    raise

                wait = self.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "CWA API 請求失敗（第 %d/%d 次，%s），%.1f 秒後重試...",
                    attempt,
                    self.max_retries,
                    f"HTTP {status_code}" if status_code else exc.__class__.__name__,
                    wait,
                )
                if wait > 0:
                    time.sleep(wait)

        if last_exc:
            raise last_exc
        raise RuntimeError("CWA API 請求失敗：未知錯誤")

    def get(self, dataset_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.BASE}/{dataset_id}"
        merged = {"format": "JSON"}
        merged.update(params)
        headers = {"Authorization": self.api_key}

        if self.skip_ssl_verify:
            urllib3.disable_warnings(InsecureRequestWarning)
            resp = self._get_with_retry(url, merged, headers, verify=False)
            resp.raise_for_status()
            data = resp.json()
            if str(data.get("success")).lower() != "true":
                raise RuntimeError(f"CWA API 回傳 success != true：{data.get('msg') or data}")
            return data

        try:
            resp = self._get_with_retry(url, merged, headers, verify=True)
        except requests.exceptions.SSLError as exc:
            logger.warning(f"CWA SSL 驗證失敗，改用未驗證連線重試：{exc}")
            urllib3.disable_warnings(InsecureRequestWarning)
            resp = self._get_with_retry(url, merged, headers, verify=False)
        resp.raise_for_status()
        data = resp.json()

        # 平臺常見：success: "true"/"false"
        if str(data.get("success")).lower() != "true":
            raise RuntimeError(f"CWA API 回傳 success != true：{data.get('msg') or data}")

        return data


class OpenMeteoClient:
    BASE = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, timeout: int = 15, max_retries: int = 3, retry_backoff_seconds: float = 1.0):
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    def get_forecast(self, latitude: float, longitude: float) -> Dict[str, Any]:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "Asia/Taipei",
            "forecast_days": 7,
            "daily": [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "apparent_temperature_max",
                "apparent_temperature_min",
                "precipitation_probability_max",
                "wind_speed_10m_max",
            ],
            "hourly": [
                "weather_code",
                "wind_speed_10m",
            ],
        }

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(self.BASE, params=params, timeout=self.timeout)
                if resp.status_code in CWAClient.RETRY_STATUS_CODES:
                    raise requests.exceptions.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                return resp.json()
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                wait = self.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning("Open-Meteo 請求失敗（第 %d/%d 次），%.1f 秒後重試...", attempt, self.max_retries, wait)
                if wait > 0:
                    time.sleep(wait)

        if last_exc:
            raise last_exc
        raise RuntimeError("Open-Meteo 請求失敗：未知錯誤")


# ==========================================
# CWA 解析工具
# ==========================================

def _ensure_records_locations(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = data.get("records") or {}
    locations_list = records.get("Locations") or records.get("locations") or []
    if not locations_list:
        raise ValueError("回傳資料缺少 records.Locations")
    return locations_list


def _find_city_block(locations_list: List[Dict[str, Any]], city: str) -> Dict[str, Any]:
    for blk in locations_list:
        if blk.get("LocationsName") == city:
            return blk
    # 若 API 已被 LocationsName 篩過，可能只有一個
    return locations_list[0]


def _index_weather_elements(location_obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    回傳 dict：{ElementName: element_object}
    element_object 內含 Time 列表
    """
    elements = location_obj.get("WeatherElement") or []
    idx = {}
    for e in elements:
        name = e.get("ElementName")
        if name:
            idx[name] = e
    return idx


def _parse_dt(s: str) -> datetime:
    """
    支援：
    - 2026-03-04T06:00:00+08:00
    - 2026-03-04T06:00:00

    為了避免 aware/naive 混算，統一轉成本機時區的 naive datetime。
    """
    if not s:
        raise ValueError("空時間字串")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def _to_mmdd_weekday(dt: date) -> str:
    weekday_map = ["一", "二", "三", "四", "五", "六", "日"]
    return f"{dt.strftime('%m-%d')}({weekday_map[dt.weekday()]})"


def _open_meteo_code_to_text(code: Optional[int]) -> str:
    if code is None:
        return "多雲"
    if code == 0:
        return "晴"
    if code in {1, 2, 3}:
        return "多雲"
    if code in {45, 48}:
        return "霧"
    if code in {51, 53, 55, 56, 57}:
        return "毛毛雨"
    if code in {61, 63, 65, 66, 67, 80, 81, 82}:
        return "雨"
    if code in {71, 73, 75, 77, 85, 86}:
        return "雪"
    if code in {95, 96, 99}:
        return "雷雨"
    return "多雲"


def _describe_open_meteo_day(code: Optional[int], pop: Optional[int], wind_kmh: Optional[float]) -> str:
    parts = [_open_meteo_code_to_text(code)]
    if pop is not None:
        parts.append(f"降雨機率 {pop}%")
    if wind_kmh is not None:
        parts.append(f"最大風速 {wind_kmh:.0f} km/h")
    return "，".join(parts)


# ==========================================
# 原始資料輸出：CSV（中文欄位）
# ==========================================

CSV_FIELDNAMES = [
    "資料集代碼",
    "資料集說明",
    "縣市",
    "鄉鎮市區",
    "天氣要素",
    "開始時間",
    "結束時間",
    "資料時間",
    "值序號",
    "值欄位",
    "值內容",
    "測量單位",
]

VALUE_FIELD_LABEL_MAP = {
    "Temperature": "溫度",
    "MaxTemperature": "最高溫度",
    "MinTemperature": "最低溫度",
    "ApparentTemperature": "體感溫度",
    "MaxApparentTemperature": "最高體感溫度",
    "MinApparentTemperature": "最低體感溫度",
    "ProbabilityOfPrecipitation": "降雨機率",
    "RelativeHumidity": "相對濕度",
    "Weather": "天氣現象",
    "WeatherDescription": "天氣預報綜合描述",
    "WindSpeed": "風速",
    "BeaufortScale": "蒲福風級",
    "ComfortIndex": "舒適度指數",
    "ComfortIndexDescription": "舒適度描述",
    "DewPoint": "露點溫度",
    "WeatherCode": "天氣代碼",
    "WindDirection": "風向",
    "MaxComfortIndex": "最高舒適度指數",
    "MaxComfortIndexDescription": "最高舒適度描述",
    "MinComfortIndex": "最低舒適度指數",
    "MinComfortIndexDescription": "最低舒適度描述",
}

WIDE_BASE_FIELDNAMES = [
    "資料集代碼",
    "資料集說明",
    "縣市",
    "鄉鎮市區",
    "開始時間",
    "結束時間",
    "資料時間",
]


def _normalize_scalar_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return str(value)


def _to_chinese_value_field_name(field_name: str) -> str:
    return VALUE_FIELD_LABEL_MAP.get(field_name, field_name)


def _to_wide_column_name(
    element_name: str,
    value_field_name: str,
    value_index: int = 1,
    value_count: int = 1,
) -> str:
    zh_field_name = _to_chinese_value_field_name(value_field_name)

    if not zh_field_name or zh_field_name == "值":
        column_name = element_name
    elif zh_field_name == element_name:
        column_name = element_name
    else:
        column_name = f"{element_name}_{zh_field_name}"

    if value_count > 1:
        column_name = f"{column_name}_{value_index}"

    return column_name


def _flatten_element_value_rows(
    element_values: List[Dict[str, Any]],
    base_row: Dict[str, Any],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    for idx, ev in enumerate(element_values, start=1):
        if not isinstance(ev, dict):
            row = dict(base_row)
            row.update(
                {
                    "值序號": str(idx),
                    "值欄位": "值",
                    "值內容": _normalize_scalar_value(ev),
                    "測量單位": "",
                }
            )
            rows.append(row)
            continue

        unit = _normalize_scalar_value(ev.get("Measures"))
        value_keys = [k for k in ev.keys() if k != "Measures" and ev.get(k) is not None]

        if not value_keys:
            row = dict(base_row)
            row.update(
                {
                    "值序號": str(idx),
                    "值欄位": "測量單位",
                    "值內容": unit,
                    "測量單位": "",
                }
            )
            rows.append(row)
            continue

        for key in value_keys:
            row = dict(base_row)
            row.update(
                {
                    "值序號": str(idx),
                    "值欄位": _to_chinese_value_field_name(key),
                    "值內容": _normalize_scalar_value(ev.get(key)),
                    "測量單位": unit,
                }
            )
            rows.append(row)

    return rows


def build_cwa_raw_csv_rows(
    dataset_id: str,
    dataset_label: str,
    data: Dict[str, Any],
    city: str,
    townships: List[str],
) -> List[Dict[str, str]]:
    locations_list = _ensure_records_locations(data)
    city_blk = _find_city_block(locations_list, city)
    locs = city_blk.get("Location") or []
    target_towns = set(townships or [])

    rows: List[Dict[str, str]] = []

    for loc in locs:
        town = loc.get("LocationName") or ""
        if target_towns and town not in target_towns:
            continue

        for element in loc.get("WeatherElement") or []:
            element_name = element.get("ElementName") or "未命名天氣要素"
            time_items = element.get("Time") or []

            for time_item in time_items:
                base_row = {
                    "資料集代碼": dataset_id,
                    "資料集說明": dataset_label,
                    "縣市": city_blk.get("LocationsName") or city,
                    "鄉鎮市區": town,
                    "天氣要素": element_name,
                    "開始時間": _normalize_scalar_value(time_item.get("StartTime")),
                    "結束時間": _normalize_scalar_value(time_item.get("EndTime")),
                    "資料時間": _normalize_scalar_value(time_item.get("DataTime")),
                    "值序號": "",
                    "值欄位": "",
                    "值內容": "",
                    "測量單位": "",
                }

                element_values = time_item.get("ElementValue") or []
                if element_values:
                    rows.extend(_flatten_element_value_rows(element_values, base_row))
                else:
                    rows.append(base_row)

    return rows


def export_cwa_raw_csv(output_path: Path, rows: List[Dict[str, str]]) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"原始資料 CSV 已生成：{output_path}")


def build_cwa_raw_wide_rows(
    dataset_id: str,
    dataset_label: str,
    data: Dict[str, Any],
    city: str,
    townships: List[str],
) -> Tuple[List[Dict[str, str]], List[str]]:
    locations_list = _ensure_records_locations(data)
    city_blk = _find_city_block(locations_list, city)
    locs = city_blk.get("Location") or []
    target_towns = set(townships or [])

    rows_by_key: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    dynamic_fieldnames: List[str] = []
    dynamic_seen = set()

    def ensure_dynamic_field(field_name: str) -> None:
        if field_name not in dynamic_seen:
            dynamic_seen.add(field_name)
            dynamic_fieldnames.append(field_name)

    for loc in locs:
        town = loc.get("LocationName") or ""
        if target_towns and town not in target_towns:
            continue

        for element in loc.get("WeatherElement") or []:
            element_name = element.get("ElementName") or "未命名天氣要素"
            time_items = element.get("Time") or []

            for time_item in time_items:
                start_time = _normalize_scalar_value(time_item.get("StartTime"))
                end_time = _normalize_scalar_value(time_item.get("EndTime"))
                data_time = _normalize_scalar_value(time_item.get("DataTime"))
                key = (town, start_time, end_time, data_time)

                row = rows_by_key.setdefault(
                    key,
                    {
                        "資料集代碼": dataset_id,
                        "資料集說明": dataset_label,
                        "縣市": city_blk.get("LocationsName") or city,
                        "鄉鎮市區": town,
                        "開始時間": start_time,
                        "結束時間": end_time,
                        "資料時間": data_time,
                    },
                )

                element_values = time_item.get("ElementValue") or []
                if not element_values:
                    column_name = element_name
                    ensure_dynamic_field(column_name)
                    row[column_name] = ""
                    continue

                value_count = len(element_values)
                for idx, ev in enumerate(element_values, start=1):
                    if not isinstance(ev, dict):
                        column_name = _to_wide_column_name(element_name, "值", idx, value_count)
                        ensure_dynamic_field(column_name)
                        row[column_name] = _normalize_scalar_value(ev)
                        continue

                    value_keys = [k for k in ev.keys() if k != "Measures" and ev.get(k) is not None]
                    if not value_keys:
                        column_name = _to_wide_column_name(element_name, "值", idx, value_count)
                        ensure_dynamic_field(column_name)
                        row[column_name] = ""
                        continue

                    for value_key in value_keys:
                        column_name = _to_wide_column_name(element_name, value_key, idx, value_count)
                        ensure_dynamic_field(column_name)
                        row[column_name] = _normalize_scalar_value(ev.get(value_key))

    sorted_rows = sorted(
        rows_by_key.values(),
        key=lambda r: (
            r.get("鄉鎮市區", ""),
            r.get("開始時間", ""),
            r.get("資料時間", ""),
            r.get("結束時間", ""),
        ),
    )
    return sorted_rows, WIDE_BASE_FIELDNAMES + dynamic_fieldnames


def export_cwa_raw_wide_csv(
    output_path: Path,
    rows: List[Dict[str, str]],
    fieldnames: List[str],
) -> None:
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"原始資料寬表 CSV 已生成：{output_path}")


# ==========================================
# 週報（F-D0047-071）彙整為「每日」
# ==========================================

@dataclass
class DailyForecast:
    d: date
    condition: str
    tmax: Optional[float]
    tmin: Optional[float]
    feel_day: Optional[float]   # 以最高體感代表日間感受
    feel_night: Optional[float] # 以最低體感代表夜間感受
    pop: Optional[int]          # 降雨機率（取當日最大）
    humidity: Optional[float]   # 相對濕度（若有，取平均）
    desc: Optional[str]


def build_weekly_daily_forecast(
    data_071: Dict[str, Any],
    city: str,
    townships: List[str],
) -> Dict[str, List[DailyForecast]]:
    """
    回傳：{ township_name: [DailyForecast, ...] }
    071 為逐 12 小時；本函數彙整成每日資訊。
    """
    locations_list = _ensure_records_locations(data_071)
    city_blk = _find_city_block(locations_list, city)
    locs = city_blk.get("Location") or []

    target = {t: [] for t in townships}

    for loc in locs:
        name = loc.get("LocationName")
        if name not in target:
            continue

        idx = _index_weather_elements(loc)

        # 可能的 ElementName（以你先前 fields 為準）
        # - 天氣現象：Weather / 或 ElementValue 內 Weather
        # - 最高溫：MaxTemperature
        # - 最低溫：MinTemperature
        # - 最高體感：MaxApparentTemperature
        # - 最低體感：MinApparentTemperature
        # - 降雨機率：ProbabilityOfPrecipitation
        # - 相對濕度：RelativeHumidity
        # - 綜合描述：WeatherDescription

        def collect_series(element_name: str) -> List[Dict[str, Any]]:
            e = idx.get(element_name)
            return (e.get("Time") or []) if e else []

        wx_series = collect_series("天氣現象") or collect_series("Weather")
        tmax_series = collect_series("最高溫度") or collect_series("MaxTemperature")
        tmin_series = collect_series("最低溫度") or collect_series("MinTemperature")
        atmax_series = collect_series("最高體感溫度") or collect_series("MaxApparentTemperature")
        atmin_series = collect_series("最低體感溫度") or collect_series("MinApparentTemperature")
        pop_series = collect_series("12小時降雨機率") or collect_series("ProbabilityOfPrecipitation")
        rh_series = collect_series("平均相對濕度") or collect_series("RelativeHumidity")
        desc_series = collect_series("天氣預報綜合描述") or collect_series("WeatherDescription")

        # 以 StartTime 作為分桶鍵（12 小時一筆）
        day_bucket: Dict[date, Dict[str, Any]] = {}

        def bucket_by_date(series: List[Dict[str, Any]], kind: str):
            for it in series:
                st = it.get("StartTime") or it.get("DataTime")
                if not st:
                    continue
                dt = _parse_dt(st)
                d0 = dt.date()
                b = day_bucket.setdefault(d0, {"wx": [], "tmax": [], "tmin": [], "atmax": [], "atmin": [], "pop": [], "rh": [], "desc": []})
                b[kind].append(it)

        bucket_by_date(wx_series, "wx")
        bucket_by_date(tmax_series, "tmax")
        bucket_by_date(tmin_series, "tmin")
        bucket_by_date(atmax_series, "atmax")
        bucket_by_date(atmin_series, "atmin")
        bucket_by_date(pop_series, "pop")
        bucket_by_date(rh_series, "rh")
        bucket_by_date(desc_series, "desc")

        # 依日期排序
        days = sorted(day_bucket.keys())
        out: List[DailyForecast] = []

        for d0 in days[:7]:
            b = day_bucket[d0]

            # 天氣：選當日第一筆 Weather
            condition = None
            if b["wx"]:
                ev = (b["wx"][0].get("ElementValue") or [{}])[0]
                condition = ev.get("Weather") or ev.get("value") or ev.get("Wx") or None
            condition = condition or "—"

            # 最高/最低溫：取當日所有值的 max/min
            def extract_float(items: List[Dict[str, Any]], key: str) -> List[float]:
                vals = []
                for it in items:
                    ev = (it.get("ElementValue") or [{}])[0]
                    v = ev.get(key)
                    if v is None:
                        continue
                    try:
                        vals.append(float(v))
                    except Exception:
                        continue
                return vals

            tmax_vals = extract_float(b["tmax"], "MaxTemperature") or extract_float(b["tmax"], "Temperature")
            tmin_vals = extract_float(b["tmin"], "MinTemperature") or extract_float(b["tmin"], "Temperature")
            atmax_vals = extract_float(b["atmax"], "MaxApparentTemperature") or extract_float(b["atmax"], "ApparentTemperature")
            atmin_vals = extract_float(b["atmin"], "MinApparentTemperature") or extract_float(b["atmin"], "ApparentTemperature")

            # 降雨機率：取最大
            pop_vals = []
            for it in b["pop"]:
                ev = (it.get("ElementValue") or [{}])[0]
                v = ev.get("ProbabilityOfPrecipitation")
                if v is None:
                    continue
                try:
                    pop_vals.append(int(float(v)))
                except Exception:
                    continue

            # 濕度：取平均
            rh_vals = []
            for it in b["rh"]:
                ev = (it.get("ElementValue") or [{}])[0]
                v = ev.get("RelativeHumidity")
                if v is None:
                    continue
                try:
                    rh_vals.append(float(v))
                except Exception:
                    continue

            # 描述：選當日第一筆
            desc = None
            if b["desc"]:
                ev = (b["desc"][0].get("ElementValue") or [{}])[0]
                desc = ev.get("WeatherDescription") or ev.get("value")

            out.append(
                DailyForecast(
                    d=d0,
                    condition=condition,
                    tmax=max(tmax_vals) if tmax_vals else None,
                    tmin=min(tmin_vals) if tmin_vals else None,
                    feel_day=max(atmax_vals) if atmax_vals else None,
                    feel_night=min(atmin_vals) if atmin_vals else None,
                    pop=max(pop_vals) if pop_vals else None,
                    humidity=float(np.mean(rh_vals)) if rh_vals else None,
                    desc=desc,
                )
            )

        target[name] = out

    return target


# ==========================================
# 近三天（F-D0047-069）用於「今日概況/風力提醒」
# ==========================================

@dataclass
class ShortTermSnapshot:
    today_desc: Optional[str]
    wind_warning: Optional[str]


def build_today_snapshot_and_wind_warning(
    data_069: Dict[str, Any],
    city: str,
    townships: List[str],
) -> Dict[str, ShortTermSnapshot]:
    """
    回傳每個鄉鎮：
    - today_desc：取「天氣預報綜合描述」中，最接近現在的一個 3 小時區間文字
    - wind_warning：取未來 24 小時最大風速（m/s -> km/h）作為警示依據
    """
    now = datetime.now()
    locations_list = _ensure_records_locations(data_069)
    city_blk = _find_city_block(locations_list, city)
    locs = city_blk.get("Location") or []

    out: Dict[str, ShortTermSnapshot] = {}

    for loc in locs:
        name = loc.get("LocationName")
        if name not in townships:
            continue

        idx = _index_weather_elements(loc)

        # 風速（3小時）
        wind_series = (idx.get("風速") or idx.get("WindSpeed") or {}).get("Time") or []
        # 綜合描述（3小時）
        desc_series = (idx.get("天氣預報綜合描述") or idx.get("WeatherDescription") or {}).get("Time") or []

        # 1) today_desc：找離 now 最近的 StartTime
        best_desc = None
        best_dt_diff = None
        for it in desc_series:
            st = it.get("StartTime") or it.get("DataTime")
            if not st:
                continue
            dt0 = _parse_dt(st)
            diff = abs((dt0 - now).total_seconds())
            if best_dt_diff is None or diff < best_dt_diff:
                ev = (it.get("ElementValue") or [{}])[0]
                best_desc = ev.get("WeatherDescription") or ev.get("value")
                best_dt_diff = diff

        # 2) wind_warning：未來 24 小時最大風速
        max_wind_ms = None
        horizon = now + timedelta(hours=24)
        for it in wind_series:
            st = it.get("DataTime") or it.get("StartTime")
            if not st:
                continue
            dt0 = _parse_dt(st)
            if dt0 < now or dt0 > horizon:
                continue
            ev = (it.get("ElementValue") or [{}])[0]
            ws = ev.get("WindSpeed")
            if ws is None:
                continue
            try:
                ws_ms = float(ws)
            except Exception:
                continue
            if max_wind_ms is None or ws_ms > max_wind_ms:
                max_wind_ms = ws_ms

        wind_warning = None
        if max_wind_ms is not None:
            wind_kmh = max_wind_ms * 3.6
            wind_warning = WindWarningSystem.get_warning(wind_kmh) or None

        out[name] = ShortTermSnapshot(today_desc=best_desc, wind_warning=wind_warning)

    return out


def build_open_meteo_weekly_daily_forecast(data: Dict[str, Any]) -> List[DailyForecast]:
    daily = data.get("daily") or {}
    times = daily.get("time") or []
    weather_codes = daily.get("weather_code") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    feel_max = daily.get("apparent_temperature_max") or []
    feel_min = daily.get("apparent_temperature_min") or []
    pop_max = daily.get("precipitation_probability_max") or []
    wind_max = daily.get("wind_speed_10m_max") or []

    out: List[DailyForecast] = []
    for i, day_str in enumerate(times[:7]):
        d0 = datetime.fromisoformat(day_str).date()
        code = weather_codes[i] if i < len(weather_codes) else None
        pop = pop_max[i] if i < len(pop_max) else None
        wind = wind_max[i] if i < len(wind_max) else None
        out.append(
            DailyForecast(
                d=d0,
                condition=_open_meteo_code_to_text(int(code)) if code is not None else "多雲",
                tmax=float(tmax[i]) if i < len(tmax) and tmax[i] is not None else None,
                tmin=float(tmin[i]) if i < len(tmin) and tmin[i] is not None else None,
                feel_day=float(feel_max[i]) if i < len(feel_max) and feel_max[i] is not None else None,
                feel_night=float(feel_min[i]) if i < len(feel_min) and feel_min[i] is not None else None,
                pop=int(round(float(pop))) if pop is not None else None,
                humidity=None,
                desc=_describe_open_meteo_day(
                    int(code) if code is not None else None,
                    int(round(float(pop))) if pop is not None else None,
                    float(wind) if wind is not None else None,
                ),
            )
        )
    return out


def build_open_meteo_snapshot(data: Dict[str, Any]) -> ShortTermSnapshot:
    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    weather_codes = hourly.get("weather_code") or []
    wind_speeds = hourly.get("wind_speed_10m") or []
    now = datetime.now()
    horizon = now + timedelta(hours=24)

    best_desc = None
    best_dt_diff = None
    max_wind_kmh = None

    for i, ts in enumerate(times):
        try:
            dt0 = datetime.fromisoformat(ts)
        except Exception:
            continue

        code = weather_codes[i] if i < len(weather_codes) else None
        wind = wind_speeds[i] if i < len(wind_speeds) else None

        diff = abs((dt0 - now).total_seconds())
        if best_dt_diff is None or diff < best_dt_diff:
            best_desc = _open_meteo_code_to_text(int(code)) if code is not None else "多雲"
            best_dt_diff = diff

        if now <= dt0 <= horizon and wind is not None:
            wind_kmh = float(wind)
            if max_wind_kmh is None or wind_kmh > max_wind_kmh:
                max_wind_kmh = wind_kmh

    wind_warning = WindWarningSystem.get_warning(max_wind_kmh) if max_wind_kmh is not None else None
    return ShortTermSnapshot(today_desc=best_desc, wind_warning=wind_warning or None)


# ==========================================
# 報表輸出：圖表
# ==========================================

def generate_image_report(
    output_path: Path,
    days: List[str],
    tmax: List[Optional[float]],
    tmin: List[Optional[float]],
    day_feels: List[Optional[float]],
    night_feels: List[Optional[float]],
    conditions: List[str],
    rain_probs: List[Optional[int]],
    humidities: List[Optional[float]],
) -> None:
    if plt is None:
        logger.warning(f"matplotlib 不可用，跳過圖表報表生成：{MATPLOTLIB_IMPORT_ERROR}")
        return

    fig, (ax_table, ax_chart) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 10),
        gridspec_kw={"height_ratios": [0.8, 1]},
        facecolor="white",
    )

    # 表格
    ax_table.axis("off")
    ax_table.set_title("未來 7 天天氣預報 (CWA)", fontsize=16, pad=20, weight="bold")

    columns = ("日期", "天氣", "最高溫\n(°C)", "最低溫\n(°C)", "體感(°C)\n日/夜", "降雨\n(%)", "濕度\n(%)")
    cell_text = []

    for i in range(len(days)):
        d_feel = f"{day_feels[i]:.1f}" if day_feels[i] is not None else "-"
        n_feel = f"{night_feels[i]:.1f}" if night_feels[i] is not None else "-"
        feel_str = f"{d_feel} / {n_feel}"

        tmax_str = f"{tmax[i]:.0f}" if tmax[i] is not None else "-"
        tmin_str = f"{tmin[i]:.0f}" if tmin[i] is not None else "-"
        pop_str = f"{rain_probs[i]:.0f}" if rain_probs[i] is not None else "-"
        rh_str = f"{humidities[i]:.0f}" if humidities[i] is not None else "-"

        cell_text.append([days[i], conditions[i], tmax_str, tmin_str, feel_str, pop_str, rh_str])

    table = ax_table.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        bbox=[0.05, 0.1, 0.9, 0.8],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # 表格樣式
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4A90E2")
            cell.set_text_props(weight="bold", color="white")
        elif row % 2 == 0:
            cell.set_facecolor("#f9f9f9")
        if col == 4 and row > 0:
            cell.set_text_props(weight="bold", color="#d62728")

    # 折線圖
    ax_chart.set_facecolor("white")

    # 只畫有值的（避免 None 造成斷線）
    x = np.arange(len(days))

    def _plot_series(y, label, marker, linestyle):
        yv = np.array([np.nan if v is None else float(v) for v in y], dtype=float)
        ax_chart.plot(days, yv, marker=marker, label=label, linestyle=linestyle, linewidth=2.5, alpha=0.8)

    _plot_series(tmax, "最高溫", "o", "-")
    _plot_series(tmin, "最低溫", "o", "-")
    _plot_series(day_feels, "白天體感", "^", "--")
    _plot_series(night_feels, "夜間體感", "v", ":")

    ax_chart.set_xlabel("日期", fontsize=12)
    ax_chart.set_ylabel("溫度 (°C)", fontsize=12)
    ax_chart.set_title("氣溫與體感走勢", fontsize=14, weight="bold")
    ax_chart.grid(True, linestyle="--", alpha=0.3)
    ax_chart.legend(loc="best", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close()
    logger.info(f"圖表已生成：{output_path}")


# ==========================================
# 報表輸出：純文字（LINE 友善）
# ==========================================

class StructuredTextReportGenerator:
    ICON_MAP = {
        "晴": "☀️",
        "多雲": "☁️",
        "陰": "☁️",
        "雨": "🌧️",
        "雷": "⛈️",
        "霧": "🌫️",
    }

    @classmethod
    def get_icon(cls, condition: str) -> str:
        for k, icon in cls.ICON_MAP.items():
            if k in condition:
                return icon
        return "☁️"

    @classmethod
    def generate(
        cls,
        output_path: Path,
        township: str,
        today_desc: Optional[str],
        wind_warning: Optional[str],
        daily: List[DailyForecast],
    ) -> None:
        now = datetime.now()
        weekday_map = ["一", "二", "三", "四", "五", "六", "日"]
        today_title = f"{now.strftime('%m-%d')}({weekday_map[now.weekday()]}) 氣象日報 - {township}"

        lines: List[str] = []
        lines.append(today_title)
        lines.append("")
        lines.append("🌤️ 今日概況")

        if daily:
            d0 = daily[0]
            tmin = f"{d0.tmin:.0f}" if d0.tmin is not None else "N/A"
            tmax = f"{d0.tmax:.0f}" if d0.tmax is not None else "N/A"
            lines.append(f"氣溫：{tmin}~{tmax}°C")

            fd = f"{d0.feel_day:.0f}" if d0.feel_day is not None else "N/A"
            fn = f"{d0.feel_night:.0f}" if d0.feel_night is not None else "N/A"
            lines.append(f"體感：日 {fd}°C / 夜 {fn}°C")

            pop = f"{d0.pop:d}%" if d0.pop is not None else "N/A"
            lines.append(f"降雨機率：{pop}")
        else:
            lines.append("氣溫：N/A")
            lines.append("體感：N/A")
            lines.append("降雨機率：N/A")

        if today_desc:
            lines.append(today_desc)
        else:
            lines.append("無特殊提醒")

        if wind_warning:
            lines.append("")
            lines.append("🛵 風力提醒")
            lines.append(wind_warning)

        lines.append("")
        lines.append("📅 未來一週")

        # 從第二天開始列（避免與今日重複）
        for i in range(1, min(7, len(daily))):
            di = daily[i]
            day_name = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"][di.d.weekday()]
            icon = cls.get_icon(di.condition)

            tmin = f"{di.tmin:.0f}" if di.tmin is not None else "-"
            tmax = f"{di.tmax:.0f}" if di.tmax is not None else "-"
            fd = f"{di.feel_day:.0f}" if di.feel_day is not None else "-"
            fn = f"{di.feel_night:.0f}" if di.feel_night is not None else "-"
            pop = f"{di.pop:d}%" if di.pop is not None else "N/A"

            line = f"- {day_name}({_to_mmdd_weekday(di.d)[:5]})：{icon} 氣溫 {tmin}-{tmax}°C / 體感 {fd}-{fn}°C / 降雨 {pop}"
            if di.desc:
                # 控制長度，避免 LINE 太長
                short = di.desc.strip()
                if len(short) > 60:
                    short = short[:60] + "..."
                line += f"（{short}）"
            lines.append(line)

        lines.append("")
        lines.append("💡 貼心提醒")
        lines.append("1) 以「降雨機率」作為通勤決策參考，雨天請穿著雨衣並注意視線。")
        lines.append("2) 風速/陣風升高時，高架、橋面、路口與大型車旁邊請降低速度。")
        lines.append("3) 體感溫度較低時，請注意手套與保暖層，避免受寒。")

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"文字報告已生成：{output_path}")


# ==========================================
# 主流程
# ==========================================

def main() -> None:
    logger.info("=" * 60)
    logger.info("CWA 天氣預報系統啟動")
    logger.info("=" * 60)

    cfg = Config.load_from_yaml(CONFIG_PATH)
    setup_font(cfg)

    env_settings = load_cwa_env_settings(ENV_PATH)
    api_key = env_settings.get("CWA_AUTHORIZATION", "").strip()
    if not api_key:
        raise ValueError(f"授權碼內容無值：{ENV_PATH}")

    skip_ssl_verify = _parse_bool_env_value(env_settings.get("CWA_SKIP_SSL_VERIFY"), default=False)
    if skip_ssl_verify:
        logger.warning("已啟用 CWA_SKIP_SSL_VERIFY，將直接跳過 CWA HTTPS 憑證驗證")

    client = CWAClient(api_key, skip_ssl_verify=skip_ssl_verify)
    open_meteo_client = OpenMeteoClient()

    def fetch_dataset_with_cache(dataset_id: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            data = client.get(dataset_id, params=params)
            save_dataset_cache(dataset_id, data)
            return data
        except requests.exceptions.RequestException as exc:
            logger.error("取得 %s 失敗：%s", dataset_id, exc)
            cached = load_dataset_cache(dataset_id)
            if cached is not None:
                return cached
            return None

    def fetch_open_meteo_by_town() -> Tuple[Dict[str, List[DailyForecast]], Dict[str, ShortTermSnapshot]]:
        weekly: Dict[str, List[DailyForecast]] = {}
        snapshots: Dict[str, ShortTermSnapshot] = {}

        for town in cfg.townships:
            coord = cfg.open_meteo_coords.get(town) if cfg.open_meteo_coords else None
            if not coord:
                logger.warning("找不到 %s 的 Open-Meteo 座標設定，略過備援", town)
                continue

            latitude = coord.get("latitude")
            longitude = coord.get("longitude")
            if latitude is None or longitude is None:
                logger.warning("%s 的 Open-Meteo 座標不完整，略過備援", town)
                continue

            try:
                logger.info("正在取得 Open-Meteo 備援資料：%s", town)
                om_data = open_meteo_client.get_forecast(float(latitude), float(longitude))
            except requests.exceptions.RequestException as exc:
                logger.error("取得 Open-Meteo 備援資料失敗（%s）：%s", town, exc)
                continue

            weekly[town] = build_open_meteo_weekly_daily_forecast(om_data)
            snapshots[town] = build_open_meteo_snapshot(om_data)

        return weekly, snapshots

    # 1) 拉資料：3 天（3 小時） + 1 週（12 小時）
    data_069: Optional[Dict[str, Any]] = None
    logger.info("正在取得 F-D0047-069（3天/3小時）資料...")
    # requests params 以 list 帶多個同名 key：
    # LocationName=...&LocationName=...
    data_069 = fetch_dataset_with_cache(
        cfg.dataset_3day,
        params={
            "LocationsName": cfg.city,
            "LocationName": cfg.townships,  # list -> LocationName=...&LocationName=...
        },
    )
    if data_069 is None:
        logger.error("取得 F-D0047-069 失敗，且無可用快取，將以降級模式繼續執行")

    logger.info("正在取得 F-D0047-071（1週/12小時）資料...")
    data_071 = fetch_dataset_with_cache(
        cfg.dataset_1week,
        params={
            "LocationsName": cfg.city,
            "LocationName": cfg.townships,
        },
    )
    open_meteo_weekly: Dict[str, List[DailyForecast]] = {}
    open_meteo_snapshots: Dict[str, ShortTermSnapshot] = {}
    if data_071 is None or data_069 is None:
        open_meteo_weekly, open_meteo_snapshots = fetch_open_meteo_by_town()

    if data_071 is None and not open_meteo_weekly:
        logger.error("取得 F-D0047-071 失敗，且 Open-Meteo 備援也不可用，無法產生新報表")
        logger.info("保留既有輸出檔案不變")
        return

    # 1.5) 輸出原始資料 CSV（僅 CWA 資料可輸出）
    if data_071 is not None:
        raw_071_long = build_cwa_raw_csv_rows(
            cfg.dataset_1week,
            "未來1週預報（逐12小時）",
            data_071,
            cfg.city,
            cfg.townships,
        )
        export_cwa_raw_csv(OUTPUT_DIR / "weather_raw_071_long.csv", raw_071_long)
    else:
        logger.warning("略過 weather_raw_071_long.csv，因為 F-D0047-071 改用 Open-Meteo 備援")

    if data_069 is not None:
        raw_069_long = build_cwa_raw_csv_rows(
            cfg.dataset_3day,
            "未來3天預報（逐3小時）",
            data_069,
            cfg.city,
            cfg.townships,
        )
        export_cwa_raw_csv(OUTPUT_DIR / "weather_raw_069_long.csv", raw_069_long)
    else:
        logger.warning("略過 weather_raw_069_long.csv，因為 F-D0047-069 無法取得")

    if data_071 is not None:
        raw_071_wide_rows, raw_071_wide_fields = build_cwa_raw_wide_rows(
            cfg.dataset_1week,
            "未來1週預報（逐12小時）",
            data_071,
            cfg.city,
            cfg.townships,
        )
        export_cwa_raw_wide_csv(OUTPUT_DIR / "weather_raw_071.csv", raw_071_wide_rows, raw_071_wide_fields)
    else:
        logger.warning("略過 weather_raw_071.csv，因為 F-D0047-071 改用 Open-Meteo 備援")

    if data_069 is not None:
        raw_069_wide_rows, raw_069_wide_fields = build_cwa_raw_wide_rows(
            cfg.dataset_3day,
            "未來3天預報（逐3小時）",
            data_069,
            cfg.city,
            cfg.townships,
        )
        export_cwa_raw_wide_csv(OUTPUT_DIR / "weather_raw_069.csv", raw_069_wide_rows, raw_069_wide_fields)
    else:
        logger.warning("略過 weather_raw_069.csv，因為 F-D0047-069 無法取得")

    # 2) 彙整：週報（每日） + 今日概況/風力提醒
    if data_071 is not None:
        weekly_by_town = build_weekly_daily_forecast(data_071, cfg.city, cfg.townships)
    else:
        weekly_by_town = open_meteo_weekly

    if data_069 is not None:
        snapshot_by_town = build_today_snapshot_and_wind_warning(data_069, cfg.city, cfg.townships)
    elif open_meteo_snapshots:
        snapshot_by_town = open_meteo_snapshots
    else:
        snapshot_by_town = {
            town: ShortTermSnapshot(today_desc=None, wind_warning=None)
            for town in cfg.townships
        }

    # 3) 產生輸出（每個鄉鎮一份文字報告；圖表以第一個鄉鎮為主）
    #    你若要把兩個鄉鎮都畫成圖，可擴充為多張圖或合併圖。
    if not cfg.townships:
        raise ValueError("config.json 未設定任何 townships")

    primary = cfg.townships[0]
    daily = weekly_by_town.get(primary, [])
    if not daily:
        raise RuntimeError(f"未取得 {primary} 的 1 週資料，請檢查 LocationName 是否正確")

    days = [_to_mmdd_weekday(x.d) for x in daily[:7]]
    conditions = [x.condition for x in daily[:7]]
    tmax = [x.tmax for x in daily[:7]]
    tmin = [x.tmin for x in daily[:7]]
    day_feels = [x.feel_day for x in daily[:7]]
    night_feels = [x.feel_night for x in daily[:7]]
    rain_probs = [x.pop for x in daily[:7]]
    humidities = [x.humidity for x in daily[:7]]

    # 圖表
    img_path = OUTPUT_DIR / "weather_report.png"
    generate_image_report(img_path, days, tmax, tmin, day_feels, night_feels, conditions, rain_probs, humidities)

    # 文字（每個鄉鎮一份；同名覆蓋可以自行改檔名）
    for town in cfg.townships:
        dlist = weekly_by_town.get(town, [])
        snap = snapshot_by_town.get(town, ShortTermSnapshot(today_desc=None, wind_warning=None))

        txt_path = OUTPUT_DIR / f"weather_analysis_{town}.txt"
        StructuredTextReportGenerator.generate(
            txt_path,
            township=town,
            today_desc=snap.today_desc,
            wind_warning=snap.wind_warning,
            daily=dlist,
        )

    # 兼容：仍輸出原檔名（用 primary 代表）
    (OUTPUT_DIR / "weather_analysis.txt").write_text(
        (OUTPUT_DIR / f"weather_analysis_{primary}.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    logger.info("=" * 60)
    logger.info("執行完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
