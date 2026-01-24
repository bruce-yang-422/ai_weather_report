"""
天氣預報腳本 - 機車通勤族專用 (LINE 純文字 + 未來一週數值化版)
主要改進：
1. 未來一週：強制要求列出具體數值 (氣溫/體感/降雨%)
2. 保持 LINE 純文字格式 (無 Markdown)
3. 保持機車族風力/體感邏輯
"""

import requests
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from datetime import datetime
import numpy as np
import json
import yaml
import math
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from openai import OpenAI

# ==========================================
# 配置與常量
# ==========================================

@dataclass
class Config:
    """配置類"""
    lat: float = 25.04694511723731  # 泰山明志書院
    lon: float = 121.42667399750172
    timezone: str = "Asia/Taipei"
    font_path: str = r"C:\Windows\Fonts\msjh.ttc"
    fallback_fonts: List[str] = None
    
    def __post_init__(self):
        if self.fallback_fonts is None:
            self.fallback_fonts = ["Microsoft JhengHei", "SimHei"]
    
    @classmethod
    def load_from_yaml(cls, config_path: Path) -> 'Config':
        """從 YAML 配置檔載入設定"""
        if not config_path.exists():
            logger.warning(f"找不到配置檔 {config_path}，使用預設值")
            return cls()
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            
            location = config_data.get("location", {})
            font = config_data.get("font", {})
            
            return cls(
                lat=location.get("latitude", 25.04694511723731),
                lon=location.get("longitude", 121.42667399750172),
                timezone=location.get("timezone", "Asia/Taipei"),
                font_path=font.get("path", r"C:\Windows\Fonts\msjh.ttc"),
                fallback_fonts=font.get("fallback", ["Microsoft JhengHei", "SimHei"])
            )
        except Exception as e:
            logger.error(f"讀取配置檔失敗: {e}，使用預設值")
            return cls()
    
    @property
    def api_url(self) -> str:
        return (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={self.lat}&longitude={self.lon}"
            "&daily=temperature_2m_max,temperature_2m_min,weathercode,"
            "precipitation_probability_max,windspeed_10m_max"
            "&hourly=temperature_2m,relative_humidity_2m,windspeed_10m,shortwave_radiation"
            f"&timezone={self.timezone}"
        )

# 初始化
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
CONFIG_PATH = PROJECT_ROOT / "config" / "Weather_descriptions_API_keys.json"
SYSTEM_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'weather.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 字型設置
# ==========================================

def setup_font(config: Config) -> None:
    """設置matplotlib字型"""
    try:
        font = FontProperties(fname=config.font_path)
        plt.rcParams["font.family"] = font.get_name()
        logger.info(f"成功載入字型: {font.get_name()}")
    except Exception as e:
        logger.warning(f"無法載入指定字型，使用備用字型: {e}")
        plt.rcParams["font.sans-serif"] = config.fallback_fonts
    plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 風力警告系統
# ==========================================

class WindWarningSystem:
    """風力警告系統 - 針對機車騎士"""
    
    # 風力等級定義 (蒲福風級)
    WIND_LEVELS = [
        (39, 49, "⚠️今日有6級強風，騎經高架或路口請抓緊龍頭。"),
        (50, 61, "⚠️今日7級疾風，車身會明顯晃動,請放慢車速。"),
        (62, 88, "⛔今日8-9級烈風，極度危險！務必慢行，防範路邊倒車。"),
        (89, float('inf'), "☠️今日10級狂風，生命受威脅，強烈建議不要騎車出門。")
    ]
    
    @classmethod
    def get_warning(cls, wind_kmh: float) -> str:
        """獲取風力警告文字"""
        for min_wind, max_wind, warning in cls.WIND_LEVELS:
            if min_wind <= wind_kmh <= max_wind:
                return warning
        return ""
    
    @classmethod
    def is_dangerous(cls, wind_kmh: float) -> bool:
        """判斷風力是否達到危險等級"""
        return wind_kmh >= 39

# ==========================================
# 體感溫度計算
# ==========================================

class RealFeelCalculator:
    """真實體感溫度計算器"""
    
    @staticmethod
    def calculate_vapor_pressure(temp_c: float, rh_percent: float) -> float:
        """計算水蒸氣壓"""
        E = 6.112 * math.exp((17.67 * temp_c) / (temp_c + 243.5))
        return E * (rh_percent / 100.0)
    
    @classmethod
    def calculate_real_feel(
        cls, 
        temp_c: float, 
        rh: float, 
        wind_kmh: float, 
        radiation_wm2: float
    ) -> float:
        """
        計算真實體感溫度
        考慮：溫度、濕度、風速、太陽輻射
        """
        wind_ms = wind_kmh / 3.6
        e = cls.calculate_vapor_pressure(temp_c, rh)
        
        # 基礎體感溫度 (考慮濕度和風寒)
        base_at = temp_c + (0.33 * e) - (0.70 * wind_ms) - 4.00
        
        # 太陽輻射修正
        solar_correction = 0.0
        if radiation_wm2 > 0:
            solar_correction = (radiation_wm2 / 120.0) * (1.0 - (0.08 * wind_ms))
            solar_correction = max(solar_correction, 0.0)
        
        return base_at + solar_correction

# ==========================================
# 天氣數據處理
# ==========================================

class WeatherDataProcessor:
    """天氣數據處理器"""
    
    WEATHER_CODE_MAP = {
        0: "晴朗", 1: "晴時多雲", 2: "多雲", 3: "陰天",
        45: "霧", 48: "霧",
        51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨",
        61: "小雨", 63: "中雨", 65: "大雨",
        80: "陣雨", 81: "陣雨", 82: "強陣雨",
        95: "雷雨", 96: "雷雨", 99: "雷雨"
    }
    
    WEEKDAY_MAP = ["一", "二", "三", "四", "五", "六", "日"]
    
    @classmethod
    def get_weather_description(cls, code: int) -> str:
        """獲取天氣描述"""
        return cls.WEATHER_CODE_MAP.get(code, "未知")
    
    @classmethod
    def format_date(cls, date_str: str) -> str:
        """格式化日期為 MM-DD(週X)"""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        mmdd = dt.strftime("%m-%d")
        weekday = cls.WEEKDAY_MAP[dt.weekday()]
        return f"{mmdd}({weekday})"
    
    @staticmethod
    def fetch_weather_data(api_url: str) -> Dict:
        """獲取天氣數據"""
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            logger.info("成功獲取天氣數據")
            return response.json()
        except requests.RequestException as e:
            logger.error(f"獲取天氣數據失敗: {e}")
            raise
    
    @staticmethod
    def compute_daily_average(data: Dict, key: str) -> List[Optional[float]]:
        """計算每日平均值"""
        hours = data["hourly"]["time"]
        values = data["hourly"][key]
        
        date_buckets = {}
        for time_str, value in zip(hours, values):
            date = time_str.split("T")[0]
            date_buckets.setdefault(date, []).append(value)
        
        daily_dates = data["daily"]["time"]
        averages = []
        for date in daily_dates:
            if date in date_buckets and date_buckets[date]:
                avg = round(np.mean(date_buckets[date]), 1)
                averages.append(avg)
            else:
                averages.append(None)
        
        return averages
    
    @classmethod
    def process_real_feel_temperatures(
        cls, 
        data: Dict
    ) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """
        處理日夜體感溫度
        日間: 09:00-14:00
        夜間: 19:00-23:00
        """
        hourly = data["hourly"]
        times = hourly["time"]
        temps = hourly["temperature_2m"]
        rhs = hourly["relative_humidity_2m"]
        winds = hourly["windspeed_10m"]
        rads = hourly["shortwave_radiation"]
        
        date_buckets = {}
        
        for i in range(len(times)):
            try:
                dt = datetime.strptime(times[i], "%Y-%m-%dT%H:%M")
                date_str = dt.strftime("%Y-%m-%d")
                hour = dt.hour
                
                real_feel = RealFeelCalculator.calculate_real_feel(
                    temps[i], rhs[i], winds[i], rads[i]
                )
                
                if date_str not in date_buckets:
                    date_buckets[date_str] = {"day": [], "night": []}
                
                if 9 <= hour <= 14:
                    date_buckets[date_str]["day"].append(real_feel)
                elif 19 <= hour <= 23:
                    date_buckets[date_str]["night"].append(real_feel)
            except (ValueError, IndexError) as e:
                logger.warning(f"處理時間點數據時出錯: {e}")
                continue
        
        daily_dates = data["daily"]["time"]
        day_feels = []
        night_feels = []
        
        for date in daily_dates:
            if date in date_buckets:
                day_vals = date_buckets[date]["day"]
                day_feels.append(
                    round(np.mean(day_vals), 1) if day_vals else None
                )
                
                night_vals = date_buckets[date]["night"]
                night_feels.append(
                    round(np.mean(night_vals), 1) if night_vals else None
                )
            else:
                day_feels.append(None)
                night_feels.append(None)
        
        return day_feels, night_feels

# ==========================================
# 報表生成
# ==========================================

class WeatherReportGenerator:
    """天氣報表生成器"""
    
    @staticmethod
    def generate_image_report(
        output_path: Path,
        days: List[str],
        tmax: List[float],
        tmin: List[float],
        day_feels: List[Optional[float]],
        night_feels: List[Optional[float]],
        conditions: List[str],
        rain_probs: List[int],
        humidities: List[Optional[float]]
    ) -> None:
        """生成圖表報表"""
        
        fig, (ax_table, ax_chart) = plt.subplots(
            nrows=2, ncols=1, figsize=(12, 10),
            gridspec_kw={'height_ratios': [0.8, 1]},
            facecolor='white'
        )
        
        # --- 表格部分 ---
        ax_table.axis('off')
        ax_table.set_title(
            "未來 7 天天氣預報 (真實體感)",
            fontsize=16, 
            pad=20,
            weight='bold'
        )
        
        columns = (
            "日期", "天氣", "最高溫\n(°C)", "最低溫\n(°C)",
            "體感溫度(°C)\n日(含日曬)/夜", "降雨\n(%)", "濕度\n(%)"
        )
        
        cell_text = []
        for i in range(len(days)):
            d_feel = f"{day_feels[i]:.1f}" if day_feels[i] is not None else "-"
            n_feel = f"{night_feels[i]:.1f}" if night_feels[i] is not None else "-"
            feel_str = f"{d_feel} / {n_feel}"
            humidity_str = f"{humidities[i]:.0f}" if humidities[i] else "-"
            
            row_data = [
                days[i], conditions[i], tmax[i], tmin[i],
                feel_str, rain_probs[i], humidity_str
            ]
            cell_text.append(row_data)
        
        table = ax_table.table(
            cellText=cell_text,
            colLabels=columns,
            loc='center',
            cellLoc='center',
            bbox=[0.05, 0.1, 0.9, 0.8]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        
        # 表格樣式
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # 標題行
                cell.set_facecolor('#4A90E2')
                cell.set_text_props(weight='bold', color='white')
            elif row % 2 == 0:  # 偶數行
                cell.set_facecolor('#f9f9f9')
            
            if col == 4 and row > 0:  # 體感溫度列
                cell.set_text_props(weight='bold', color='#d62728')
        
        # --- 折線圖部分 ---
        ax_chart.set_facecolor('white')
        
        ax_chart.plot(
            days, tmax,
            marker='o', label="實際最高溫",
            color='#ff7f0e', linewidth=2.5, alpha=0.7
        )
        ax_chart.plot(
            days, tmin,
            marker='o', label="實際最低溫",
            color='#1f77b4', linewidth=2.5, alpha=0.7
        )
        ax_chart.plot(
            days, day_feels,
            marker='^', label="白天體感 (09-14時)",
            color='#d62728', linestyle='--', linewidth=2.5
        )
        ax_chart.plot(
            days, night_feels,
            marker='v', label="晚上體感 (19-00時)",
            color='#9467bd', linestyle=':', linewidth=2.5
        )
        
        ax_chart.set_xlabel("日期", fontsize=12)
        ax_chart.set_ylabel("溫度 (°C)", fontsize=12)
        ax_chart.set_title("真實體感與氣溫走勢", fontsize=14, weight='bold')
        ax_chart.grid(True, linestyle='--', alpha=0.3)
        ax_chart.legend(loc='best', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor='white', bbox_inches='tight')
        plt.close()
        
        logger.info(f"圖表已生成: {output_path}")

# ==========================================
# AI 報告生成
# ==========================================

class AIReportGenerator:
    """AI 文字報告生成器"""
    
    @staticmethod
    def load_api_config(config_path: Path) -> Tuple[Optional[str], str]:
        """載入 API 配置"""
        if not config_path.exists():
            logger.warning(f"找不到配置檔: {config_path}")
            return None, "gpt-4o-mini"
        
        try:
            # 調試：檢查檔案大小和原始內容
            file_size = config_path.stat().st_size
            logger.info(f"API 配置檔路徑: {config_path}, 檔案大小: {file_size} bytes")
            
            with open(config_path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(f"API 配置檔原始內容: {repr(content[:100])}")
                config = json.loads(content)
            
            api_key = config.get("openai_api_key")
            model = config.get("openai_model", "gpt-4o-mini")
            logger.info(f"成功載入 OpenAI API 配置，Model: {model}")
            return api_key, model
        except Exception as e:
            logger.error(f"讀取配置檔失敗: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, "gpt-4o-mini"
    
    @classmethod
    def generate_ai_descriptions(
        cls,
        api_key: str,
        model_name: str,
        days: List[str],
        conditions: List[str],
        tmax: List[float],
        tmin: List[float],
        rain_probs: List[int],
        day_feels: List[Optional[float]],
        night_feels: List[Optional[float]],
        max_winds_kmh: List[float]
    ) -> Tuple[Optional[str], List[Optional[str]], Optional[str]]:
        """
        生成 AI 描述內容（今日描述、未來一週每日描述、貼心提醒）
        返回：(今日描述, [每日描述列表], 貼心提醒)
        """
        
        # 整理數據
        weather_summary = "未來七天數據：\n"
        for i in range(len(days)):
            wind_warning = WindWarningSystem.get_warning(max_winds_kmh[i])
            wind_str = f" [注意: {wind_warning}]" if wind_warning else ""
            
            day_feel_str = f"{day_feels[i]:.1f}" if day_feels[i] is not None else "N/A"
            night_feel_str = f"{night_feels[i]:.1f}" if night_feels[i] is not None else "N/A"
            
            weather_summary += (
                f"- {days[i]}: {conditions[i]}, "
                f"氣溫 {tmin[i]}~{tmax[i]}°C, "
                f"體感(日/夜) {day_feel_str}/{night_feel_str}°C, "
                f"降雨 {rain_probs[i]}%{wind_str}\n"
            )
        
        # 構建 Prompt - 只要求生成描述內容
        # 先確定未來一週的星期名稱
        now = datetime.now()
        weekday_map = WeatherDataProcessor.WEEKDAY_MAP
        future_weekdays = []
        for i in range(1, min(8, len(days))):
            date_match = re.match(r"\d{2}-\d{2}\(([^)]+)\)", days[i])
            if date_match:
                weekday_char = date_match.group(1)
                weekday_map_dict = {"一": "週一", "二": "週二", "三": "週三", "四": "週四", 
                                  "五": "週五", "六": "週六", "日": "週日"}
                day_name = weekday_map_dict.get(weekday_char, f"週{weekday_char}")
                future_weekdays.append(day_name)
        
        future_weekdays_str = "\n".join([f"{day}：[簡短點評]" for day in future_weekdays])
        
        system_prompt = f"""你是一個專為機車通勤族服務的氣象助理。

**用戶資料**：住公寓(不用曬衣)、只能騎機車(不要建議大眾運輸)、平日上班(09/19通勤)、晚上/假日才有空。

**任務**：根據提供的天氣數據，生成簡短的描述文字。

**嚴格規則**：
1. 使用台灣慣用詞彙、語句、繁體中文。
2. 描述要簡短實用，針對機車通勤族。
3. **風力**：只有在數據中有出現 [注意: ...] 時才在描述中提到風力，否則**完全不要提風**。
4. 今日描述：一句話點評今日騎車感受。
5. 未來一週描述：每天一句簡短點評（針對該天的天氣狀況）。
6. 貼心提醒：針對整週的騎車通勤建議。

**輸出格式（必須嚴格遵守）**：
今日描述：[一句話描述今日騎車感受]

未來一週描述：
{future_weekdays_str}

貼心提醒：[針對整週的騎車通勤建議]
"""
        
        try:
            logger.info(f"正在呼叫 OpenAI API ({model_name})生成描述...")
            
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": weather_summary}
                ]
            )
            
            ai_content = response.choices[0].message.content
            
            # 解析 AI 返回的內容
            today_desc = None
            weekly_descs = []
            tips_content = None
            
            # 提取今日描述
            today_match = re.search(r"今日描述[：:]\s*(.+?)(?=\n\n|$)", ai_content, re.DOTALL)
            if today_match:
                today_desc = today_match.group(1).strip()
            
            # 提取未來一週描述
            weekly_match = re.search(r"未來一週描述[：:]?\s*\n(.*?)(?=\n\n貼心提醒|$)", ai_content, re.DOTALL)
            if weekly_match:
                weekly_lines = weekly_match.group(1).strip().split("\n")
                # 建立星期名稱到描述的映射
                weekday_desc_map = {}
                for line in weekly_lines:
                    line = line.strip()
                    if not line:
                        continue
                    # 匹配 "週X：[描述]" 格式
                    for weekday in ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]:
                        if line.startswith(weekday):
                            desc = re.sub(rf"^{weekday}[：:]\s*", "", line).strip()
                            weekday_desc_map[weekday] = desc
                            break
                
                # 根據實際日期匹配描述（未來一週從索引 1 開始）
                for i in range(1, min(8, len(days))):
                    # 從日期字串中提取星期
                    date_match = re.match(r"\d{2}-\d{2}\(([^)]+)\)", days[i])
                    if date_match:
                        weekday_char = date_match.group(1)
                        weekday_map_dict = {"一": "週一", "二": "週二", "三": "週三", "四": "週四", 
                                          "五": "週五", "六": "週六", "日": "週日"}
                        day_name = weekday_map_dict.get(weekday_char, f"週{weekday_char}")
                        # 將描述添加到對應位置（索引 i-1 因為未來一週從索引 1 開始）
                        if day_name in weekday_desc_map:
                            if i-1 < len(weekly_descs):
                                weekly_descs[i-1] = weekday_desc_map[day_name]
                            else:
                                while len(weekly_descs) < i:
                                    weekly_descs.append(None)
                                weekly_descs.append(weekday_desc_map[day_name])
            
            # 補齊不足的描述（最多 7 天）
            while len(weekly_descs) < 7:
                weekly_descs.append(None)
            
            # 提取貼心提醒
            tips_match = re.search(r"貼心提醒[：:]\s*(.+?)(?=\n\n|$)", ai_content, re.DOTALL)
            if tips_match:
                tips_content = tips_match.group(1).strip()
            
            logger.info("AI 描述生成完成")
            return today_desc, weekly_descs[:7], tips_content
            
        except Exception as e:
            logger.error(f"AI 描述生成失敗: {e}")
            return None, [], None

# ==========================================
# 結構化文字報告生成
# ==========================================

class StructuredTextReportGenerator:
    """結構化文字報告生成器（基於 YAML 範例格式）"""
    
    # 天氣圖示映射
    ICON_MAP = {
        "晴朗": "☀️",
        "晴時多雲": "🌤️",
        "多雲": "☁️",
        "陰天": "☁️",
        "霧": "🌫️",
        "毛毛雨": "🌦️",
        "小雨": "🌧️",
        "中雨": "🌧️",
        "大雨": "⛈️",
        "陣雨": "🌦️",
        "強陣雨": "⛈️",
        "雷雨": "⛈️"
    }
    
    @classmethod
    def get_icon(cls, condition: str) -> str:
        """根據天氣狀況獲取圖示"""
        for key, icon in cls.ICON_MAP.items():
            if key in condition:
                return icon
        return "☁️"  # 預設圖示
    
    
    @classmethod
    def generate_structured_text_report(
        cls,
        output_path: Path,
        today_desc: Optional[str],
        weekly_descs: List[Optional[str]],
        tips_content: Optional[str],
        days: List[str],
        conditions: List[str],
        tmax: List[float],
        tmin: List[float],
        rain_probs: List[int],
        day_feels: List[Optional[float]],
        night_feels: List[Optional[float]]
    ) -> None:
        """生成結構化文字報告（適合人類閱讀的格式）"""
        
        # 生成日期標題
        now = datetime.now()
        weekday_map = WeatherDataProcessor.WEEKDAY_MAP
        today_str = f"{now.strftime('%m-%d')}({weekday_map[now.weekday()]})"
        
        # 構建文字報告內容（適合人類閱讀的格式）
        lines = []
        lines.append(f"{today_str} 氣象日報")
        lines.append("")
        lines.append("🌤️ 今日概況")
        
        # 今日圖示
        today_icon = cls.get_icon(conditions[0]) if conditions else "☁️"
        
        # 今日氣溫
        if len(tmin) > 0 and len(tmax) > 0:
            temp_str = f"氣溫：{round(tmin[0], 1)}~{round(tmax[0], 1)}°C"
        else:
            temp_str = "氣溫：N/A"
        lines.append(temp_str)
        
        # 今日體感
        if day_feels and day_feels[0] is not None and night_feels and night_feels[0] is not None:
            feel_str = f"體感：日 {round(day_feels[0], 1)}°C / 夜 {round(night_feels[0], 1)}°C"
        elif day_feels and day_feels[0] is not None:
            feel_str = f"體感：日 {round(day_feels[0], 1)}°C / 夜 N/A"
        elif night_feels and night_feels[0] is not None:
            feel_str = f"體感：日 N/A / 夜 {round(night_feels[0], 1)}°C"
        else:
            feel_str = "體感：N/A"
        lines.append(feel_str)
        
        # 今日降雨
        if len(rain_probs) > 0:
            rain_str = f"降雨機率：{rain_probs[0]}%"
        else:
            rain_str = "降雨機率：N/A"
        lines.append(rain_str)
        
        # 今日描述
        desc = today_desc or "無特殊提醒"
        lines.append(f"{desc}")
        lines.append("")
        
        # 未來一週預報
        lines.append("📅 未來一週")
        
        weekday_names = ["週一", "週二", "週三", "週四", "週五", "週六", "週日"]
        
        for i in range(1, min(8, len(days))):
            # 從日期字串中提取 MM-DD 和星期
            date_match = re.match(r"(\d{2}-\d{2})\(([^)]+)\)", days[i])
            if date_match:
                date_str = date_match.group(1)
                weekday_char = date_match.group(2)
                # 將星期字元轉換為星期名稱
                weekday_map_dict = {"一": "週一", "二": "週二", "三": "週三", "四": "週四", 
                                   "五": "週五", "六": "週六", "日": "週日"}
                day_name = weekday_map_dict.get(weekday_char, f"週{weekday_char}")
            else:
                date_str = days[i]
                day_name = weekday_names[i % 7] if i < len(weekday_names) else f"週{(i % 7) + 1}"
            
            # 圖示
            icon = cls.get_icon(conditions[i]) if i < len(conditions) else "☁️"
            
            # 氣溫
            if i < len(tmin) and i < len(tmax):
                temp_info = f"氣溫 {round(tmin[i], 1)}-{round(tmax[i], 1)}°C"
            else:
                temp_info = "氣溫 N/A"
            
            # 體感
            if i < len(day_feels) and day_feels[i] is not None and i < len(night_feels) and night_feels[i] is not None:
                feel_info = f"體感 {round(day_feels[i], 1)}-{round(night_feels[i], 1)}°C"
            elif i < len(day_feels) and day_feels[i] is not None:
                feel_info = f"體感 {round(day_feels[i], 1)}°C"
            elif i < len(night_feels) and night_feels[i] is not None:
                feel_info = f"體感 {round(night_feels[i], 1)}°C"
            else:
                feel_info = "體感 N/A"
            
            # 降雨
            if i < len(rain_probs):
                rain_info = f"降雨 {rain_probs[i]}%"
            else:
                rain_info = "降雨 N/A"
            
            # 描述
            desc = weekly_descs[i-1] if (i-1 < len(weekly_descs) and weekly_descs[i-1]) else "無特殊提醒"
            
            # 組合成一行
            forecast_line = f"- {day_name}({date_str})：{icon} {temp_info} / {feel_info} / {rain_info}"
            if desc and desc != "無特殊提醒":
                forecast_line += f"（{desc}）"
            lines.append(forecast_line)
        
        lines.append("")
        lines.append("💡 貼心提醒")
        
        # 貼心提醒內容
        tips = tips_content or "無特殊提醒"
        for tip_line in tips.split("\n"):
            if tip_line.strip():
                lines.append(tip_line.strip())
        
        # 寫入文字文件
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logger.info(f"結構化文字報告已生成: {output_path}")
        except Exception as e:
            logger.error(f"生成結構化文字報告失敗: {e}")
            raise

# ==========================================
# 主程式
# ==========================================

def main():
    """主程式入口"""
    try:
        logger.info("=" * 50)
        logger.info("天氣預報系統啟動")
        logger.info("=" * 50)
        
        # 初始化配置（從 YAML 載入）
        config = Config.load_from_yaml(SYSTEM_CONFIG_PATH)
        logger.info(f"載入配置：位置 ({config.lat}, {config.lon}), 時區 {config.timezone}")
        setup_font(config)
        
        # 獲取天氣數據
        logger.info("正在獲取天氣數據...")
        data = WeatherDataProcessor.fetch_weather_data(config.api_url)
        
        # 處理基礎數據
        processor = WeatherDataProcessor
        days = [processor.format_date(d) for d in data["daily"]["time"]]
        tmax = data["daily"]["temperature_2m_max"]
        tmin = data["daily"]["temperature_2m_min"]
        weather_codes = data["daily"]["weathercode"]
        conditions = [processor.get_weather_description(c) for c in weather_codes]
        rain_probs = data["daily"]["precipitation_probability_max"]
        max_winds = data["daily"]["windspeed_10m_max"]
        humidities = processor.compute_daily_average(data, "relative_humidity_2m")
        
        # 計算體感溫度
        logger.info("計算體感溫度...")
        day_feels, night_feels = processor.process_real_feel_temperatures(data)
        
        # 生成圖表報告
        logger.info("生成圖表報告...")
        img_path = OUTPUT_DIR / "weather_report.png"
        WeatherReportGenerator.generate_image_report(
            img_path, days, tmax, tmin, day_feels, night_feels,
            conditions, rain_probs, humidities
        )
        
        # 生成 AI 描述內容
        api_key, model = AIReportGenerator.load_api_config(CONFIG_PATH)
        today_desc = None
        weekly_descs = []
        tips_content = None
        
        if api_key:
            logger.info("生成 AI 描述內容...")
            today_desc, weekly_descs, tips_content = AIReportGenerator.generate_ai_descriptions(
                api_key, model, days, conditions,
                tmax, tmin, rain_probs, day_feels, night_feels, max_winds
            )
        else:
            logger.warning("未設定 OpenAI API Key，跳過 AI 描述生成")
        
        # 生成結構化文字報告（基於 YAML 範例格式，輸出為 .txt）
        logger.info("生成結構化文字報告（對應 YAML 範例格式）...")
        structured_txt_path = OUTPUT_DIR / "weather_analysis.txt"
        StructuredTextReportGenerator.generate_structured_text_report(
            structured_txt_path, today_desc, weekly_descs, tips_content,
            days, conditions, tmax, tmin, rain_probs, day_feels, night_feels
        )
        
        logger.info("=" * 50)
        logger.info("天氣預報系統執行完成")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"執行過程中發生錯誤: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()