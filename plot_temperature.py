import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np

plt.rcParams['font.family'] = ['Microsoft JhengHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 讀取資料 ---
df_monthly = pd.read_csv('氣溫.csv')
df_monthly['日期'] = pd.to_datetime(df_monthly['日期'], format='%Y-%m')

df_may = pd.read_csv('5月氣溫.csv')
df_may['年份'] = df_may['日期'].str[:4].astype(int)

# --- 建立畫布 ---
fig = plt.figure(figsize=(16, 12), facecolor='#f8f9fa')
fig.suptitle('氣溫變遷趨勢分析', fontsize=20, fontweight='bold', y=0.98, color='#2c3e50')

gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
              left=0.07, right=0.97, top=0.93, bottom=0.07)

# ---- 圖1：月均溫走勢（折線帶陰影） ----
ax1 = fig.add_subplot(gs[0, :])
ax1.fill_between(df_monthly['日期'], df_monthly['最低氣溫'], df_monthly['最高氣溫'],
                 alpha=0.18, color='#e74c3c', label='最高∼最低區間')
ax1.plot(df_monthly['日期'], df_monthly['平均氣溫'], color='#2980b9',
         linewidth=2.2, marker='o', markersize=4.5, label='月平均氣溫', zorder=5)
ax1.plot(df_monthly['日期'], df_monthly['最高氣溫'], color='#e74c3c',
         linewidth=1.2, linestyle='--', alpha=0.7, label='月最高氣溫')
ax1.plot(df_monthly['日期'], df_monthly['最低氣溫'], color='#27ae60',
         linewidth=1.2, linestyle='--', alpha=0.7, label='月最低氣溫')

# 標記各年度區塊
years = sorted(df_monthly['日期'].dt.year.unique())
colors_yr = ['#ecf0f1', '#dce8f5']
for i, yr in enumerate(years):
    start = pd.Timestamp(f'{yr}-01-01')
    end = pd.Timestamp(f'{yr}-12-31')
    ax1.axvspan(max(start, df_monthly['日期'].min()),
                min(end, df_monthly['日期'].max()),
                alpha=0.12, color=colors_yr[i % 2], zorder=0)
    mid = pd.Timestamp(f'{yr}-07-01')
    if df_monthly['日期'].min() <= mid <= df_monthly['日期'].max():
        ax1.text(mid, ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else 7,
                 str(yr), ha='center', va='bottom', fontsize=8.5,
                 color='#7f8c8d', alpha=0.8)

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=40, ha='right', fontsize=8)
ax1.set_title('2023–2026 月氣溫走勢', fontsize=13, fontweight='bold', pad=8)
ax1.set_ylabel('氣溫 (°C)', fontsize=10)
ax1.set_ylim(5, 42)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.85)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.set_facecolor('#fdfdfd')

# ---- 圖2：5月歷年平均氣溫趨勢 ----
ax2 = fig.add_subplot(gs[1, 0])
z = np.polyfit(df_may['年份'], df_may['平均氣溫'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_may['年份'].min(), df_may['年份'].max(), 200)

ax2.bar(df_may['年份'], df_may['平均氣溫'], color='#3498db', alpha=0.65,
        width=0.6, label='平均氣溫')
ax2.plot(x_line, p(x_line), color='#e74c3c', linewidth=2,
         linestyle='-', label=f'趨勢線 (斜率:{z[0]:+.3f}°C/年)')
ax2.set_title('5月歷年平均氣溫 (2010–2026)', fontsize=12, fontweight='bold', pad=8)
ax2.set_xlabel('年份', fontsize=10)
ax2.set_ylabel('平均氣溫 (°C)', fontsize=10)
ax2.set_xticks(df_may['年份'])
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
ax2.set_ylim(22, 30)
ax2.legend(fontsize=9, framealpha=0.85)
ax2.grid(axis='y', linestyle='--', alpha=0.4)
ax2.set_facecolor('#fdfdfd')
for _, row in df_may.iterrows():
    ax2.text(row['年份'], row['平均氣溫'] + 0.12, f"{row['平均氣溫']:.1f}",
             ha='center', va='bottom', fontsize=7.5, color='#2c3e50')

# ---- 圖3：5月歷年最高/最低氣溫 ----
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(df_may['年份'], df_may['最高氣溫'], color='#e74c3c',
         marker='^', linewidth=2, markersize=6, label='月最高氣溫')
ax3.plot(df_may['年份'], df_may['最低氣溫'], color='#27ae60',
         marker='v', linewidth=2, markersize=6, label='月最低氣溫')
ax3.fill_between(df_may['年份'], df_may['最低氣溫'], df_may['最高氣溫'],
                 alpha=0.12, color='#9b59b6')
ax3.set_title('5月歷年最高/最低氣溫 (2010–2026)', fontsize=12, fontweight='bold', pad=8)
ax3.set_xlabel('年份', fontsize=10)
ax3.set_ylabel('氣溫 (°C)', fontsize=10)
ax3.set_xticks(df_may['年份'])
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
ax3.set_ylim(12, 42)
ax3.legend(fontsize=9, framealpha=0.85)
ax3.grid(axis='y', linestyle='--', alpha=0.4)
ax3.set_facecolor('#fdfdfd')

# ---- 存檔 ----
out_path = 'temperature_trend.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f'已儲存：{out_path}')
plt.close()
