import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

# 忽略拟合警告
warnings.filterwarnings("ignore")

# ================= 🔧 配置区域 =================
TRAIN_DIR = r"D:\NV数据\训练集"  # 你的原始数据路径
SAVE_FILE = "全功率组_温度系数标定.csv"  # 结果保存路径


# ===============================================

# 1. 物理模型：双洛伦兹
def double_lorentz(x, y0, A1, x1, w1, A2, x2, w2):
    L1 = A1 * (w1 ** 2) / ((x - x1) ** 2 + w1 ** 2)
    L2 = A2 * (w2 ** 2) / ((x - x2) ** 2 + w2 ** 2)
    return y0 + L1 + L2


def get_D_value(freqs, counts):
    try:
        y_max = np.max(counts)
        y_min = np.min(counts)
        depth = y_min - y_max

        idx_min = np.argmin(counts)
        x_center = freqs[idx_min]

        p0 = [y_max, depth / 2, x_center - 2, 3.0, depth / 2, x_center + 2, 3.0]
        bounds = ([y_max - 0.05, -1.0, 2800, 0.5, -1.0, 2800, 0.5],
                  [y_max + 0.05, 0.0, 2950, 20.0, 0.0, 2950, 20.0])
        popt, _ = curve_fit(double_lorentz, freqs, counts, p0=p0, bounds=bounds, maxfev=2000)
        return (popt[2] + popt[5]) / 2.0
    except:
        return None


def parse_filename(filename):
    filename = filename.replace("（", "(").replace("）", ")").replace(" ", "")
    t_match = re.search(r'(\d+(\.\d+)?)[°℃]', filename)
    l_match = re.search(r'(\d+)%', filename)
    m_match = re.search(r'([-]?\d+)dbm', filename, re.IGNORECASE)
    if not t_match or not l_match: return None, None, None
    t = float(t_match.group(1))
    l = float(l_match.group(1))
    m = float(m_match.group(1)) if m_match else 0.0
    return t, l, m


def read_excel_data(path):
    try:
        xls = pd.read_excel(path, sheet_name=None, engine='openpyxl')
        df = next((v for k, v in xls.items() if 'data' in k.lower()), list(xls.values())[0])
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df.iloc[:, 0].values, df.iloc[:, 1].values
    except:
        return None, None


if __name__ == '__main__':
    # ✨✨✨ 全局锁定字体为 Times New Roman ✨✨✨
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    print("🚀 正在扫描真实训练集数据，准备标定...")
    files = glob.glob(os.path.join(TRAIN_DIR, "**", "*.xlsx"), recursive=True)

    data_groups = {}
    for i, fp in enumerate(files):
        if "~$" in fp: continue
        fname = os.path.basename(fp)
        t, l, m = parse_filename(fname)
        if t is None: continue

        f_raw, c_raw = read_excel_data(fp)
        if f_raw is None: continue

        D = get_D_value(f_raw, c_raw)
        if D is not None:
            key = (l, m)
            if key not in data_groups: data_groups[key] = {'t': [], 'D': []}
            data_groups[key]['t'].append(t)
            data_groups[key]['D'].append(D)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    results = []
    plot_idx = 0

    print("📊 开始物理双峰拟合与温度系数解算...")

    for key in sorted(data_groups.keys()):
        item = data_groups[key]
        t_arr = np.array(item['t'])
        d_arr = np.array(item['D'])

        if len(t_arr) < 2: continue

        # 线性拟合
        slope, intercept = np.polyfit(t_arr, d_arr, 1)
        d_pred = slope * t_arr + intercept

        ss_res = np.sum((d_arr - d_pred) ** 2)
        ss_tot = np.sum((d_arr - np.mean(d_arr)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        results.append({
            'Laser (%)': key[0], 'MW (dBm)': key[1],
            'Slope': slope, 'Intercept': intercept, 'R2': r2
        })

        if plot_idx < 9:
            ax = axes[plot_idx]

            # 画数据散点
            ax.scatter(t_arr, d_arr, s=30, alpha=0.7, edgecolors='blue', facecolors='none', linewidths=1.2,
                       label='Data')

            # 画拟合红线
            ax.plot(t_arr, d_pred, 'r-', linewidth=2.5, label=f'Fit: R²={r2:.5f}')

            # 第二行空数据图例装载 Slope
            ax.plot([], [], linestyle='', label=f'Slope: {slope:.6f}')

            ax.set_title(f"L:{key[0]}% / MW:{key[1]}dBm", fontsize=16, fontweight='bold')

            # 带有度数符号的坐标轴
            ax.set_xlabel("Temperature (°C)", fontsize=13, fontweight='bold')
            ax.set_ylabel("Resonance Frequency (MHz)", fontsize=13, fontweight='bold')

            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(fontsize=11)
            plot_idx += 1

    plt.tight_layout()

    # ✨✨✨ 核心修改：保存为顶级期刊推荐的无损矢量图 .svg 格式 ✨✨✨
    plt.savefig("全功率组标定_验证图.svg", format='svg', bbox_inches='tight')
    print("✅ 验证图已生成并保存为：全功率组标定_验证图.svg (矢量图格式)")

    df_res = pd.DataFrame(results)
    df_res.to_csv(SAVE_FILE, index=False)
    print(f"✅ 九组标定系数已成功计算并保存至 {SAVE_FILE}")