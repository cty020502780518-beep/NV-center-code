import os
import glob
import re
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")

# ================= 🔧 配置区域 =================
TRAIN_DIR = r"D:\NV数据\训练集"  # 数据源


# ===============================================

# 1. 物理模型
def double_lorentz(x, y0, A1, x1, w1, A2, x2, w2):
    L1 = A1 * (w1 ** 2) / ((x - x1) ** 2 + w1 ** 2)
    L2 = A2 * (w2 ** 2) / ((x - x2) ** 2 + w2 ** 2)
    return y0 + L1 + L2


def get_D_value(freqs, counts):
    try:
        y_max, y_min = np.max(counts), np.min(counts)
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
    return float(t_match.group(1)) if t_match else None


def read_excel_data(path):
    try:
        xls = pd.read_excel(path, sheet_name=None, engine='openpyxl')
        df = next((v for k, v in xls.items() if 'data' in k.lower()), list(xls.values())[0])
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df.iloc[:, 0].values, df.iloc[:, 1].values
    except:
        return None, None


if __name__ == '__main__':
    print("🌍 启动全局标定 (RAW模式：不剔除任何数据)...")
    files = glob.glob(os.path.join(TRAIN_DIR, "**", "*.xlsx"), recursive=True)

    all_temps = []
    all_Ds = []

    for i, fp in enumerate(files):
        if "~$" in fp: continue
        t = parse_filename(os.path.basename(fp))
        if t is None: continue

        f, c = read_excel_data(fp)
        if f is None: continue

        D = get_D_value(f, c)

        # ✨ 关键修改：只要拟合成功(D is not None)，不管多离谱都收录！
        # 移除了 '2860 < D < 2880' 的物理过滤
        if D is not None:
            all_temps.append(t)
            all_Ds.append(D)

        if (i + 1) % 500 == 0:
            print(f"   已处理 {i + 1} 个文件...")

    print(f"\n✅ 提取完成，共有 {len(all_temps)} 个数据点。")

    # 转换为 numpy 数组
    x = np.array(all_temps)
    y = np.array(all_Ds)

    # ✨ 关键修改：直接拟合，不进行 3-sigma 剔除
    slope, intercept = np.polyfit(x, y, 1)

    # 计算 R2 看看到底有多乱
    p = np.poly1d([slope, intercept])
    y_pred = p(x)
    r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

    print("\n" + "=" * 60)
    print("📉 原始全局系数 (Raw Global Coefficients)")
    print("=" * 60)
    print(f"Slope (k):     {slope:.6f}")
    print(f"Intercept (b): {intercept:.6f}")
    print(f"R2 (线性度):    {r2:.6f}")
    print("-" * 60)
    print("⚠️ 注意：这组系数包含了所有异常点（如微波源故障、激光跳变等数据）。")
    print("=" * 60)
    np.save("global_coeffs.npy", [slope, intercept])
    print("✅ 系数已保存为 'global_coeffs.npy'")