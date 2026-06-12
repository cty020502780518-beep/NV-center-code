import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
import warnings

# --- 配置区域 ---
# 必须放在最前面！解决 OMP Error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

# ✨✨✨ 修改点 1：设置全局字体为 Times New Roman ✨✨✨
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-whitegrid')

# 你的真实数据路径
TRAIN_DIR = r"D:\NV数据\训练集"


# ================= 1. 物理拟合核心 (复用标定代码) =================
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
        bounds = ([y_max - 0.1, -1.0, 2800, 0.5, -1.0, 2800, 0.5],
                  [y_max + 0.1, 0.0, 2950, 20.0, 0.0, 2950, 20.0])
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
    m = float(m_match.group(1)) if m_match else 0.0
    return float(t_match.group(1)), float(l_match.group(1)), m


def read_excel_data(path):
    try:
        xls = pd.read_excel(path, sheet_name=None, engine='openpyxl')
        df = next((v for k, v in xls.items() if 'data' in k.lower()), list(xls.values())[0])
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df.iloc[:, 0].values, df.iloc[:, 1].values
    except:
        return None, None


# ================= 2. 主程序：数据加载与绘图 =================
def main():
    print(f"🚀 正在扫描真实数据: {TRAIN_DIR} ...")
    files = glob.glob(os.path.join(TRAIN_DIR, "**", "*.xlsx"), recursive=True)

    if len(files) == 0:
        print("❌ 错误：未找到任何Excel文件！请检查路径。")
        return

    # --- Step 1: 提取真实数据的 D 值 ---
    # 结构: data_groups[(L, M)] = {'t': [], 'D': []}
    data_groups = {}

    print(f"   发现 {len(files)} 个文件，开始拟合物理中心 (D-value)...")

    for i, fp in enumerate(files):
        if "~$" in fp: continue
        if (i + 1) % 50 == 0: print(f"   已处理 {i + 1}/{len(files)}...")

        t, l, m = parse_filename(os.path.basename(fp))
        if t is None: continue

        f_raw, c_raw = read_excel_data(fp)
        if f_raw is None: continue

        D = get_D_value(f_raw, c_raw)
        if D is not None and 2860 < D < 2880:  # 简单过滤异常值
            key = (l, m)
            if key not in data_groups: data_groups[key] = {'t': [], 'D': []}
            data_groups[key]['t'].append(t)
            data_groups[key]['D'].append(D)

    print(f"✅ 数据提取完成！共 {len(data_groups)} 个功率组。")

    # --- Step 2: 准备绘图 ---
    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])

    # 🎨 Panel A: 9组真实数据的线性拟合
    ax1 = plt.subplot(gs[0])
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_groups)))

    # 用于 Panel B 的最佳展示组 (找数据点最多的那一组)
    best_group_key = None
    max_points = 0
    group_params = {}  # 存 k, b

    print("\n📊 正在绘制 Panel A (物理先验)...")
    for idx, key in enumerate(sorted(data_groups.keys())):
        item = data_groups[key]
        t_arr = np.array(item['t'])
        d_arr = np.array(item['D'])

        if len(t_arr) > max_points:
            max_points = len(t_arr)
            best_group_key = key

        # 线性拟合
        if len(t_arr) > 1:
            k, b = np.polyfit(t_arr, d_arr, 1)
            group_params[key] = (k, b)

            # 绘图
            label = f"L{int(key[0])}%/M{int(key[1])}dB"
            ax1.scatter(t_arr, d_arr, color=colors[idx], s=30, alpha=0.6, edgecolors='w')

            # 画拟合线 (延伸一点范围)
            x_line = np.linspace(35, 85, 100)
            y_line = k * x_line + b
            ax1.plot(x_line, y_line, color=colors[idx], lw=1.5, alpha=0.8, linestyle='--')

    ax1.set_title("(a) Real Physical Prior: Linear Fit of 9 Power Groups", fontsize=15, fontweight='bold')
    ax1.set_xlabel("Temperature (°C)", fontsize=13)
    ax1.set_ylabel("Resonance Frequency (MHz)", fontsize=13)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 🎨 Panel B: 真实数据增强演示 (基于最佳组)
    ax2 = plt.subplot(gs[1])

    if best_group_key:
        print(f"📊 正在绘制 Panel B (使用 {best_group_key} 组作为演示)...")
        real_t = np.array(data_groups[best_group_key]['t'])
        real_d = np.array(data_groups[best_group_key]['D'])
        k_best, b_best = group_params[best_group_key]

        # 1. 生成增强数据云 (模拟 generate_db 的逻辑)
        n_aug = 2000  # 生成 2000 个点用于展示
        t_aug = np.random.uniform(35, 85, n_aug)

        jitter = np.random.normal(0, 0.05, n_aug)
        d_aug = k_best * t_aug + b_best + jitter

        # ✨✨✨ 修改点 2：绘制增强云，使用单一颜色 'dodgerblue' 替代渐变色 ✨✨✨
        ax2.scatter(t_aug, d_aug, color='dodgerblue', s=10, alpha=0.3, label='Augmented Data Cloud')

        # 3. 绘制真实种子点
        ax2.scatter(real_t, real_d, color='red', s=60, edgecolors='k', zorder=10,
                    label=f'Real Seed Data ({len(real_t)} pts)')

        # 4. 绘制物理规律线
        x_law = np.linspace(35, 85, 100)
        y_law = k_best * x_law + b_best
        ax2.plot(x_law, y_law, 'k--', lw=2, label=f'Physical Law (Slope={k_best:.4f})')

        ax2.set_title(
            f"(b) Physics-Informed Augmentation\n(Demo: L{int(best_group_key[0])}% / M{int(best_group_key[1])}dBm)",
            fontsize=15, fontweight='bold')
        ax2.set_xlabel("Temperature (°C)", fontsize=13)
        ax2.set_ylabel("Resonance Frequency (MHz)", fontsize=13)
        ax2.legend(loc='upper right', framealpha=0.9, edgecolor='gray')
        ax2.grid(True, linestyle=':', alpha=0.5)

        # 添加注释
        ax2.text(0.05, 0.05,
                 "Algorithm:\n1. Select Real Seed\n2. Random Temp T'\n3. Shift Spectrum via $D=kT'+b$\n4. Add Jitter Noise",
                 transform=ax2.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    save_path = "真实数据加强原理图.png"
    # ✨✨✨ 修改点 3：提升分辨率到 600 DPI，让图片在电脑屏幕上极其清晰无锯齿 ✨✨✨
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"\n✅ 真实数据原理图已生成: {save_path}")


if __name__ == '__main__':
    main()