import os

# --- 必须放在最前面！解决 OMP Error ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import glob
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")

# ================= 🔧 配置区域 =================
REAL_TEST_DIR = r"D:\NV数据\测试集"
GLOBAL_COEFF_FILE = "global_coeffs.npy"
# 备用参数 (如果没有生成标定文件，用这个)
BACKUP_K = -0.091205
BACKUP_B = 2872.236864

FREQ_AXIS = np.linspace(2858, 2878, 400)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型配置
MODELS_CFG = {
    'Blind': {'path': 'model_blind.pth', 'aux_dim': 0, 'col': None, 'color': 'orange'},
    'MW Only': {'path': 'model_mw.pth', 'aux_dim': 1, 'col': 1, 'color': 'purple'},
    'Laser Only': {'path': 'model_laser.pth', 'aux_dim': 1, 'col': 0, 'color': 'dodgerblue'},
    'Full Model': {'path': 'model_full.pth', 'aux_dim': 2, 'col': None, 'color': 'forestgreen'}
}


# ===============================================

# --- 1. 网络结构定义 ---
class DualStreamNet(nn.Module):
    def __init__(self, aux_dim=2):
        super(DualStreamNet, self).__init__()
        self.shape_conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, 2, 3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )
        self.amp_fc = nn.Sequential(
            nn.Linear(400, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU()
        )
        self.aux_dim = aux_dim
        if self.aux_dim > 0:
            self.aux_fc = nn.Sequential(nn.Linear(aux_dim, 16), nn.ReLU())
            fusion_dim = 64 + 32 + 16
        else:
            self.aux_fc = None
            fusion_dim = 64 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x, aux):
        x_min = x.min(dim=2, keepdim=True)[0]
        x_max = x.max(dim=2, keepdim=True)[0]
        x_shape = (x - x_min) / (x_max - x_min + 1e-8) - 0.5
        feat_shape = self.shape_conv(x_shape)
        x_flat = x.view(x.size(0), -1)
        feat_amp = self.amp_fc(x_flat)
        if self.aux_dim > 0:
            feat_aux = self.aux_fc(aux)
            combined = torch.cat((feat_shape, feat_amp, feat_aux), dim=1)
        else:
            combined = torch.cat((feat_shape, feat_amp), dim=1)
        return self.fusion(combined)


# --- 2. 传统拟合算法 (包含单、双洛伦兹) ---

# 新增：单洛伦兹函数
def single_lorentz(x, y0, A, x0, w):
    return y0 + A * (w ** 2) / ((x - x0) ** 2 + w ** 2)

def single_lorentz_fitting(f, c):
    try:
        y_max, y_min = np.max(c), np.min(c)
        depth = y_min - y_max
        f_dip = f[np.argmin(c)]
        p0 = [y_max, depth, f_dip, 3.0]
        bounds = ([y_max - 0.05, -1.0, 2800, 0.5],
                  [y_max + 0.05, 0.0, 2950, 20.0])
        popt, _ = curve_fit(single_lorentz, f, c, p0=p0, bounds=bounds, maxfev=2000)
        return popt[2]  # 单洛伦兹直接返回中心 x0
    except:
        return None

# 原有：双洛伦兹函数
def double_lorentz(x, y0, A1, x1, w1, A2, x2, w2):
    L1 = A1 * (w1 ** 2) / ((x - x1) ** 2 + w1 ** 2)
    L2 = A2 * (w2 ** 2) / ((x - x2) ** 2 + w2 ** 2)
    return y0 + L1 + L2

def double_lorentz_fitting(f, c):
    try:
        y_max, y_min = np.max(c), np.min(c)
        depth = y_min - y_max
        idx_min = np.argmin(c)
        f_dip = f[idx_min]
        p0 = [y_max, depth / 2, f_dip - 2, 3.0, depth / 2, f_dip + 2, 3.0]
        bounds = ([y_max - 0.05, -1.0, 2800, 0.5, -1.0, 2800, 0.5],
                  [y_max + 0.05, 0.0, 2950, 20.0, 0.0, 2950, 20.0])
        popt, _ = curve_fit(double_lorentz, f, c, p0=p0, bounds=bounds, maxfev=2000)
        return (popt[2] + popt[5]) / 2.0  # 返回双峰中心点
    except:
        return None


# --- 3. 工具函数 ---
def parse_filename(filename):
    filename = filename.replace("（", "(").replace("）", ")").replace(" ", "")
    t_match = re.search(r'(\d+(\.\d+)?)[°℃]', filename)
    l_match = re.search(r'(\d+)%', filename)
    m_match = re.search(r'([-]?\d+)dbm', filename, re.IGNORECASE)
    if not t_match: return None, None, None
    t = float(t_match.group(1))
    l = float(l_match.group(1)) if l_match else 50.0
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


# ================= 🚀 主程序逻辑 =================
if __name__ == '__main__':
    # ✨ 设置全局字体为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    print("🛡️ 启动稳定性与可靠性深度分析...")

    # 1. 加载标定参数
    if os.path.exists(GLOBAL_COEFF_FILE):
        std_k, std_b = np.load(GLOBAL_COEFF_FILE)
        print(f"📍 读取标定参数: b={std_b:.4f}, k={std_k:.4f}")
    else:
        std_k, std_b = BACKUP_K, BACKUP_B
        print(f"⚠️ 使用备用参数: b={std_b:.4f}, k={std_k:.4f}")

    # 2. 加载模型
    models = {}
    for name, cfg in MODELS_CFG.items():
        if os.path.exists(cfg['path']):
            net = DualStreamNet(aux_dim=cfg['aux_dim']).to(DEVICE)
            net.load_state_dict(torch.load(cfg['path'], map_location=DEVICE))
            net.eval()
            models[name] = {'net': net, 'cfg': cfg}
            print(f"✅ Loaded: {name}")

    # 3. 扫描数据
    files = glob.glob(os.path.join(REAL_TEST_DIR, "**", "*.xlsx"), recursive=True)
    results = []

    print("-" * 60)
    print("正在计算所有测试样本误差...")

    with torch.no_grad():
        for fp in files:
            if "~$" in fp: continue
            fname = os.path.basename(fp)
            t_real, l, m = parse_filename(fname)
            if t_real is None: continue
            f_raw, c_raw = read_excel_data(fp)
            if f_raw is None: continue

            c_interp = np.interp(FREQ_AXIS, f_raw, c_raw)
            x_tensor = torch.tensor(c_interp, dtype=torch.float32).view(1, 1, 400).to(DEVICE)
            full_aux = torch.tensor([l / 100.0, m / 5.0], dtype=torch.float32).view(1, 2).to(DEVICE)

            row = {'Temp': t_real}

            # --- 单双洛伦兹算法提取 ---
            D_single = single_lorentz_fitting(FREQ_AXIS, c_interp)
            row['Single Lorentz'] = (D_single - std_b) / std_k if D_single else np.nan

            D_double = double_lorentz_fitting(FREQ_AXIS, c_interp)
            row['Double Lorentz'] = (D_double - std_b) / std_k if D_double else np.nan

            # AI
            for name, item in models.items():
                cfg = item['cfg']
                if cfg['aux_dim'] == 0:
                    aux_in = full_aux
                elif cfg['aux_dim'] == 1:
                    aux_in = full_aux[:, cfg['col']:cfg['col'] + 1]
                else:
                    aux_in = full_aux
                pred = item['net'](x_tensor, aux_in).item() * 60.0 + 30.0
                row[name] = pred

            results.append(row)

    df = pd.DataFrame(results).dropna()
    print(f"📊 统计样本数: {len(df)}")

    # ================= 📊 核心：计算稳定性指标 =================

    THRESHOLD = 0.5
    # 更新选手名单，区分单双洛伦兹
    contestants = ['Single Lorentz', 'Double Lorentz'] + list(models.keys())

    print("\n" + "=" * 95)
    print(f"{'Model':<16} | {'Avg Err(C)':<12} | {'Max Err (Stability)':<22} | {'Pass Rate (<0.5C)':<20} | {'Win Rate (vs Double)'}")
    print("-" * 95)

    for name in contestants:
        if name not in df.columns: continue

        errors = np.abs(df[name] - df['Temp'])
        mae = np.mean(errors)
        max_err = np.max(errors)
        pass_rate = np.mean(errors < THRESHOLD) * 100

        # 以双洛伦兹作为传统方法的对比基准算胜率
        if 'Lorentz' in name:
            win_rate = 0.0
        else:
            trad_errs = np.abs(df['Double Lorentz'] - df['Temp'])
            win_rate = np.mean(errors < trad_errs) * 100

        print(f"{name:<16} | {mae:<12.4f} | {max_err:<22.4f} | {pass_rate:<20.1f}% | {win_rate:.1f}%")

    print("=" * 95)

    # ================= 📈 画终极图：REC 曲线 (CDF) =================
    try:
        plt.style.use('ggplot')
    except:
        pass

    plt.figure(figsize=(10, 7))

    thresholds = np.linspace(0, 2.0, 200)

    # 为两种传统方法和各个AI模型分配颜色和线型
    colors = {'Single Lorentz': 'gray', 'Double Lorentz': 'black', 'Blind': 'orange', 'MW Only': 'purple', 'Laser Only': 'dodgerblue', 'Full Model': 'forestgreen'}
    styles = {'Single Lorentz': ':', 'Double Lorentz': '--', 'Blind': '-', 'MW Only': '-', 'Laser Only': '-', 'Full Model': '-'}
    widths = {'Single Lorentz': 2, 'Double Lorentz': 2, 'Blind': 2, 'MW Only': 2, 'Laser Only': 3, 'Full Model': 3}

    for name in contestants:
        if name not in df.columns: continue

        errors = np.abs(df[name] - df['Temp'])
        pass_rates = [np.mean(errors < t) * 100 for t in thresholds]

        plt.plot(thresholds, pass_rates, label=name, color=colors.get(name, 'black'),
                 linestyle=styles.get(name, '-'), linewidth=widths.get(name, 2))

    plt.xlabel('Error Tolerance Threshold (°C)', fontsize=14)
    plt.ylabel('Success Rate (Percentage of Samples)', fontsize=14)
    plt.title('Reliability Analysis: Error CDF Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, 2.0)
    plt.ylim(0, 105)

    plt.axhline(90, color='red', linestyle=':', alpha=0.5)
    plt.text(0.1, 92, 'Industrial Standard (90% Pass)', color='red', fontsize=10)

    plt.tight_layout()
    plt.savefig('模型稳定性图.png', dpi=600)  # 顺便把分辨率提升到了 600
    print("\n✅ 终极稳定性图表已生成: 模型稳定性图.png")