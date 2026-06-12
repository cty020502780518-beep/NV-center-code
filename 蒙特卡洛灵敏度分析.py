import os

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
# 全局真实测试集目录 (代码将读取所有样本求平均稳定性，绝不 Cherry-pick)
REAL_TEST_DIR = r"D:\NV数据\测试集"
GLOBAL_COEFF_FILE = "global_coeffs.npy"

# 备用参数
BACKUP_K = -0.090394
BACKUP_B = 2872.187615

FREQ_AXIS = np.linspace(2858, 2878, 400)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型配置
MODELS_CFG = {
    'Blind': {'path': 'model_blind.pth', 'aux_dim': 0, 'col': None, 'color': '#FFA500'},
    'MW Only': {'path': 'model_mw.pth', 'aux_dim': 1, 'col': 1, 'color': '#9370DB'},
    'Laser Only': {'path': 'model_laser.pth', 'aux_dim': 1, 'col': 0, 'color': '#1E90FF'},
    'Full Model': {'path': 'model_full.pth', 'aux_dim': 2, 'col': None, 'color': '#228B22'}
}


# ===============================================

# --- 1. 网络定义 ---
class DualStreamNet(nn.Module):
    def __init__(self, aux_dim=2):
        super(DualStreamNet, self).__init__()
        self.shape_conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, 2, 3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten()
        )
        self.amp_fc = nn.Sequential(nn.Linear(400, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU())
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


# --- 2. 传统方法定义 ---
def single_lorentz(x, y0, A, x0, w):
    return y0 + A * (w ** 2) / ((x - x0) ** 2 + w ** 2)


def fit_single_lorentz(f, c):
    try:
        y_max, y_min = np.max(c), np.min(c)
        depth = y_min - y_max
        f_dip = f[np.argmin(c)]
        p0 = [y_max, depth, f_dip, 3.0]
        bounds = ([y_max - 0.05, -1.0, 2800, 0.5], [y_max + 0.05, 0.0, 2950, 20.0])
        popt, _ = curve_fit(single_lorentz, f, c, p0=p0, bounds=bounds, maxfev=1500)
        return popt[2]
    except:
        return None


def double_lorentz(x, y0, A1, x1, w1, A2, x2, w2):
    L1 = A1 * (w1 ** 2) / ((x - x1) ** 2 + w1 ** 2)
    L2 = A2 * (w2 ** 2) / ((x - x2) ** 2 + w2 ** 2)
    return y0 + L1 + L2


def fit_double_lorentz(f, c):
    try:
        y_max, y_min = np.max(c), np.min(c)
        depth = y_min - y_max
        idx_min = np.argmin(c)
        f_dip = f[idx_min]
        p0 = [y_max, depth / 2, f_dip - 2, 3.0, depth / 2, f_dip + 2, 3.0]
        bounds = ([y_max - 0.05, -1.0, 2800, 0.5, -1.0, 2800, 0.5], [y_max + 0.05, 0.0, 2950, 20.0, 0.0, 2950, 20.0])
        popt, _ = curve_fit(double_lorentz, f, c, p0=p0, bounds=bounds, maxfev=1500)
        return (popt[2] + popt[5]) / 2.0
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


# ================= 主程序 =================
if __name__ == '__main__':
    # ✨✨✨ 全局设置字体为新罗马 ✨✨✨
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    print("🎲 启动【全局均值】蒙特卡洛鲁棒性分析 (Global Monte Carlo)...")

    # 1. 自动寻找并读取所有真实测试集数据
    files = glob.glob(os.path.join(REAL_TEST_DIR, "**", "*.xlsx"), recursive=True)
    if not files:
        print("❌ 找不到Excel文件")
        exit()

    all_samples = []
    for fp in files:
        fname = os.path.basename(fp)
        t_real, l, m = parse_filename(fname)
        if t_real is None: continue
        f_raw, c_raw = read_excel_data(fp)
        if f_raw is not None:
            c_base = np.interp(FREQ_AXIS, f_raw, c_raw)
            all_samples.append({'c_base': c_base, 'l': l, 'm': m, 't': t_real})

    print(f"📄 成功加载 {len(all_samples)} 个真实全局样本。")

    # 2. 加载模型 & 标定
    if os.path.exists(GLOBAL_COEFF_FILE):
        std_k, std_b = np.load(GLOBAL_COEFF_FILE)
    else:
        std_k, std_b = BACKUP_K, BACKUP_B

    models = {}
    for name, cfg in MODELS_CFG.items():
        if os.path.exists(cfg['path']):
            net = DualStreamNet(aux_dim=cfg['aux_dim']).to(DEVICE)
            net.load_state_dict(torch.load(cfg['path'], map_location=DEVICE))
            net.eval()
            models[name] = {'net': net, 'cfg': cfg}

    # 3. 蒙特卡洛全局模拟循环
    noise_levels = np.linspace(0, 0.02, 20)
    REPEAT = 30  # 每个样本每个噪声点重复 30 次

    contestants = ['Single Lorentz', 'Double Lorentz'] + list(models.keys())
    # 存储最终每个模型在各个噪声等级下的【全局平均标准差】
    global_results_std = {name: [] for name in contestants}

    print(f"🚀 开始模拟注入噪声 (共 {len(noise_levels)} 级，将计算所有样本均值)...")

    for sigma in noise_levels:
        # 记录当前噪声等级下，所有样本的标准差
        sample_stds = {name: [] for name in contestants}

        for sample in all_samples:
            c_base = sample['c_base']
            l, m = sample['l'], sample['m']

            # 批量生成噪声数据 (REPEAT, 400)
            noise = np.random.normal(0, sigma, size=(REPEAT, 400))
            c_noisy_batch = c_base + noise

            # AI 模型并行批量预测 (瞬间完成)
            x_tensor = torch.tensor(c_noisy_batch, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            full_aux = torch.tensor([l / 100.0, m / 5.0], dtype=torch.float32).repeat(REPEAT, 1).to(DEVICE)

            preds = {name: [] for name in contestants}

            with torch.no_grad():
                for name, item in models.items():
                    cfg = item['cfg']
                    if cfg['aux_dim'] == 0:
                        aux_in = full_aux
                    elif cfg['aux_dim'] == 1:
                        aux_in = full_aux[:, cfg['col']:cfg['col'] + 1]
                    else:
                        aux_in = full_aux

                    p_batch = item['net'](x_tensor, aux_in).cpu().numpy().flatten() * 60.0 + 30.0
                    preds[name] = p_batch

            # 传统方法预测 (循环跑)
            for i in range(REPEAT):
                c_noisy = c_noisy_batch[i]
                d_single = fit_single_lorentz(FREQ_AXIS, c_noisy)
                preds['Single Lorentz'].append((d_single - std_b) / std_k if d_single else np.nan)

                d_double = fit_double_lorentz(FREQ_AXIS, c_noisy)
                preds['Double Lorentz'].append((d_double - std_b) / std_k if d_double else np.nan)

            # 计算这一个样本在这个噪声下的不确定度 (Std)
            for name in contestants:
                valid_preds = [x for x in preds[name] if not np.isnan(x)]
                if len(valid_preds) > 1:
                    sample_stds[name].append(np.std(valid_preds))

        # 将所有样本的 Std 求平均，作为该噪声等级的全局代表值
        for name in contestants:
            if len(sample_stds[name]) > 0:
                global_results_std[name].append(np.mean(sample_stds[name]))
            else:
                global_results_std[name].append(np.nan)

    # ================= 优化后的全局大图 =================
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(10, 7))

    # 1. 绘制主图 (加入极其清晰的图注区分)
    ax.plot(noise_levels, global_results_std['Single Lorentz'], 'x:', label='Single Lorentz (Underfitted, High Bias)',
            color='#A0A0A0', linewidth=1.5, alpha=0.8)
    ax.plot(noise_levels, global_results_std['Double Lorentz'], 'o--', label='Double Lorentz (Overfitted, Low Bias)',
            color='#505050', linewidth=1.5, alpha=0.8)

    for name in models.keys():
        cfg = MODELS_CFG[name]
        ax.plot(noise_levels, global_results_std[name], label=name, color=cfg['color'],
                linewidth=2.5 if 'Full' in name else 1.5)

    ax.set_xlabel('Injected Noise Level ($\sigma_{noise}$)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Mean Prediction Uncertainty ($\sigma_{temp}$, °C)', fontweight='bold', fontsize=14)
    ax.set_title('Global Noise Robustness Analysis (Monte Carlo across All Data)', pad=20, fontsize=16,
                 fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', fontsize=11, frameon=True, framealpha=0.9)

    # 标注文字
    ax.text(noise_levels[-1], global_results_std['Double Lorentz'][-1], ' Breakdown', ha='right', va='bottom',
            color='#505050', fontsize=12, fontweight='bold')
    ax.text(noise_levels[-1], global_results_std['Full Model'][-1] - 1, ' Stable', ha='right', va='top', color='green',
            fontsize=12, fontweight='bold')

    # --- 2. 添加局部放大图 (Inset) ---
    axins = ax.inset_axes([0.4, 0.4, 0.45, 0.45])

    for name in models.keys():
        cfg = MODELS_CFG[name]
        axins.plot(noise_levels, global_results_std[name], color=cfg['color'], linewidth=2.5 if 'Full' in name else 1.5)

    axins.set_xlim(0, 0.02)
    axins.set_ylim(0, 5)
    axins.set_title('Zoom-in: AI Models Only', fontsize=12, fontweight='bold')
    axins.grid(True, linestyle=':', alpha=0.5)

    rect = Rectangle((0, 0), 0.02, 5, linewidth=1, edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()
    plt.savefig('全局_蒙特卡洛灵敏度分析图.png', dpi=600)
    print("✅ 全局均值蒙特卡洛图已生成，无懈可击！")