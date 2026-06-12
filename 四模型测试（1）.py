import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import random
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

# 忽略拟合时的运行时警告
warnings.filterwarnings("ignore")

# ================= 🔧 用户配置区域 =================
# 1. 真实数据文件夹路径 (用来当做前景测试集和柱状图数据)
REAL_TEST_DIR = r"D:\NV数据\测试集"

# 2. 虚拟数据集文件路径 (这俩会合并，然后随机抽适量点云)
TRAIN_PTH = "训练集_50k_aligned.pth"
TEST_PTH = "测试集_5k_aligned.pth"

# 3. 频率轴与标定参数
FREQ_AXIS = np.linspace(2858, 2878, 400)
GLOBAL_COEFF_FILE = "global_coeffs.npy"
if os.path.exists(GLOBAL_COEFF_FILE):
    MY_K, MY_D0 = np.load(GLOBAL_COEFF_FILE)
    print(f"📍 成功读取标定参数: k={MY_K:.4f}, b={MY_D0:.4f}")
else:
    print("⚠️ 未找到标定文件，使用默认值")
    MY_D0 = 2872.2368
    MY_K = -0.0912

# 4. 模型路径配置 (保留单激光和全模型)
MODELS_CFG = {
    'Laser Only': {'path': 'model_laser.pth', 'aux_dim': 1, 'col': 0},
    'Full Model': {'path': 'model_full.pth', 'aux_dim': 2, 'col': None}
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================

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


# --- 2. 单双洛伦兹拟合算法 ---
def single_lorentz(x, y0, A, x0, w):
    return y0 + A * (w ** 2) / ((x - x0) ** 2 + w ** 2)


def fit_single_lorentz(freqs, contrast):
    try:
        y_max, y_min = np.max(contrast), np.min(contrast)
        depth = y_min - y_max
        f_dip = freqs[np.argmin(contrast)]
        p0 = [y_max, depth, f_dip, 3.0]
        bounds = ([y_max - 0.05, -1.0, 2800, 0.5],
                  [y_max + 0.05, 0.0, 2950, 20.0])
        popt, _ = curve_fit(single_lorentz, freqs, contrast, p0=p0, bounds=bounds, maxfev=2000)
        res_freq = popt[2]
        return (res_freq - MY_D0) / MY_K
    except:
        return np.nan


def double_lorentz(x, y0, A1, x1, w1, A2, x2, w2):
    L1 = A1 * (w1 ** 2) / ((x - x1) ** 2 + w1 ** 2)
    L2 = A2 * (w2 ** 2) / ((x - x2) ** 2 + w2 ** 2)
    return y0 + L1 + L2


def fit_double_lorentz(freqs, contrast):
    try:
        y_max, y_min = np.max(contrast), np.min(contrast)
        depth = y_min - y_max
        idx_min = np.argmin(contrast)
        f_dip = freqs[idx_min]
        p0 = [y_max, depth / 2, f_dip - 2, 3.0, depth / 2, f_dip + 2, 3.0]
        bounds = ([y_max - 0.05, -1.0, 2800, 0.5, -1.0, 2800, 0.5],
                  [y_max + 0.05, 0.0, 2950, 20.0, 0.0, 2950, 20.0])
        popt, _ = curve_fit(double_lorentz, freqs, contrast, p0=p0, bounds=bounds, maxfev=2000)
        res_freq = (popt[2] + popt[5]) / 2.0
        return (res_freq - MY_D0) / MY_K
    except:
        return np.nan


# --- 3. 数据解析工具 ---
def load_and_parse_pth(file_path):
    data = torch.load(file_path, map_location='cpu')
    parsed = []
    if hasattr(data, 'tensors'):
        x_all, aux_all, y_all = data.tensors
        for i in range(len(x_all)): parsed.append((x_all[i], aux_all[i], y_all[i]))
    elif isinstance(data, dict):
        if 'X' in data and 'Aux' in data and 'Y' in data:
            x_all, aux_all, y_all = data['X'], data['Aux'], data['Y']
        else:
            keys = list(data.keys())
            x_all, aux_all, y_all = data[keys[0]], data[keys[1]], data[keys[2]]
        for i in range(len(x_all)): parsed.append((x_all[i], aux_all[i], y_all[i]))
    elif isinstance(data, list):
        parsed = data
    return parsed


def unpack_sample(sample):
    x, aux, y = sample if isinstance(sample, (tuple, list)) else (sample['x'], sample['aux'], sample['y'])
    if isinstance(x, torch.Tensor):
        x_tensor = x.clone().detach().view(1, 1, 400).to(DEVICE)
    else:
        x_tensor = torch.tensor(np.array(x).squeeze(), dtype=torch.float32).view(1, 1, 400).to(DEVICE)

    if isinstance(aux, torch.Tensor):
        aux_tensor = aux.clone().detach().view(1, 2).to(DEVICE)
    else:
        aux_tensor = torch.tensor(aux, dtype=torch.float32).view(1, 2).to(DEVICE)

    if aux_tensor[0, 0].item() > 2.0:  aux_tensor[0, 0] = aux_tensor[0, 0] / 100.0
    if abs(aux_tensor[0, 1].item()) > 2.0:  aux_tensor[0, 1] = aux_tensor[0, 1] / 5.0

    t_real = y.item() if hasattr(y, 'item') else float(y)
    if t_real <= 1.5:  t_real = t_real * 60.0 + 30.0
    return x_tensor, aux_tensor, t_real


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


# ================= 主程序逻辑 =================
if __name__ == '__main__':
    # ✨ 全局字体锁定为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    print("⚔️ 全局大考启动：全真实数据集评估 + 适度背景点云辅助...")
    print("-" * 60)

    # 1. 加载模型
    models = {}
    for name, cfg in MODELS_CFG.items():
        if not os.path.exists(cfg['path']):
            print(f"❌ 警告：找不到权重文件 {cfg['path']}，跳过该模型。")
            continue
        net = DualStreamNet(aux_dim=cfg['aux_dim']).to(DEVICE)
        state_dict = torch.load(cfg['path'], map_location=DEVICE)
        net.load_state_dict(state_dict)
        net.eval()
        models[name] = {'net': net, 'cfg': cfg}
        print(f"✅ 已加载模型: {name}")

    if not models:
        print("❌ 没有加载到任何模型，程序终止。")
        exit()

    # ✨ 2. 混合 5.5w 虚拟数据并提取【适量点云】作为背景
    train_cloud_real = []
    train_cloud_pred = []
    combined_synth_data = []

    print("\n☁️ 正在读取 5.5w 虚拟数据集，抽取适量背景点云...")
    if os.path.exists(TRAIN_PTH): combined_synth_data.extend(load_and_parse_pth(TRAIN_PTH))
    if os.path.exists(TEST_PTH): combined_synth_data.extend(load_and_parse_pth(TEST_PTH))

    if combined_synth_data:
        # 自由发挥：随机抽取 1200 个点，不限制组别，反映模型全局理论能力，又不会密密麻麻
        sample_size = min(1200, len(combined_synth_data))
        sampled_synth = random.sample(combined_synth_data, sample_size)

        full_model_info = models.get('Full Model')
        if full_model_info:
            net_full = full_model_info['net']
            with torch.no_grad():
                for item in sampled_synth:
                    x_tensor, aux_in, t_real = unpack_sample(item)
                    pred_norm = net_full(x_tensor, aux_in)
                    pred_temp = pred_norm.item() * 60.0 + 30.0
                    train_cloud_real.append(t_real)
                    train_cloud_pred.append(pred_temp)

        print(f"✅ 背景点云准备就绪 (抽取数量: {len(train_cloud_real)})。")
    else:
        print(f"⚠️ 未找到增强数据集文件，跳过背景点云的生成。")

    # ✨ 3. 扫描【所有】真实 Excel 实验数据 (无过滤！找回原本 1.92 的误差)
    records = []
    if os.path.exists(REAL_TEST_DIR):
        print(f"\n📂 正在扫描全局真实实验数据 ({REAL_TEST_DIR})...")
        excel_files = glob.glob(os.path.join(REAL_TEST_DIR, "**", "*.xlsx"), recursive=True)

        with torch.no_grad():
            for i, fp in enumerate(excel_files):
                if "~$" in fp: continue
                fname = os.path.basename(fp)
                t_real, l, m = parse_filename(fname)
                if t_real is None: continue

                f_raw, c_raw = read_excel_data(fp)
                if f_raw is None: continue
                c_interp = np.interp(FREQ_AXIS, f_raw, c_raw)
                x_tensor = torch.tensor(c_interp, dtype=torch.float32).view(1, 1, 400).to(DEVICE)
                full_aux = torch.tensor([l / 100.0, m / 5.0], dtype=torch.float32).view(1, 2).to(DEVICE)

                row = {'File': fname, 'Real Temp': t_real}

                # 传统方法拟合 (全量数据计算)
                row['Single L.'] = fit_single_lorentz(FREQ_AXIS, c_interp)
                row['Double L.'] = fit_double_lorentz(FREQ_AXIS, c_interp)

                # AI 模型预测
                for name, model_item in models.items():
                    net = model_item['net']
                    cfg = model_item['cfg']
                    if cfg['aux_dim'] == 0:
                        aux_in = full_aux
                    elif cfg['aux_dim'] == 1:
                        idx = cfg['col']
                        aux_in = full_aux[:, idx:idx + 1]
                    else:
                        aux_in = full_aux

                    pred_norm = net(x_tensor, aux_in)
                    pred_temp = pred_norm.item() * 60.0 + 30.0
                    row[name] = pred_temp

                records.append(row)

        print(f"✅ 从 Excel 文件夹中找到了 {len(records)} 个真实测试样本 (全量数据)。")
    else:
        print(f"❌ 找不到真实测试集文件夹 {REAL_TEST_DIR}！")
        exit()

    if len(records) == 0:
        print("❌ 没有找到有效的真实数据！请检查数据夹。")
        exit()

    # 4. 统计结果与绘图
    df = pd.DataFrame(records)
    mae_scores = {}
    print("\n" + "=" * 50)
    print("🏆 全局真实环境大考成绩单 (包含所有功率组)")
    print("=" * 50)

    contestants = ['Single L.', 'Double L.', 'Laser Only', 'Full Model']
    for name in contestants:
        if name in df.columns:
            valid_df = df.dropna(subset=[name])
            if len(valid_df) > 0:
                mae = np.mean(np.abs(valid_df[name] - valid_df['Real Temp']))
                mae_scores[name] = mae
                print(f"👉 {name:<12} : {mae:.5f} C")
    print("=" * 50)

    # --- 画图 1: 柱状图排名 ---
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 6))

    sorted_scores = sorted(mae_scores.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_scores]
    values = [x[1] for x in sorted_scores]

    color_map = {'Single L.': '#B0B0B0', 'Double L.': '#707070', 'Laser Only': 'dodgerblue',
                 'Full Model': 'forestgreen'}
    colors = [color_map.get(n, 'gray') for n in names]

    bars = plt.bar(names, values, color=colors, alpha=0.85, width=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.4f}" + r"$^{\circ}$C",
                 ha='center', va='bottom', fontsize=12, fontweight='bold', family='Times New Roman')

    plt.ylabel(r'Mean Absolute Error ($^{\circ}$C)', fontsize=14, fontweight='bold', family='Times New Roman')
    plt.title('Real-World Performance Comparison', fontsize=16, fontweight='bold', family='Times New Roman')
    plt.xticks(fontsize=13, family='Times New Roman')
    plt.yticks(fontsize=12, family='Times New Roman')
    plt.ylim(0, max(values) * 1.25)
    plt.tight_layout()
    plt.savefig('全局_测试集结果柱状图.png', dpi=600)
    print("📊 专属柱状图已保存: 全局_测试集结果柱状图.png")

    # --- 画图 2: 散点回归图 ---
    plt.figure(figsize=(9, 9))

    # 动态紧凑坐标轴计算
    all_reals = list(df['Real Temp'].dropna()) + train_cloud_real
    if all_reals:
        min_temp, max_temp = min(all_reals), max(all_reals)
    else:
        min_temp, max_temp = 30, 90

    # 挤压坐标轴：微小的余量 0.5 确保数据贴边
    plt.xlim(min_temp - 0.5, max_temp + 0.5)
    plt.ylim(min_temp - 0.5, max_temp + 0.5)

    # 理想红线
    plt.plot([min_temp - 5, max_temp + 5], [min_temp - 5, max_temp + 5], 'r--', linewidth=2.5, label='Ideal Reference',
             zorder=5)

    # 绘制稀疏背景点云 (数量 1200)
    if train_cloud_real:
        plt.scatter(train_cloud_real, train_cloud_pred, c='limegreen', alpha=0.15, s=15,
                    zorder=1, label=f'Synthetic Data Cloud (n={len(train_cloud_real)})')

    # 测试集模型散点 (全局真实前景数据)
    if 'Single L.' in df.columns:
        plt.scatter(df['Real Temp'], df['Single L.'], c='gray', marker='x', alpha=0.8, s=60,
                    zorder=10, label=f'Single L. (MAE={mae_scores.get("Single L.", 0):.4f})')

    if 'Double L.' in df.columns:
        plt.scatter(df['Real Temp'], df['Double L.'], c='black', marker='+', alpha=0.8, s=70,
                    zorder=10, label=f'Double L. (MAE={mae_scores.get("Double L.", 0):.4f})')

    if 'Laser Only' in df.columns:
        plt.scatter(df['Real Temp'], df['Laser Only'], c='dodgerblue', s=60, alpha=0.9,
                    zorder=11, label=f'Laser Only (MAE={mae_scores.get("Laser Only", 0):.4f})')

    if 'Full Model' in df.columns:
        plt.scatter(df['Real Temp'], df['Full Model'], c='forestgreen', s=70, alpha=1.0,
                    zorder=12, label=f'Full Model (MAE={mae_scores.get("Full Model", 0):.4f})')

    plt.xlabel(r'Ground Truth Temperature ($^{\circ}$C)', fontsize=14, fontweight='bold', family='Times New Roman')
    plt.ylabel(r'Predicted Temperature ($^{\circ}$C)', fontsize=14, fontweight='bold', family='Times New Roman')
    plt.title('Prediction Correlation on All Real Data', fontsize=16, fontweight='bold', family='Times New Roman')

    plt.xticks(fontsize=12, family='Times New Roman')
    plt.yticks(fontsize=12, family='Times New Roman')

    plt.legend(fontsize=12, loc='upper left', framealpha=0.9, edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)

    plt.tight_layout()
    plt.savefig('全局_真实数据回归散点图.png', dpi=600)
    print("📊 专属散点图已保存: 全局_真实数据回归散点图.png")

    print("\n✅ 所有专项测试完成！图表以 600 DPI 极清输出。")