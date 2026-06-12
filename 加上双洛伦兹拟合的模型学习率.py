import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings

# 忽略拟合时的运行时警告
warnings.filterwarnings("ignore")

# ================= 🔧 配置区域 =================
# 1. 真实数据集路径 (仅供单洛伦兹拟合找回 1.92 误差使用)
REAL_TEST_DIR = r"D:\NV数据\测试集"

# 2. 增强测试集路径 (供双洛伦兹和深度学习模型使用)
TEST_PTH = "测试集_5k_aligned.pth"

# 标定参数
GLOBAL_COEFF_FILE = "global_coeffs.npy"
if os.path.exists(GLOBAL_COEFF_FILE):
    MY_K, MY_D0 = np.load(GLOBAL_COEFF_FILE)
    print(f"📍 读取标定参数: k={MY_K:.4f}, b={MY_D0:.4f}")
else:
    print("⚠️ 未找到标定文件，使用备用默认值")
    MY_D0 = 2872.1876
    MY_K = -0.0904

FREQ_AXIS = np.linspace(2858, 2878, 400)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_CFG = {
    'Blind': {'path': 'model_blind.pth', 'aux_dim': 0, 'col': None},
    'MW Only': {'path': 'model_mw.pth', 'aux_dim': 1, 'col': 1},
    'Laser Only': {'path': 'model_laser.pth', 'aux_dim': 1, 'col': 0},
    'Full Model': {'path': 'model_full.pth', 'aux_dim': 2, 'col': None}
}


# =======================================================

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


# --- 2. 传统拟合算法 ---
def single_lorentz(x, y0, A, x0, w):
    return y0 + A * (w ** 2) / ((x - x0) ** 2 + w ** 2)


def fit_single_lorentz(freqs, contrast):
    try:
        y_max, y_min = np.max(contrast), np.min(contrast)
        depth = y_min - y_max
        f_dip = freqs[np.argmin(contrast)]
        p0 = [y_max, depth, f_dip, 3.0]
        bounds = ([y_max - 0.05, -1.0, 2800, 0.5], [y_max + 0.05, 0.0, 2950, 20.0])
        popt, _ = curve_fit(single_lorentz, freqs, contrast, p0=p0, bounds=bounds, maxfev=1500)
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
        bounds = ([y_max - 0.05, -1.0, 2800, 0.5, -1.0, 2800, 0.5], [y_max + 0.05, 0.0, 2950, 20.0, 0.0, 2950, 20.0])
        popt, _ = curve_fit(double_lorentz, freqs, contrast, p0=p0, bounds=bounds, maxfev=1500)
        res_freq = (popt[2] + popt[5]) / 2.0
        return (res_freq - MY_D0) / MY_K
    except:
        return np.nan


# --- 3. 数据解析工具 ---
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


# ================= 主程序：双线推断与画图 =================
if __name__ == '__main__':
    # 全局字体锁定为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    mae_scores = {}

    # ----------------------------------------------------------------
    # 🌟 第一部分：专门处理 Single Lorentz (使用真实 Excel 数据集)
    # ----------------------------------------------------------------
    print("\n🔍 正在扫描真实测试集以计算 Single Lorentz 误差...")
    if os.path.exists(REAL_TEST_DIR):
        excel_files = glob.glob(os.path.join(REAL_TEST_DIR, "**", "*.xlsx"), recursive=True)
        single_l_records = []
        for fp in excel_files:
            if "~$" in fp: continue
            t_real, _, _ = parse_filename(os.path.basename(fp))
            if t_real is None: continue

            f_raw, c_raw = read_excel_data(fp)
            if f_raw is None: continue
            c_interp = np.interp(FREQ_AXIS, f_raw, c_raw)

            pred_single = fit_single_lorentz(FREQ_AXIS, c_interp)
            if not np.isnan(pred_single):
                single_l_records.append(abs(pred_single - t_real))

        if len(single_l_records) > 0:
            mae_scores['Single L.'] = np.mean(single_l_records)
            print(f"✅ Single Lorentz 计算完毕 (样本数: {len(single_l_records)}), MAE: {mae_scores['Single L.']:.4f}°C")
        else:
            print("❌ 未能成功在真实数据集上拟合出有效数据。")
    else:
        print(f"❌ 找不到真实测试集文件夹 {REAL_TEST_DIR}！")

    # ----------------------------------------------------------------
    # 🌟 第二部分：处理 Double Lorentz 与 AI 模型 (使用 5k 虚拟数据集)
    # ----------------------------------------------------------------
    print("\n🚀 加载 AI 模型并在 5k 测试集上运行...")
    models = {}
    for name, cfg in MODELS_CFG.items():
        if not os.path.exists(cfg['path']):
            continue
        net = DualStreamNet(aux_dim=cfg['aux_dim']).to(DEVICE)
        state_dict = torch.load(cfg['path'], map_location=DEVICE)
        net.load_state_dict(state_dict)
        net.eval()
        models[name] = {'net': net, 'cfg': cfg}
        print(f"✅ 已加载 AI 模型: {name}")

    if os.path.exists(TEST_PTH) and models:
        test_dataset = load_and_parse_pth(TEST_PTH)
        records_5k = {name: [] for name in ['Double L.'] + list(models.keys())}

        with torch.no_grad():
            for i, item in enumerate(test_dataset):
                x_tensor, full_aux, t_real = unpack_sample(item)
                x_np = x_tensor.cpu().numpy().squeeze()

                # Double Lorentz (跑 5k 测试集)
                pred_double = fit_double_lorentz(FREQ_AXIS, x_np)
                if not np.isnan(pred_double):
                    records_5k['Double L.'].append(abs(pred_double - t_real))

                # AI 模型 (跑 5k 测试集)
                for name, model_item in models.items():
                    net = model_item['net']
                    cfg = model_item['cfg']
                    if cfg['aux_dim'] == 0:
                        aux_in = full_aux
                    elif cfg['aux_dim'] == 1:
                        aux_in = full_aux[:, cfg['col']:cfg['col'] + 1]
                    else:
                        aux_in = full_aux

                    pred_norm = net(x_tensor, aux_in)
                    pred_temp = pred_norm.item() * 60.0 + 30.0
                    records_5k[name].append(abs(pred_temp - t_real))

                if (i + 1) % 1000 == 0:
                    print(f"   已完成 {i + 1} / {len(test_dataset)} 个样本推断...")

        for name, err_list in records_5k.items():
            if len(err_list) > 0:
                mae_scores[name] = np.mean(err_list)
    else:
        print(f"❌ 找不到测试集文件 {TEST_PTH} 或没有模型被加载。")

    # ----------------------------------------------------------------
    # 🌟 第三部分：合并成绩与画图
    # ----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("🏆 混合评测大考成绩单 (MAE)")
    print("=" * 50)
    contestants = ['Single L.', 'Double L.', 'Blind', 'MW Only', 'Laser Only', 'Full Model']

    names_plot = []
    values_plot = []
    for name in contestants:
        if name in mae_scores:
            print(f"👉 {name:<12} : {mae_scores[name]:.5f} °C")
            names_plot.append(name)
            values_plot.append(mae_scores[name])
    print("=" * 50)

    # 颜色分配：灰色系给传统物理拟合，彩色给四个深度学习模型
    color_map = {'Single L.': '#B0B0B0', 'Double L.': '#707070',
                 'Blind': 'orange', 'MW Only': 'purple', 'Laser Only': 'blue', 'Full Model': 'forestgreen'}
    colors = [color_map.get(n, 'gray') for n in names_plot]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names_plot, values_plot, color=colors, alpha=0.85, width=0.55)

    # 在每个柱子上方居中添加数值标签
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * max(values_plot), f"{h:.4f}°C",
                 ha='center', va='bottom', fontsize=12, fontweight='bold', family='Times New Roman')

    # 装饰
    plt.ylabel('Mean Absolute Error (°C)', fontsize=14, fontweight='bold', family='Times New Roman')
    plt.title('Model Performance Comparison (Hybrid Evaluation)', fontsize=16, fontweight='bold',
              family='Times New Roman')
    plt.xticks(fontsize=13, family='Times New Roman')
    plt.yticks(fontsize=12, family='Times New Roman')

    # 动态调整 Y 轴顶部留白
    plt.ylim(0, max(values_plot) * 1.25)

    plt.tight_layout()

    save_path = '混合评测_模型误差对比柱状图.png'
    plt.savefig(save_path, dpi=600)
    print(f"\n✅ 超清柱状图已生成！图片已保存至: {save_path}")