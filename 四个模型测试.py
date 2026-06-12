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

# 忽略拟合时的运行时警告
warnings.filterwarnings("ignore")

# ================= 🔧 用户配置区域 =================
# 1. 真实 Excel 数据文件夹路径
REAL_TEST_DIR = r"D:\NV数据\全数据集"

# 2. 频率轴 (必须与训练时完全一致!)
FREQ_AXIS = np.linspace(2858, 2878, 400)

# 3. 标定参数
GLOBAL_COEFF_FILE = "global_coeffs.npy"
if os.path.exists(GLOBAL_COEFF_FILE):
    MY_K, MY_D0 = np.load(GLOBAL_COEFF_FILE)
    print(f"📍 成功读取 Step 1 标定参数: k={MY_K:.4f}, b={MY_D0:.4f}")
else:
    print("⚠️ 未找到标定文件，使用默认值")
    MY_D0 = 2872.2368
    MY_K = -0.0912

# 4. 模型路径配置
MODELS_CFG = {
    'Blind': {'path': 'model_blind.pth', 'aux_dim': 0, 'col': None},
    'MW Only': {'path': 'model_mw.pth', 'aux_dim': 1, 'col': 1},
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

# --- 2. 传统洛伦兹拟合算法 ---
def lorentzian(x, y0, A, x0, gamma):
    return y0 - A * (gamma ** 2) / ((x - x0) ** 2 + gamma ** 2)

def fit_traditional(freqs, contrast):
    try:
        p0 = [np.max(contrast), np.max(contrast) - np.min(contrast), freqs[np.argmin(contrast)], 5.0]
        popt, _ = curve_fit(lorentzian, freqs, contrast, p0=p0, maxfev=5000)
        res_freq = popt[2]
        T_pred = (res_freq - MY_D0) / MY_K
        return T_pred
    except:
        return np.nan

# --- 3. 文件解析工具 ---
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
    # 设置字体，优先尝试 SimHei，如果没有则回退到系统默认
    # 关键：单位我们改用 LaTeX，不再依赖中文字体显示特殊符号
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    print("⚔️ 终极对决启动：传统方法 vs 4大AI模型...")
    print(f"📍 标定参数检查: D0={MY_D0}, K={MY_K}")
    print("-" * 60)

    # 1. 加载模型
    models = {}
    for name, cfg in MODELS_CFG.items():
        if not os.path.exists(cfg['path']):
            print(f"❌ 错误：找不到文件 {cfg['path']}，请先训练！")
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

    # 2. 扫描数据
    files = glob.glob(os.path.join(REAL_TEST_DIR, "**", "*.xlsx"), recursive=True)
    print(f"📂 找到 {len(files)} 个测试文件，开始逐个分析...")

    records = []
    with torch.no_grad():
        for i, fp in enumerate(files):
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

            # 传统方法
            t_trad = fit_traditional(FREQ_AXIS, c_interp)
            row['Traditional'] = t_trad

            # AI 模型
            for name, item in models.items():
                net = item['net']
                cfg = item['cfg']
                if cfg['aux_dim'] == 0: aux_in = full_aux
                elif cfg['aux_dim'] == 1:
                    idx = cfg['col']
                    aux_in = full_aux[:, idx:idx + 1]
                else: aux_in = full_aux
                pred_norm = net(x_tensor, aux_in)
                pred_temp = pred_norm.item() * 60.0 + 30.0
                row[name] = pred_temp
            records.append(row)
            if (i + 1) % 10 == 0:
                print(f"   已处理 {i + 1}/{len(files)} 个文件...")

    # 3. 统计结果与绘图
    df = pd.DataFrame(records)
    mae_scores = {}
    print("\n" + "=" * 50)
    print("🏆 最终大考成绩单 (真实 MAE)")
    print("=" * 50)

    contestants = ['Traditional'] + list(models.keys())
    for name in contestants:
        if name in df.columns:
            valid_df = df.dropna(subset=[name])
            mae = np.mean(np.abs(valid_df[name] - valid_df['Real Temp']))
            mae_scores[name] = mae
            # 控制台输出保留 5 位
            print(f"👉 {name:<12} : {mae:.5f} C") # 控制台直接用 C 避免乱码
    print("=" * 50)

    # --- 画图 1: 柱状图排名 ---
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(10, 6))

    sorted_scores = sorted(mae_scores.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in sorted_scores]
    values = [x[1] for x in sorted_scores]
    color_map = {'Traditional': 'gray', 'Blind': 'orange', 'MW Only': 'purple',
                 'Laser Only': 'dodgerblue', 'Full Model': 'forestgreen'}
    colors = [color_map.get(n, 'gray') for n in names]

    bars = plt.bar(names, values, color=colors, alpha=0.85, width=0.6)

    for bar in bars:
        h = bar.get_height()
        # ✨ 关键修正：使用 LaTeX 格式 r"$^{\circ}$C" 确保显示正常
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.5f}" + r"$^{\circ}$C",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Y轴标签使用 LaTeX
    plt.ylabel(r'Mean Absolute Error ($^{\circ}$C)', fontsize=12)
    plt.title('Real-World Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, max(values) * 1.2)
    plt.tight_layout()
    plt.savefig('模型测试集结果图（全）.png', dpi=300)
    print("📊 柱状图已保存: 模型测试集结果图（全）.png")

    # --- 画图 2: 散点回归图 ---
    plt.figure(figsize=(9, 9))
    plt.plot([30, 90], [30, 90], 'r--', linewidth=2, label='Ideal Reference')

    if 'Traditional' in df.columns:
        plt.scatter(df['Real Temp'], df['Traditional'], c='gray', marker='x', alpha=0.5,
                    label=f'Traditional (MAE={mae_scores["Traditional"]:.5f})')

    if 'Laser Only' in df.columns:
        plt.scatter(df['Real Temp'], df['Laser Only'], c='dodgerblue', s=60, alpha=0.7,
                    label=f'Laser Only (MAE={mae_scores["Laser Only"]:.5f})')

    if 'Full Model' in df.columns:
        plt.scatter(df['Real Temp'], df['Full Model'], c='forestgreen', s=30, alpha=0.8,
                    label=f'Full Model (MAE={mae_scores["Full Model"]:.5f})')

    # 坐标轴标签使用 LaTeX
    plt.xlabel(r'Ground Truth Temperature ($^{\circ}$C)', fontsize=12)
    plt.ylabel(r'Predicted Temperature ($^{\circ}$C)', fontsize=12)
    plt.title('Prediction Correlation', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('测试集结果回归散点图（全）.png', dpi=300)
    print("📊 散点图已保存: 测试集结果回归散点图（全）.png")

    print("\n✅ 所有测试完成！快去看看图表吧！")