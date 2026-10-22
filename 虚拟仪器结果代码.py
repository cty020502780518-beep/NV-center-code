import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ================= 🔧 配置区域 =================
REAL_TEST_DIR = r"D:\NV数据\测试集"  # 真实Excel路径
FREQ_AXIS = np.linspace(2858, 2878, 400)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型路径
PATH_CLASSIFIER = "model_power_classifier_9groups.pth"
PATH_FULL = "model_full.pth"
PATH_BLIND = "model_blind.pth"

# 🏷️ 核心：分类ID转物理参数
CLASS_TO_AUX = {
    0: [0.2, -1.0],  # L20 / M-5
    1: [0.2, 0.0],  # L20 / M0
    2: [0.2, 1.0],  # L20 / M5
    3: [0.5, -1.0],  # L50 / M-5
    4: [0.5, 0.0],  # L50 / M0
    5: [0.5, 1.0],  # L50 / M5
    6: [1.0, -1.0],  # L100 / M-5
    7: [1.0, 0.0],  # L100 / M0
    8: [1.0, 1.0]  # L100 / M5
}


# ================= 1. 模型定义 =================
# --- A. 分类器 (ResNet1D) ---
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self):
        super(ResNet1D, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(32, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, 9)

    def _make_layer(self, in_c, out_c, stride):
        return nn.Sequential(ResBlock(in_c, out_c, stride), ResBlock(out_c, out_c, 1))

    def forward(self, x):
        x_min = x.min(dim=2, keepdim=True)[0]
        x_max = x.max(dim=2, keepdim=True)[0]
        x_norm = (x - x_min) / (x_max - x_min + 1e-8) - 0.5
        x = self.prep(x_norm)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.fc(x.flatten(1))


# --- B. 温度回归器 (DualStreamNet) ---
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


# ================= 2. 工具函数 =================
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


# ================= 3. 主程序 =================
if __name__ == '__main__':
    # ✨✨✨ IEEE 顶刊级图表全局配置：强制 Times New Roman ✨✨✨
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'  # 公式也用 Times 风格
    plt.rcParams['font.size'] = 11  # 基础字号适度调大
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['axes.linewidth'] = 1.0

    print("🤖 启动 [虚拟传感器] 级联验证程序 (9组全数据版)...")

    # --- 加载模型 ---
    if not os.path.exists(PATH_CLASSIFIER):
        print(f"❌ 错误：找不到分类器 {PATH_CLASSIFIER}")
        exit()
    classifier = ResNet1D().to(DEVICE)
    classifier.load_state_dict(torch.load(PATH_CLASSIFIER, map_location=DEVICE))
    classifier.eval()
    print(f"✅ 分类器 ({PATH_CLASSIFIER}) 加载完毕")

    if not os.path.exists(PATH_FULL):
        print("❌ 错误：找不到 Full Model")
        exit()
    full_model = DualStreamNet(aux_dim=2).to(DEVICE)
    full_model.load_state_dict(torch.load(PATH_FULL, map_location=DEVICE))
    full_model.eval()
    print("✅ 温度主模型 (Full Model) 加载完毕")

    blind_model = None
    if os.path.exists(PATH_BLIND):
        blind_model = DualStreamNet(aux_dim=0).to(DEVICE)
        blind_model.load_state_dict(torch.load(PATH_BLIND, map_location=DEVICE))
        blind_model.eval()
        print("✅ 盲测模型 (Blind) 加载完毕")

    # --- 开始遍历文件 ---
    files = glob.glob(os.path.join(REAL_TEST_DIR, "**", "*.xlsx"), recursive=True)
    print(f"\n📂 开始分析 {len(files)} 个测试文件...")

    results = []

    with torch.no_grad():
        for i, fp in enumerate(files):
            if "~$" in fp: continue
            fname = os.path.basename(fp)
            t_real, l_real, m_real = parse_filename(fname)
            if t_real is None: continue

            # 读取数据
            f_raw, c_raw = read_excel_data(fp)
            if f_raw is None: continue
            c_interp = np.interp(FREQ_AXIS, f_raw, c_raw)
            x_tensor = torch.tensor(c_interp, dtype=torch.float32).view(1, 1, 400).to(DEVICE)

            # === 核心流程 ===
            # Step 1: AI 猜功率
            logits = classifier(x_tensor)
            pred_class = torch.argmax(logits, dim=1).item()
            pred_aux_vals = CLASS_TO_AUX.get(pred_class, [0.5, 0.0])

            # 构造虚拟参数
            aux_virtual = torch.tensor(pred_aux_vals, dtype=torch.float32).view(1, 2).to(DEVICE)

            # Step 2: 虚拟参数 -> Full Model -> 预测温度
            pred_norm = full_model(x_tensor, aux_virtual)
            t_virtual = pred_norm.item() * 60.0 + 30.0

            # === 对比组 ===
            # 对比 A: 盲测
            t_blind = np.nan
            if blind_model:
                t_blind = blind_model(x_tensor, None).item() * 60.0 + 30.0

            # 对比 B: 完美全参
            m_norm = -1.0 if m_real < -2.5 else (1.0 if m_real > 2.5 else 0.0)
            aux_ideal = torch.tensor([l_real / 100.0, m_norm], dtype=torch.float32).view(1, 2).to(DEVICE)
            t_ideal = full_model(x_tensor, aux_ideal).item() * 60.0 + 30.0

            results.append({
                'Real': t_real,
                'Blind': t_blind,
                'Virtual': t_virtual,
                'Ideal': t_ideal,
                'Class_Correct': 1 if (l_real / 100.0 == pred_aux_vals[0] and m_norm == pred_aux_vals[1]) else 0
            })

    # --- 统计分析 ---
    df = pd.DataFrame(results)
    mae_blind = np.mean(np.abs(df['Blind'] - df['Real']))
    mae_virtual = np.mean(np.abs(df['Virtual'] - df['Real']))
    mae_ideal = np.mean(np.abs(df['Ideal'] - df['Real']))
    class_acc = df['Class_Correct'].mean() * 100

    print("\n" + "=" * 50)
    print("🏆 最终级联验证结果 (MAE)")
    print("=" * 50)
    print(f"🍊 单级 AI (Single-Stage)           : {mae_blind:.4f} C")
    print(f"🚀 提出级联 AI (Virtual Power)     : {mae_virtual:.4f} C")
    print(f"🌲 理想级联 AI (Hardware Power)    : {mae_ideal:.4f} C")
    print("-" * 50)
    print(f"🎯 物理验证集准确率: {class_acc:.2f}%")

    # --- 📊 绘图 (顶刊化修改版) ---
    plt.figure(figsize=(9, 7))

    # 替换为高级学术名称，并加入换行符以防重叠
    methods = [
        'Single-Stage AI\n(No Power Prior)',
        'Proposed Cascaded AI\n(Virtual Power)',
        'Ideal Cascaded AI\n(Hardware Power)'
    ]
    maes = [mae_blind, mae_virtual, mae_ideal]
    colors = ['#FFA500', '#DC143C', '#228B22']

    bars = plt.bar(methods, maes, color=colors, alpha=0.85, width=0.55)

    # 1. 标注数值 (保留4位小数，加单位，使用 LaTeX 公式格式)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f"{h:.4f}" + r"$^{\circ}$C",
                 ha='center', va='bottom', fontweight='bold', fontsize=12)

    # ✨ 2. 添加分类准确率“勋章” (更改为 Physical Validation Accuracy) ✨
    info_text = f"Physical Validation Accuracy: {class_acc:.2f}%"
    plt.text(0.05, 0.92, info_text, transform=plt.gca().transAxes,
             fontsize=12, fontweight='bold', color='#333333',
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

    # 3. 样式调整
    plt.ylabel(r"Mean Absolute Error ($^{\circ}$C)", fontsize=13)
    plt.title("Performance of Virtual Sensing Pipeline", fontsize=15, fontweight='bold', pad=15)

    # 增加顶部空间
    plt.ylim(0, max(maes) * 1.35)

    # 加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    # 存为高分辨率 PNG 和 矢量 PDF 格式备用
    plt.savefig("虚拟仪器测试集结果图_IEEE.png", dpi=600)
    plt.savefig("虚拟仪器测试集结果图_IEEE.pdf", format='pdf', bbox_inches='tight')
    print("📊 验证对比图已保存为 PNG 和 PDF 双格式！")