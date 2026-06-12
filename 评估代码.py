import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# ================= 🔧 配置区域 =================
TEST_FILE = "测试集_5k_aligned.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_CFG = {
    'Blind': ['model_blind.pth', 0, None, 'orange'],
    'MW Only': ['model_mw.pth', 1, 1, 'purple'],
    'Laser Only': ['model_laser.pth', 1, 0, 'dodgerblue'],
    'Full Model': ['model_full.pth', 2, None, 'forestgreen']
}

# ===========================================
# --- 1. 网络结构 (保持一致) ---
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

class NVDataset(Dataset):
    def __init__(self, pth):
        d = torch.load(pth)
        self.X, self.Y, self.Aux = d['X'], d['Y'], d['Aux']
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i], self.Aux[i]

# --- 主程序 ---
if __name__ == '__main__':
    # ✨✨✨ 改动1：设置中文字体 ✨✨✨
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    print("⚔️ 启动四模型终极对比评估...")

    if not os.path.exists(TEST_FILE):
        print(f"❌ 错误：找不到测试集 {TEST_FILE}")
        exit()

    test_ds = NVDataset(TEST_FILE)
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False)
    results = {}

    real_temps = []
    for _, y, _ in test_dl:
        real_temps.extend(y.numpy().flatten())
    real_temps = np.array(real_temps)

    for name, cfg in MODELS_CFG.items():
        fname, aux_dim, col_idx, color = cfg
        print(f"👉 正在评估: {name:<12} (加载 {fname})...")

        if not os.path.exists(fname):
            print(f"   ⚠️ 跳过：找不到文件 {fname}")
            continue

        model = DualStreamNet(aux_dim=aux_dim).to(DEVICE)
        model.load_state_dict(torch.load(fname, map_location=DEVICE))
        model.eval()

        preds = []
        with torch.no_grad():
            for x, y, aux in test_dl:
                x = x.to(DEVICE)
                aux = aux.to(DEVICE)
                if aux_dim == 0: pred = model(x, None)
                elif aux_dim == 1: pred = model(x, aux[:, col_idx:col_idx + 1])
                else: pred = model(x, aux)
                preds.extend(pred.cpu().numpy().flatten())

        preds = np.array(preds) * 60.0 + 30.0
        mae = np.mean(np.abs(preds - real_temps))
        rmse = np.sqrt(np.mean((preds - real_temps) ** 2))

        results[name] = {'preds': preds, 'mae': mae, 'rmse': rmse, 'color': color}

    # --- 📊 1. 打印成绩单 ---
    print("\n" + "=" * 60)
    print(f"{'Model Name':<15} | {'MAE (℃)':<10} | {'RMSE (℃)':<10} | {'评价'}")
    print("-" * 60)

    sorted_models = sorted(results.items(), key=lambda x: x[1]['mae'])
    for name, data in sorted_models:
        mae = data['mae']
        tag = ""
        if mae < 0.5: tag = "🏆 完美"
        elif mae < 0.7: tag = "✅ 优秀"
        elif mae < 1.0: tag = "⭕ 良好"
        else: tag = "⚠️ 一般"
        # ✨✨✨ 改动2：控制台表格保留 5 位小数 ✨✨✨
        print(f"{name:<15} | {mae:<10.5f} | {data['rmse']:<10.5f} | {tag}")
    print("=" * 60)

    # --- 📈 2. 画图 (2x2 子图) ---
    plt.figure(figsize=(15, 10))

    # A. 柱状图对比 (MAE)
    plt.subplot(2, 2, 1)
    names = [x[0] for x in sorted_models]
    maes = [x[1]['mae'] for x in sorted_models]
    colors = [x[1]['color'] for x in sorted_models]
    bars = plt.bar(names, maes, color=colors, alpha=0.8)
    plt.title("Model Comparison (MAE)", fontsize=14)
    plt.ylabel("Mean Absolute Error (℃)")
    for bar in bars:
        # ✨✨✨ 改动3：柱状图标签保留 5 位小数 ✨✨✨
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.5f}",
                 ha='center', va='bottom', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # B. 散点回归图 (Best Model)
    best_name = sorted_models[0][0]
    best_data = sorted_models[0][1]
    plt.subplot(2, 2, 2)
    plt.scatter(real_temps, best_data['preds'], s=2, alpha=0.3, c=best_data['color'])
    plt.plot([30, 90], [30, 90], 'r--', lw=2, label='Ideal')
    plt.title(f"Best Model Regression: {best_name}", fontsize=14)
    plt.xlabel("Real Temp (℃)")
    plt.ylabel("Predicted Temp (℃)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # C. 误差分布直方图 (All Models)
    plt.subplot(2, 1, 2)
    for name, data in results.items():
        errors = data['preds'] - real_temps
        # ✨✨✨ 改动4：分布图例保留 5 位小数 ✨✨✨
        plt.hist(errors, bins=50, alpha=0.4, label=f"{name} (MAE={data['mae']:.5f})", color=data['color'], density=True)

    plt.title("Error Distribution Comparison (Stacked)", fontsize=14)
    plt.xlabel("Error (℃)")
    plt.ylabel("Density")
    plt.legend()
    plt.xlim(-3, 3)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()