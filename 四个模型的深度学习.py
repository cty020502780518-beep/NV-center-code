import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ================= 配置 =================
TRAIN_FILE = "训练集_50k_aligned.pth"
TEST_FILE = "测试集_5k_aligned.pth"
BATCH_SIZE = 128
EPOCHS = 200
PATIENCE = 20
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =======================================

class DualStreamNet(nn.Module):
    def __init__(self, aux_dim=2):
        super(DualStreamNet, self).__init__()
        # ... (网络结构部分保持完全不变) ...
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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
        self.X = d['X']
        self.Y = d['Y']
        self.Aux = d['Aux']  # [N, 2] -> [Laser, Microwave]
        # 这里需要注意，原始代码里 Y 是真实温度，这里我们做个简单的归一化用于训练
        self.Y_norm = (self.Y - 30.0) / 60.0

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        # 返回: 光谱, 归一化温度, 真实温度, 辅助参数
        return self.X[idx], self.Y_norm[idx], self.Y[idx], self.Aux[idx]


# 训练函数
def train_variant(variant_name, aux_dim, target_col, save_path, train_ds, test_ds):
    print(f"\n🚀 开始训练模型: 【{variant_name}】...")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = DualStreamNet(aux_dim=aux_dim).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, factor=0.5)

    best_mae = 999.0
    patience_counter = 0

    for ep in range(EPOCHS):
        model.train()
        for x, y_norm, y_real, aux in train_loader:
            x, y_norm, aux = x.to(DEVICE), y_norm.to(DEVICE), aux.to(DEVICE)

            if aux_dim == 0:
                aux_input = aux  # 盲测不用管，模型里没用到
            elif aux_dim == 1:
                # 只取指定的一列 (Batch, 1)
                aux_input = aux[:, target_col:target_col + 1]
            else:
                aux_input = aux  # 全配

            opt.zero_grad()
            pred = model(x, aux_input)
            loss = loss_fn(pred, y_norm)
            loss.backward()
            opt.step()

        # 验证集评估
        model.eval()
        total_err = 0
        with torch.no_grad():
            for x, y_norm, y_real, aux in test_loader:
                x, y_real, aux = x.to(DEVICE), y_real.to(DEVICE), aux.to(DEVICE)

                if aux_dim == 0:
                    aux_input = aux
                elif aux_dim == 1:
                    aux_input = aux[:, target_col:target_col + 1]
                else:
                    aux_input = aux

                pred = model(x, aux_input)
                # 反归一化计算真实 MAE
                pred_real_temp = pred * 60.0 + 30.0
                total_err += torch.sum(torch.abs(pred_real_temp - y_real)).item()

        curr_mae = total_err / len(test_ds)
        scheduler.step(curr_mae)

        if curr_mae < best_mae:
            best_mae = curr_mae
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if ep % 10 == 0:
            print(f"   Ep {ep:03d} | MAE: {curr_mae:.5f} | Best: {best_mae:.5f}")

        if patience_counter >= PATIENCE:
            print(f"🛑 早停 at Epoch {ep}")
            break

    print(f"✅ 【{variant_name}】 最佳 MAE: {best_mae:.5f} ℃")
    return best_mae


if __name__ == '__main__':
    # ✨✨✨ 设置中文字体和 LaTeX 符号支持 ✨✨✨
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    print("📂 预加载数据...")
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        print("❌ 错误：找不到训练集或测试集文件 (.pth)，请检查路径！")
        exit()

    train_ds = NVDataset(TRAIN_FILE)
    test_ds = NVDataset(TEST_FILE)

    # 依次训练 4 个变体
    # 1. 盲测 (Aux=0)
    mae_blind = train_variant("Blind (No Aux)", 0, None, "model_blind.pth", train_ds, test_ds)

    # 2. 激光单参 (Aux=1, col=0)
    mae_laser = train_variant("Laser Only", 1, 0, "model_laser.pth", train_ds, test_ds)

    # 3. 微波单参 (Aux=1, col=1)
    mae_mw = train_variant("Microwave Only", 1, 1, "model_mw.pth", train_ds, test_ds)

    # 4. 全参 (Aux=2)
    mae_full = train_variant("Full Model", 2, None, "model_full.pth", train_ds, test_ds)

    # --- 绘图部分 ---
    print("\n" + "=" * 50)
    print(f"Blind: {mae_blind:.5f}")
    print(f"Laser: {mae_laser:.5f}")
    print(f"MW:    {mae_mw:.5f}")
    print(f"Full:  {mae_full:.5f}")
    print("=" * 50)

    names = ['Traditional', 'Blind', 'MW Only', 'Laser Only', 'Full Model']
    # 注意：Traditional 的值这里是硬编码的 1.70，如果你有更精确的值请替换
    # 例如：1.92000
    values = [1.70, mae_blind, mae_mw, mae_laser, mae_full]
    colors = ['gray', 'orange', 'purple', 'blue', 'green']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=colors, alpha=0.8)

    # ✨✨✨ 关键修改：显示 5 位小数，并加上 LaTeX 单位 ✨✨✨
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.05,
                 f"{yval:.5f}" + r"$^{\circ}$C",
                 ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.title('Ablation Study: Individual Contribution', fontsize=14, fontweight='bold')
    # 使用 LaTeX 格式显示单位
    plt.ylabel(r'Mean Absolute Error ($^{\circ}$C)', fontsize=12)
    plt.ylim(0, max(values) * 1.2)  # 增加一点头部空间放标签

    plt.tight_layout()
    plt.savefig('模型学习率图.png', dpi=300)
    print("📊 最终四件套对比图已生成: 模型学习率图.png")