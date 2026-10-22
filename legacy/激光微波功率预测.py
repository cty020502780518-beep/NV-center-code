import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report

# ================= 🔧 配置区域 =================
TRAIN_FILE = "训练集_50k_aligned.pth"
TEST_FILE = "测试集_5k_aligned.pth"
BATCH_SIZE = 128
EPOCHS = 100
PATIENCE = 15
LR = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型保存路径 (升级为 9 组)
BEST_MODEL_PATH = "model_power_classifier_9groups.pth"

# 类别映射 (9 组全覆盖)
CLASS_MAP = {
    0: "Laser 20% / MW -5dBm",
    1: "Laser 20% / MW 0dBm",
    2: "Laser 20% / MW 5dBm",  # ✅ 这一组回来了！
    3: "Laser 50% / MW -5dBm",
    4: "Laser 50% / MW 0dBm",
    5: "Laser 50% / MW 5dBm",
    6: "Laser 100% / MW -5dBm",
    7: "Laser 100% / MW 0dBm",
    8: "Laser 100% / MW 5dBm"
}


# ================= 1. 数据集 =================
class PowerClassificationDataset(Dataset):
    def __init__(self, pth_path):
        print(f"📂 正在加载数据: {pth_path} ...")
        if not os.path.exists(pth_path):
            print(f"❌ 错误: 找不到文件 {pth_path}")
            exit()

        d = torch.load(pth_path)
        self.X = d['X']
        self.Aux = d['Aux']

        max_l = self.Aux[:, 0].max().item()
        is_normalized = max_l <= 1.5

        self.labels = []
        for i in range(len(self.Aux)):
            l, m = self.Aux[i]
            l = l.item();
            m = m.item()

            # Laser (20%, 50%, 100%)
            if is_normalized:
                if l < 0.35:
                    l_idx = 0
                elif l < 0.75:
                    l_idx = 1
                else:
                    l_idx = 2
            else:
                if l < 35:
                    l_idx = 0
                elif l < 75:
                    l_idx = 1
                else:
                    l_idx = 2

            # MW (-5, 0, 5) -> (-1, 0, 1 after norm)
            if m < -0.5:
                m_idx = 0
            elif m < 0.5:
                m_idx = 1
            else:
                m_idx = 2

            class_id = l_idx * 3 + m_idx
            self.labels.append(class_id)

        self.labels = torch.tensor(self.labels, dtype=torch.long)

        # 打印分布
        counts = torch.bincount(self.labels, minlength=9).tolist()
        active_groups = sum([1 for c in counts if c > 0])
        print(f"   ✅ 实际有效功率组: {active_groups} 组 (Expect: 9)")
        if counts[2] > 0:
            print(f"      🎉 成功检测到 [Laser 20% / MW 5dBm] 数据: {counts[2]} 条")
        else:
            print(f"      ⚠️ 警告: 依然没有检测到第 2 类数据，请检查数据生成步骤！")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]


# ================= 2. 网络结构 (ResNet-1D) =================
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


# ================= 3. 评估函数 =================
def evaluate_model(model, dl, name):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0;
    total = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = 100 * correct / total if total > 0 else 0
    print(f"\n📊 [{name}] 准确率: {acc:.2f}%")

    unique_labels = sorted(list(set(all_labels)))
    target_names = [CLASS_MAP[i] for i in unique_labels]

    print("-" * 60)
    print(classification_report(all_labels, all_preds, labels=unique_labels, target_names=target_names, digits=4,
                                zero_division=0))
    print("-" * 60)
    return acc


# ================= 4. 主程序 =================
if __name__ == '__main__':
    print("🔮 启动 [虚拟功率分类器 4.0] (9组全数据完整版)...")

    train_ds = PowerClassificationDataset(TRAIN_FILE)
    test_ds = PowerClassificationDataset(TEST_FILE)

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, BATCH_SIZE, shuffle=False)
    train_eval_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=False)

    model = ResNet1D().to(DEVICE)
    # 标签平滑，防止对噪声过拟合
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_acc = 0.0
    patience_cnt = 0

    print(f"\n🚀 开始训练 (Patience={PATIENCE})...")
    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0;
        train_total = 0
        running_loss = 0.0

        for x, y in train_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, pred = torch.max(out.data, 1)
            train_total += y.size(0)
            train_correct += (pred == y).sum().item()

        train_acc = 100 * train_correct / train_total
        loss_avg = running_loss / len(train_dl)

        # 验证
        model.eval()
        test_correct = 0;
        test_total = 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                _, pred = torch.max(out.data, 1)
                test_total += y.size(0)
                test_correct += (pred == y).sum().item()

        test_acc = 100 * test_correct / test_total if test_total > 0 else 0

        is_best = ""
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            is_best = "⭐ New Best!"
            patience_cnt = 0
        else:
            patience_cnt += 1

        print(
            f"Epoch {epoch + 1:<3} | Loss: {loss_avg:.4f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | {is_best}")

        if patience_cnt >= PATIENCE:
            print("⏹️ 早停触发。")
            break

    print(f"\n🏆 加载最佳模型 (Acc: {best_acc:.2f}%) ...")
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH))

    print("\n🧐 重点核查：训练集上的 9 组全覆盖验证")
    evaluate_model(model, train_eval_dl, "Full 9-Group Check")

    print("\n🧐 辅助核查：测试集验证")
    evaluate_model(model, test_dl, "Test Set Check")