import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import warnings

# 忽略拟合时的运行时警告
warnings.filterwarnings("ignore")

# ================= 🔧 配置区域 =================
TEST_PTH = "测试集_5k_aligned.pth"
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


# ================= 1. 网络定义 =================
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


# ================= 2. 传统拟合算法 =================
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


# ================= 3. 数据解析工具 =================
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
    if isinstance(sample, (tuple, list)):
        x, aux, y = sample[0], sample[1], sample[2]
    else:
        x, aux, y = sample['x'], sample['aux'], sample['y']

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


# ================= 主程序：公平大考与数字映射修正图 =================
if __name__ == '__main__':
    # 全局字体锁定为 Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    mae_scores = {}

    print("\n🚀 加载 AI 模型并在 5k 虚拟测试集上进行【全平台公平大考】...")
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
        records_5k = {name: [] for name in ['Single L.', 'Double L.'] + list(models.keys())}

        with torch.no_grad():
            for i, item in enumerate(test_dataset):
                try:
                    x_tensor, full_aux, t_real = unpack_sample(item)
                except Exception as e:
                    continue

                x_np = x_tensor.cpu().numpy().squeeze()

                pred_single = fit_single_lorentz(FREQ_AXIS, x_np)
                if not np.isnan(pred_single):
                    records_5k['Single L.'].append(abs(pred_single - t_real))

                pred_double = fit_double_lorentz(FREQ_AXIS, x_np)
                if not np.isnan(pred_double):
                    records_5k['Double L.'].append(abs(pred_double - t_real))

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
        exit()

    # ----------------------------------------------------------------
    # 🌟 终端打印最终成绩
    # ----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("🏆 混合评测大考成绩单 (统一在 5k 虚拟集上评测)")
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

    if len(values_plot) == 0:
        exit()

    # ----------------------------------------------------------------
    # 🌟 极简高级版：完整直线柱体 + Y轴真实数字映射截断
    # ----------------------------------------------------------------
    single_l_idx = names_plot.index('Single L.') if 'Single L.' in names_plot else -1

    if single_l_idx != -1 and values_plot[single_l_idx] > max(
            [v for i, v in enumerate(values_plot) if i != single_l_idx]) * 2:
        print("\n🔧 检测到单洛伦兹误差过高，启用【极简直线柱体+真实映射】模式作图...")

        real_single_mae = values_plot[single_l_idx]
        other_max = max([v for i, v in enumerate(values_plot) if i != single_l_idx])

        # 视觉映射参数
        break_start = other_max * 1.10  # Y轴截断下沿
        break_end = other_max * 1.25  # Y轴截断上沿
        gap_size = break_end - break_start
        cap_height = other_max * 1.40  # 破表柱子的视觉封顶高度

        visual_values = []
        for v in values_plot:
            if v > break_end:
                visual_values.append(cap_height)
            else:
                visual_values.append(v)

        colors = ['#B0B0B0', '#707070', '#FFAA00', '#8A2BE2', '#4169E1', '#32CD32']
        fig, ax = plt.subplots(figsize=(10, 6))

        # 绘制极简直线柱体
        bars = ax.bar(names_plot, visual_values, color=colors, alpha=0.9, width=0.55, edgecolor='black', linewidth=1.2)

        for i, (bar, real_val) in enumerate(zip(bars, values_plot)):
            if real_val > break_end:
                ax.text(bar.get_x() + bar.get_width() / 2, cap_height + 0.02 * (other_max * 1.55),
                        f"{real_val:.4f}°C", ha='center', va='bottom', fontweight='bold', fontsize=12,
                        family='Times New Roman')
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, visual_values[i] + 0.02 * (other_max * 1.55),
                        f"{real_val:.4f}°C", ha='center', va='bottom', fontweight='bold', fontsize=12,
                        family='Times New Roman')

        # ================= ✨✨ 关键修复：数学坐标逆向映射与生成真实标签 ✨✨ =================
        # 1. 提取底部正常的坐标轴刻度和步长
        original_yticks = ax.get_yticks()
        valid_lower_yticks = [t for t in original_yticks if t <= break_start]
        tick_step = valid_lower_yticks[1] - valid_lower_yticks[0] if len(valid_lower_yticks) > 1 else 0.2

        # 2. 计算顶部坐标对应的“真实物理起点”
        # 我们让视觉上的 cap_height 严格对应物理上的 real_single_mae
        real_break_end = real_single_mae - (cap_height - break_end)

        # 3. 按照底部相同的步长，自动生成真实的高位数字（比如 5.0, 6.0 等）
        first_upper_r_tick = np.ceil(real_break_end / tick_step) * tick_step
        upper_real_ticks = []
        upper_visual_ticks = []

        for i in range(4):  # 向上探几个刻度
            r_tick = first_upper_r_tick + i * tick_step
            if r_tick > real_single_mae + tick_step * 1.2:
                break  # 防止高刻度无限延伸
            # 把真实的物理大数字，按照 1:1 的比例重新塞回缩短的视觉区间里
            v_tick = break_end + (r_tick - real_break_end)
            upper_real_ticks.append(r_tick)
            upper_visual_ticks.append(v_tick)

        # 4. 把上下两段视觉刻度和真实数字缝合起来
        all_v_ticks = valid_lower_yticks + upper_visual_ticks
        all_labels = [f"{t:g}" for t in valid_lower_yticks] + [f"{t:g}" for t in upper_real_ticks]

        # 计算新的完整画布顶端
        visual_max = max(cap_height + tick_step * 0.6, all_v_ticks[-1] + tick_step * 0.4 if all_v_ticks else cap_height)

        # ================= 绘制带有折断标记的 Y 轴 =================
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(True)

        dx = 0.015  # Y轴闪电折线的横向偏移幅度
        y_axis_pts = [0, break_start, break_start + gap_size * 0.33, break_start + gap_size * 0.66, break_end,
                      visual_max]
        x_axis_pts = [0, 0, -dx, dx, 0, 0]

        ax.plot(x_axis_pts, y_axis_pts, color='black', lw=1.2, transform=ax.get_yaxis_transform(), clip_on=False)

        # ✨ 将真实的标签强行写在被压缩的坐标带上 ✨
        ax.set_yticks(all_v_ticks)
        ax.set_yticklabels(all_labels, family='Times New Roman', fontsize=12)
        ax.set_ylim(0, visual_max)

        # ================= 添加网格与装饰 =================
        ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
        ax.set_ylabel('Mean Absolute Error (°C)', fontsize=14, fontweight='bold', family='Times New Roman', labelpad=15)
        ax.set_title('Ablation Study: Extreme Deformation vs. AI Robustness', fontsize=16, fontweight='bold',
                     family='Times New Roman', pad=15)

        plt.xticks(fontsize=13, family='Times New Roman')

        plt.tight_layout()
        save_path = '混合评测_完美真实刻度映射对比图.png'
        plt.savefig(save_path, dpi=600)
        print(f"📊 带有【真实数字映射】的终极对比图已保存至: {save_path}")

    else:
        # 常规作图代码
        colors = ['#B0B0B0', '#707070', '#FFAA00', '#8A2BE2', '#4169E1', '#32CD32']
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(names_plot, values_plot, color=colors[:len(names_plot)], alpha=0.9, width=0.55, edgecolor='black',
                      linewidth=1.2)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * max(values_plot), f"{h:.4f}°C",
                    ha='center', va='bottom', fontsize=12, fontweight='bold', family='Times New Roman')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
        ax.set_ylabel('Mean Absolute Error (°C)', fontsize=14, fontweight='bold', family='Times New Roman')
        ax.set_title('Ablation Study: Extreme Deformation vs. AI Robustness', fontsize=16, fontweight='bold',
                     family='Times New Roman', pad=15)
        plt.xticks(fontsize=13, family='Times New Roman')
        plt.yticks(fontsize=12, family='Times New Roman')
        ax.set_ylim(0, max(values_plot) * 1.25)
        plt.tight_layout()
        save_path = '混合评测_常规对比图.png'
        plt.savefig(save_path, dpi=600)
        print(f"📊 常规超清对比图已保存至: {save_path}")