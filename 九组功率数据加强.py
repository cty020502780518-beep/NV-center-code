import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import re
import pandas as pd
import numpy as np
import torch
from scipy.optimize import curve_fit

# ================= 🔧 配置区域 =================
TRAIN_DIR = r"D:\NV数据\训练集"
TEST_DIR_SRC = r"D:\NV数据\测试集"

# 输入文件 (指向你刚刚生成的、包含9组数据的标定文件)
# 注意：文件名要和你上一步保存的一致！
COEFF_FILE = "全功率组_温度系数标定.csv"

# 输出文件
OUTPUT_TRAIN_DB = "训练集_50k_aligned.pth"
OUTPUT_TEST_DB = "测试集_5k_aligned.pth"

# 统一频轴 (必须与训练时一致)
FREQ_AXIS = np.linspace(2858, 2878, 400)


# ===============================================

# 物理模型：双峰拟合
def double_lorentz(x, y0, A1, x1, w1, A2, x2, w2):
    L1 = A1 * (w1 ** 2) / ((x - x1) ** 2 + w1 ** 2)
    L2 = A2 * (w2 ** 2) / ((x - x2) ** 2 + w2 ** 2)
    return y0 + L1 + L2


def get_physics_center(f, c):
    y_max = np.max(c)
    y_min = np.min(c)
    depth = y_min - y_max
    idx_min = np.argmin(c)
    x_center_guess = f[idx_min]

    p0 = [y_max, depth / 2, x_center_guess - 2, 3.0, depth / 2, x_center_guess + 2, 3.0]
    bounds = ([y_max - 0.05, -1.0, 2800, 0.5, -1.0, 2800, 0.5],
              [y_max + 0.05, 0.0, 2950, 20.0, 0.0, 2950, 20.0])
    try:
        popt, _ = curve_fit(double_lorentz, f, c, p0=p0, bounds=bounds, maxfev=2000)
        x1, x2 = popt[2], popt[5]
        # 物理中心 = 双峰平均
        center = (x1 + x2) / 2.0
        return center
    except:
        return f[idx_min]


def load_physics_profile(csv_path):
    print(f"📖 正在读取标定参数: {csv_path}")
    df = pd.read_csv(csv_path)
    profile = {}

    # 自动识别列名 (兼容不同版本的标定代码)
    slope_col = 'Slope' if 'Slope' in df.columns else 'Slope (MHz/K)'
    int_col = 'Intercept' if 'Intercept' in df.columns else 'Intercept (MHz)'

    print(f"   检测到列名: {slope_col}, {int_col}")

    for _, row in df.iterrows():
        key = (float(row['Laser (%)']), float(row['MW (dBm)']))
        profile[key] = (float(row[slope_col]), float(row[int_col]))

    print(f"   成功加载 {len(profile)} 组标定参数。")
    if (20.0, 5.0) in profile:
        print("   ✅ 确认包含: Laser 20% / MW 5dBm (数据已恢复)")
    else:
        print("   ⚠️ 警告: 未检测到 20% / 5dBm 组，请检查标定文件！")

    return profile


def augment_baseline(signal):
    offset = np.random.uniform(-0.02, 0.02) * np.max(np.abs(signal))
    slope = np.random.uniform(-0.00005, 0.00005)
    tilt = slope * np.arange(len(signal))
    return signal + offset + tilt


def parse_filename(filename):
    filename = filename.replace("（", "(").replace("）", ")").replace(" ", "")
    l_match = re.search(r'(\d+)%', filename)
    m_match = re.search(r'([-]?\d+)dbm', filename, re.IGNORECASE)
    l = float(l_match.group(1)) if l_match else 50.0
    m = float(m_match.group(1)) if m_match else 0.0
    return l, m


def read_excel_data(path):
    try:
        xls = pd.read_excel(path, sheet_name=None, engine='openpyxl')
        df = next((v for k, v in xls.items() if 'data' in k.lower()), list(xls.values())[0])
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df.iloc[:, 0].values, df.iloc[:, 1].values
    except:
        return None, None


def generate_db(source_dir, profile, num_samples, out_pth):
    print(f"\n🏭 正在生成数据库: {out_pth} ...")

    # 1. 加载种子
    seeds_by_power = {}
    files = glob.glob(os.path.join(source_dir, "**", "*.xlsx"), recursive=True)

    print("   正在扫描并匹配种子文件...")
    for fp in files:
        if "~$" in fp: continue
        l, m = parse_filename(os.path.basename(fp))

        # 在 profile 里找对应参数
        best_key = None
        min_dist = 999
        for key in profile.keys():
            dist = abs(l - key[0]) + abs(m - key[1])
            if dist < min_dist:
                min_dist = dist
                best_key = key

        # 只要找到了 (距离<2)，就收录！(现在20/5也能被找到了)
        if min_dist > 2: continue

        f, c = read_excel_data(fp)
        if f is not None:
            c_norm = np.interp(FREQ_AXIS, f, c)
            # 计算物理中心
            f_center = get_physics_center(FREQ_AXIS, c_norm)

            if best_key not in seeds_by_power: seeds_by_power[best_key] = []
            seeds_by_power[best_key].append({'c': c_norm, 'f_center': f_center})

    print(f"   种子加载完毕，共涵盖 {len(seeds_by_power)} 组功率配置。")

    # 2. 生成循环
    X, Y, Aux = [], [], []
    valid_keys = list(seeds_by_power.keys())

    if len(valid_keys) == 0:
        print("❌ 错误：没有匹配到任何种子文件！请检查路径或文件名。")
        return

    print(f"   目标生成数量: {num_samples}")
    count = 0
    while count < num_samples:
        # 随机选一组功率 (现在包含 20/5 了)
        key = valid_keys[np.random.randint(len(valid_keys))]

        # 随机选该组下的一个种子
        if len(seeds_by_power[key]) == 0: continue
        seed = seeds_by_power[key][np.random.randint(len(seeds_by_power[key]))]

        # 随机温度 & 物理平移
        target_temp = np.random.uniform(35, 85)
        slope, intercept = profile[key]

        f_target_center = slope * target_temp + intercept
        shift = f_target_center - seed['f_center']

        # 施加平移 + 抖动
        jitter = np.random.normal(0, 0.005)
        c_shifted = np.interp(FREQ_AXIS - (shift + jitter), FREQ_AXIS, seed['c'])

        # 增强
        c_final = c_shifted + np.random.normal(0, 0.0003, 400)
        c_final = augment_baseline(c_final)

        X.append(c_final)
        Y.append(target_temp)

        # 记录辅助参数 (归一化)
        Aux.append([key[0] / 100.0, key[1] / 5.0])  # Laser/100, MW/5

        count += 1
        if count % (num_samples // 5) == 0: print(f"   ...已生成 {count} 条")

    # 保存
    torch.save({
        'X': torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1),
        'Y': torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(1),
        'Aux': torch.tensor(np.array(Aux), dtype=torch.float32)
    }, out_pth)
    print("   ✅ 数据库保存成功。")


if __name__ == '__main__':
    if not os.path.exists(COEFF_FILE):
        print(f"❌ 错误：找不到标定文件 {COEFF_FILE}")
        print("   请确保你已经运行了新的标定代码，并生成了该文件。")
    else:
        # 加载标定参数
        physics_profile = load_physics_profile(COEFF_FILE)

        # 生成训练集
        generate_db(TRAIN_DIR, physics_profile, 50000, OUTPUT_TRAIN_DB)

        # 生成测试集
        generate_db(TEST_DIR_SRC, physics_profile, 5000, OUTPUT_TEST_DB)

        print("\n🎉 全功率组数据库构建完成！")
        print("   下一步：请重新运行 'ResNet-1D' 分类器训练，这次它应该能识别第 2 类了！")