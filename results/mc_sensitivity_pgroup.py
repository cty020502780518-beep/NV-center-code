# mc_sensitivity_pgroup.py — T1: 经典 Double-L + 分态标定链路的 Monte Carlo 噪声鲁棒性
# 在 mc_sensitivity.py 同框架下，新增两条 σ_T 曲线：
#   Double-L per-group (oracle) : 用真实功率标签选 (k_i,b_i) 做温度反演
#   Double-L per-group (routed) : 用 Stage-I 分类器 argmax 预测标签选 (k_i,b_i)
# 噪声注入/插值/拟合初值界 全部与 mc_sensitivity.py 一致（REPEAT=15, 20级, seed=7）
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob, re, json, warnings
import numpy as np
import torch, torch.nn as nn
import openpyxl
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

REAL_TEST_DIR = "newdata/测试集"
FREQ_AXIS = np.linspace(2858, 2878, 400)
std_k, std_b = np.load("eval_models/global_coeffs.npy")
REPEAT = 15
noise_levels = np.linspace(0, 0.02, 20)

# ---- 分态标定表（规范源 全功率组_温度系数标定.csv，slope/intercept 单位 MHz 与 MHz/K）----
CAL_CSV = "eval_models/全功率组_温度系数标定.csv"
import csv as _csv
GROUP_COEFFS = {}  # (laser_pct, mw_dbm) -> (k_i, b_i)  MHz/K, MHz
with open(CAL_CSV, encoding='utf-8-sig') as f:
    for row in _csv.DictReader(f):
        l = float(row['Laser (%)']); m = float(row['MW (dBm)'])
        GROUP_COEFFS[(l, m)] = (float(row['Slope']), float(row['Intercept']))
assert len(GROUP_COEFFS) == 9, f"expect 9 groups, got {len(GROUP_COEFFS)}"

# class index -> (laser_pct, mw_dbm)  与 eval_pipeline.CLASS_TO_AUX 同序
CLASS_TO_LM = {0:(20,-5),1:(20,0),2:(20,5),3:(50,-5),4:(50,0),5:(50,5),6:(100,-5),7:(100,0),8:(100,5)}

def double_lorentz(x, y0, A1, x1, w1, A2, x2, w2):
    return y0 + A1*w1*w1/((x-x1)**2+w1*w1) + A2*w2*w2/((x-x2)**2+w2**2)

def fit_double_center(f, c):
    """返回双洛伦兹中心 (x1+x2)/2 (MHz)，与 mc_sensitivity.fit_double 同初值同界。"""
    try:
        y_max, y_min = c.max(), c.min(); dep = y_min - y_max; fd = f[np.argmin(c)]
        p0 = [y_max, dep/2, fd-2, 3.0, dep/2, fd+2, 3.0]
        p,_ = curve_fit(double_lorentz, f, c, p0=p0,
                        bounds=([y_max-0.05,-1,2800,0.5,-1,2800,0.5],
                                [y_max+0.05,0,2950,20,0,2950,20]), maxfev=1500)
        return (p[2]+p[5])/2.0
    except Exception:
        return np.nan

# ---- Stage-I 分类器（routed 用），定义与 eval_pipeline.py 完全一致 ----
class ResBlock(nn.Module):
    def __init__(self, ic, oc, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(ic, oc, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm1d(oc)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(oc, oc, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm1d(oc)
        self.shortcut = nn.Sequential()
        if stride != 1 or ic != oc:
            self.shortcut = nn.Sequential(nn.Conv1d(ic, oc, 1, stride, bias=False), nn.BatchNorm1d(oc))
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out)) + self.shortcut(x)
        return self.relu(out)
class ResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep = nn.Sequential(nn.Conv1d(1,32,7,2,3,bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(3,2,1))
        self.layer1 = self._mk(32,64,1); self.layer2 = self._mk(64,128,2); self.layer3 = self._mk(128,256,2)
        self.avgpool = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(256,9)
    def _mk(self,a,b,s): return nn.Sequential(ResBlock(a,b,s), ResBlock(b,b,1))
    def forward(self,x):
        xm = x.min(dim=2,keepdim=True)[0]; xM = x.max(dim=2,keepdim=True)[0]
        x = (x-xm)/(xM-xm+1e-8)-0.5
        x = self.prep(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.fc(self.avgpool(x).flatten(1))

classifier = ResNet1D()
classifier.load_state_dict(torch.load("eval_models/model_power_classifier_9groups.pth", map_location="cpu"))
classifier.eval()

def parse_name(fn):
    s = os.path.basename(fn).replace("（","(").replace("）",")").replace(" ","")
    t = re.search(r'(\d+(\.\d+)?)[°℃]', s); l = re.search(r'(\d+)%', s); m = re.search(r'([-]?\d+)dbm', s, re.I)
    return (float(t.group(1)), float(l.group(1)), float(m.group(1))) if t and l else None

samples = []  # (c_interp_400, laser_pct, mw_dbm)
for fp in sorted(glob.glob(os.path.join(REAL_TEST_DIR, "*.xlsx"))):
    if "~$" in fp: continue
    meta = parse_name(fp)
    if not meta: continue
    t, l, m = meta
    wb = openpyxl.load_workbook(fp, read_only=True); ws = wb["data"]
    rr = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
    f = np.array([r[0] for r in rr], float); c0 = np.array([r[1] for r in rr], float)
    c = np.interp(FREQ_AXIS, f, c0)
    samples.append((c, l, m))
print("spectra:", len(samples))

rng = np.random.default_rng(7)  # 与 mc_sensitivity.py 同种子
contestants = ['Double-L pg(oracle)', 'Double-L pg(routed)']
results = {k: [] for k in contestants}
class_acc = []  # 每个噪声级下 routed 分类准确率（诊断）

for sigma in noise_levels:
    tmp = {k: [] for k in contestants}
    n_ok = 0; n_tot = 0
    for c, l, m in samples:
        noise = rng.normal(0, sigma, size=(REPEAT, 400))
        cb = c + noise
        # oracle 标定系数（真实标签）
        k_o, b_o = GROUP_COEFFS[(float(l), float(m))]
        # routed: 分类器对每条噪声实现预测标签
        x = torch.tensor(cb, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            pred_cls = torch.argmax(classifier(x), 1).numpy()
        # 干净谱下分类是否对（仅诊断，sigma=0时）
        true_cls = [k for k,v in CLASS_TO_LM.items() if v==(int(l), int(m))]
        if sigma == 0.0 and true_cls:
            n_tot += 1
            if pred_cls[0] == true_cls[0]: n_ok += 1
        centers = [fit_double_center(FREQ_AXIS, cb[i]) for i in range(REPEAT)]
        # oracle 温度
        T_o = [(ce - b_o)/k_o if not np.isnan(ce) else np.nan for ce in centers]
        # routed 温度：逐实现按预测标签选 (k_i,b_i)
        T_r = []
        for i in range(REPEAT):
            ce = centers[i]
            if np.isnan(ce):
                T_r.append(np.nan); continue
            lm = CLASS_TO_LM[int(pred_cls[i])]
            k_r, b_r = GROUP_COEFFS[(float(lm[0]), float(lm[1]))]
            T_r.append((ce - b_r)/k_r)
        for k, vals in zip(contestants, [T_o, T_r]):
            v = np.array(vals, float); v = v[~np.isnan(v)]
            if len(v) > 1: tmp[k].append(np.std(v, ddof=1))
    for k in contestants:
        results[k].append(float(np.mean(tmp[k])) if tmp[k] else np.nan)
    if sigma == 0.0:
        class_acc.append(n_ok/max(n_tot,1))
    print(f"sigma={sigma:.4f} done: oracle={results[contestants[0]][-1]:.3f}, routed={results[contestants[1]][-1]:.3f}")

out = {'sigma': [float(x) for x in noise_levels],
       'results': results,
       'note': 'Double-L center fit identical to mc_sensitivity.fit_double; per-group inversion via (k_i,b_i) from 全功率组_温度系数标定.csv; oracle uses true power label, routed uses Stage-I argmax label (same noise realization).',
       'classifier_acc_sigma0': class_acc[0] if class_acc else None,
       'REPEAT': REPEAT, 'seed': 7}
json.dump(out, open('eval_models/mc_sensitivity_pgroup.json','w'), indent=1)
print("saved eval_models/mc_sensitivity_pgroup.json")
print("classifier acc at sigma=0 (routed):", class_acc[0] if class_acc else "n/a")
