import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob, re, warnings
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
import openpyxl
warnings.filterwarnings("ignore")

REAL_TEST_DIR = "newdata/测试集"
FREQ_AXIS = np.linspace(2858, 2878, 400)
DEVICE = "cpu"
std_k, std_b = np.load("eval_models/global_coeffs.npy")
print(f"global: k={std_k:.6f}, b={std_b:.4f}")

# ---------- network definitions (verbatim from 四模型测试（1）.py) ----------
class DualStreamNet(nn.Module):
    def __init__(self, aux_dim=2):
        super().__init__()
        self.shape_conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, 2, 3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 3, 2, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.amp_fc = nn.Sequential(nn.Linear(400, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU())
        self.aux_dim = aux_dim
        self.aux_fc = nn.Sequential(nn.Linear(aux_dim, 16), nn.ReLU()) if aux_dim > 0 else None
        fusion_dim = 64 + 32 + (16 if aux_dim > 0 else 0)
        self.fusion = nn.Sequential(nn.Linear(fusion_dim, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x, aux):
        x_min = x.min(dim=2, keepdim=True)[0]; x_max = x.max(dim=2, keepdim=True)[0]
        x_shape = (x - x_min) / (x_max - x_min + 1e-8) - 0.5
        fs = self.shape_conv(x_shape); fa = self.amp_fc(x.view(x.size(0), -1))
        if self.aux_dim > 0:
            return self.fusion(torch.cat((fs, fa, self.aux_fc(aux)), 1))
        return self.fusion(torch.cat((fs, fa), 1))

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
        self.prep = nn.Sequential(nn.Conv1d(1, 32, 7, 2, 3, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(3, 2, 1))
        self.layer1 = self._mk(32, 64, 1); self.layer2 = self._mk(64, 128, 2); self.layer3 = self._mk(128, 256, 2)
        self.avgpool = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(256, 9)
    def _mk(self, a, b, s): return nn.Sequential(ResBlock(a, b, s), ResBlock(b, b, 1))
    def forward(self, x):
        xm = x.min(dim=2, keepdim=True)[0]; xM = x.max(dim=2, keepdim=True)[0]
        x = (x - xm) / (xM - xm + 1e-8) - 0.5
        x = self.prep(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.fc(self.avgpool(x).flatten(1))

CLASS_TO_AUX = {0:[0.2,-1.0],1:[0.2,0.0],2:[0.2,1.0],3:[0.5,-1.0],4:[0.5,0.0],5:[0.5,1.0],6:[1.0,-1.0],7:[1.0,0.0],8:[1.0,1.0]}

model_full = DualStreamNet(2); model_full.load_state_dict(torch.load("eval_models/model_full.pth", map_location="cpu")); model_full.eval()
model_blind = DualStreamNet(0); model_blind.load_state_dict(torch.load("eval_models/model_blind.pth", map_location="cpu")); model_blind.eval()
model_laser = DualStreamNet(1); model_laser.load_state_dict(torch.load("eval_models/model_laser.pth", map_location="cpu")); model_laser.eval()
model_mw = DualStreamNet(1); model_mw.load_state_dict(torch.load("eval_models/model_mw.pth", map_location="cpu")); model_mw.eval()
classifier = ResNet1D(); classifier.load_state_dict(torch.load("eval_models/model_power_classifier_9groups.pth", map_location="cpu")); classifier.eval()
print("models loaded.")

# ---------- traditional fits (verbatim) ----------
def single_lorentz(x, y0, A, x0, w): return y0 + A*w*w/((x-x0)**2+w*w)
def fit_single(f, c):
    try:
        y_max, y_min = np.max(c), np.min(c); depth = y_min-y_max
        p0 = [y_max, depth, f[np.argmin(c)], 3.0]
        bounds = ([y_max-0.05, -1.0, 2800, 0.5],[y_max+0.05, 0.0, 2950, 20.0])
        popt, _ = curve_fit(single_lorentz, f, c, p0=p0, bounds=bounds, maxfev=2000)
        return (popt[2]-std_b)/std_k
    except: return np.nan
def double_lorentz(x, y0, A1, x1, w1, A2, x2, w2):
    return y0 + A1*w1*w1/((x-x1)**2+w1*w1) + A2*w2*w2/((x-x2)**2+w2*w2)
def fit_double(f, c):
    try:
        y_max, y_min = np.max(c), np.min(c); depth = y_min-y_max
        idx = np.argmin(c); fd = f[idx]
        p0 = [y_max, depth/2, fd-2, 3.0, depth/2, fd+2, 3.0]
        bounds = ([y_max-0.05,-1.0,2800,0.5,-1.0,2800,0.5],[y_max+0.05,0.0,2950,20.0,0.0,2950,20.0])
        popt, _ = curve_fit(double_lorentz, f, c, p0=p0, bounds=bounds, maxfev=2000)
        return ((popt[2]+popt[5])/2.0 - std_b)/std_k
    except: return np.nan

def parse_name(fn):
    fn2 = os.path.basename(fn).replace("（","(").replace("）",")").replace(" ","")
    t = re.search(r'(\d+(\.\d+)?)[°℃]', fn2); l = re.search(r'(\d+)%', fn2); m = re.search(r'([-]?\d+)dbm', fn2, re.I)
    return (float(t.group(1)), float(l.group(1)), float(m.group(1))) if t and l else None

rows = []
files = [f for f in glob.glob(os.path.join(REAL_TEST_DIR, "*.xlsx")) if "~$" not in f]
print("files:", len(files))
with torch.no_grad():
    for fp in sorted(files):
        meta = parse_name(fp)
        if not meta: continue
        t_real, l, m = meta
        try:
            wb = openpyxl.load_workbook(fp, read_only=True); ws = wb["data"]
            rr = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
            f_raw = np.array([r[0] for r in rr], float); c_raw = np.array([r[1] for r in rr], float)
        except: continue
        c = np.interp(FREQ_AXIS, f_raw, c_raw)
        x = torch.tensor(c, dtype=torch.float32).view(1,1,400)
        full_aux = torch.tensor([l/100.0, m/5.0], dtype=torch.float32).view(1,2)
        row = dict(T=t_real, l=l, m=m)
        row['Single L.'] = fit_single(FREQ_AXIS, c)
        row['Double L.'] = fit_double(FREQ_AXIS, c)
        row['Full(true)'] = model_full(x, full_aux).item()*60+30
        row['Blind'] = model_blind(x, None).item()*60+30
        row['LaserOnly'] = model_laser(x, full_aux[:,0:1]).item()*60+30
        row['MWOnly'] = model_mw(x, full_aux[:,1:2]).item()*60+30
        logits = classifier(x); pred_cls = torch.argmax(logits,1).item()
        va = CLASS_TO_AUX[pred_cls]
        row['VirtualCascade'] = model_full(x, torch.tensor([va],dtype=torch.float32)).item()*60+30
        m_norm = -1.0 if m < -2.5 else (1.0 if m > 2.5 else 0.0)
        row['ClassOK'] = 1 if (abs(l/100.0-va[0])<1e-6 and abs(m_norm-va[1])<1e-6) else 0
        row['pred_cls'] = pred_cls
        rows.append(row)

import json
json.dump(rows, open("eval_models/eval_results.json","w"), ensure_ascii=False, indent=1)
T = np.array([r['T'] for r in rows])
print(f"\nn = {len(rows)}")
print(f"{'method':<16}{'MAE':>8}{'RMSE':>8}{'P95':>8}{'max':>8}")
for k in ['Single L.','Double L.','Blind','LaserOnly','MWOnly','Full(true)','VirtualCascade']:
    e = np.abs(np.array([r[k] for r in rows]) - T)
    ok = ~np.isnan(e); e = e[ok]
    print(f"{k:<16}{e.mean():>8.4f}{np.sqrt((e**2).mean()):>8.4f}{np.percentile(e,95):>8.2f}{e.max():>8.2f}   (n={ok.sum()}, dropped={(~ok).sum()})")
acc = np.mean([r['ClassOK'] for r in rows])*100
print(f"\nStage-I classifier real-data accuracy: {acc:.2f}%")
# confusion matrix vs true class
from collections import Counter
true_cls = []
for r in rows:
    li = 0 if r['l']<35 else (1 if r['l']<75 else 2)
    mi = 0 if r['m']<-2.5 else (1 if r['m']<2.5 else 2)
    true_cls.append(li*3+mi)
cm = np.zeros((9,9), int)
for t,p in zip(true_cls, [r['pred_cls'] for r in rows]): cm[t,p]+=1
print("confusion (rows=true, cols=pred):")
print(cm)
# errors conditioned on classification
T_ = T
ve = np.abs(np.array([r['VirtualCascade'] for r in rows])-T_)
okc = np.array([r['ClassOK'] for r in rows])
print("VirtualCascade MAE on correctly-classified: %.4f (n=%d) | misclassified: %.4f (n=%d)" % (
    ve[okc==1].mean(), (okc==1).sum(), ve[okc==0].mean() if (okc==0).sum() else float('nan'), (okc==0).sum()))
