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
        self.fusion = nn.Sequential(nn.Linear(64+32+(16 if aux_dim>0 else 0), 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x, aux):
        xm = x.min(dim=2, keepdim=True)[0]; xM = x.max(dim=2, keepdim=True)[0]
        xs = (x - xm) / (xM - xm + 1e-8) - 0.5
        fs = self.shape_conv(xs); fa = self.amp_fc(x.view(x.size(0), -1))
        return self.fusion(torch.cat((fs, fa, self.aux_fc(aux)), 1)) if self.aux_dim > 0 else self.fusion(torch.cat((fs, fa), 1))

def single_lorentz(x, y0, A, x0, w): return y0 + A*w*w/((x-x0)**2+w*w)
def double_lorentz(x, y0, A1, x1, w1, A2, x2, w2): return y0 + A1*w1*w1/((x-x1)**2+w1**2) + A2*w2*w2/((x-x2)**2+w2**2)
def fit_single(f, c):
    try:
        y_max, y_min = c.max(), c.min(); dep = y_min-y_max
        p0=[y_max, dep, f[np.argmin(c)], 3.0]
        p,_=curve_fit(single_lorentz,f,c,p0=p0,bounds=([y_max-0.05,-1,2800,0.5],[y_max+0.05,0,2950,20]),maxfev=1500)
        return (p[2]-std_b)/std_k
    except: return np.nan
def fit_double(f, c):
    try:
        y_max, y_min = c.max(), c.min(); dep = y_min-y_max; fd=f[np.argmin(c)]
        p0=[y_max, dep/2, fd-2, 3.0, dep/2, fd+2, 3.0]
        p,_=curve_fit(double_lorentz,f,c,p0=p0,bounds=([y_max-0.05,-1,2800,0.5,-1,2800,0.5],[y_max+0.05,0,2950,20,0,2950,20]),maxfev=1500)
        return ((p[2]+p[5])/2-std_b)/std_k
    except: return np.nan

models = {}
for name, po, ax_dim in [('Blind','model_blind.pth',0),('Laser Only','model_laser.pth',1),('MW Only','model_mw.pth',1),('Full Model','model_full.pth',2)]:
    m = DualStreamNet(ax_dim); m.load_state_dict(torch.load(f'eval_models/{po}', map_location='cpu')); m.eval(); models[name]=m

def parse_name(fn):
    s = os.path.basename(fn).replace("（","(").replace("）",")").replace(" ","")
    t = re.search(r'(\d+(\.\d+)?)[°℃]', s); l = re.search(r'(\d+)%', s); m = re.search(r'([-]?\d+)dbm', s, re.I)
    return (float(t.group(1)), float(l.group(1)), float(m.group(1))) if t and l else None

samples = []
for fp in sorted(glob.glob(os.path.join(REAL_TEST_DIR,"*.xlsx"))):
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

rng = np.random.default_rng(7)
contestants = ['Single Lor.', 'Double Lor.'] + list(models.keys())
results = {k: [] for k in contestants}
for sigma in noise_levels:
    tmp = {k: [] for k in contestants}
    for c, l, m in samples:
        noise = rng.normal(0, sigma, size=(REPEAT, 400))
        cb = c + noise
        x = torch.tensor(cb, dtype=torch.float32).unsqueeze(1)
        aux = torch.tensor([l/100., m/5.], dtype=torch.float32).repeat(REPEAT, 1)
        preds = {}
        with torch.no_grad():
            preds['Blind'] = models['Blind'](x, None).numpy().flatten()*60+30
            preds['Laser Only'] = models['Laser Only'](x, aux[:,0:1]).numpy().flatten()*60+30
            preds['MW Only'] = models['MW Only'](x, aux[:,1:2]).numpy().flatten()*60+30
            preds['Full Model'] = models['Full Model'](x, aux).numpy().flatten()*60+30
        ps = [fit_single(FREQ_AXIS, cb[i]) for i in range(REPEAT)]
        pd_ = [fit_double(FREQ_AXIS, cb[i]) for i in range(REPEAT)]
        for k, vals in zip(contestants, [ps, pd_, preds['Blind'], preds['Laser Only'], preds['MW Only'], preds['Full Model']]):
            v = np.array(vals, float); v = v[~np.isnan(v)]
            if len(v) > 1: tmp[k].append(np.std(v, ddof=1))
    for k in contestants:
        results[k].append(float(np.mean(tmp[k])) if tmp[k] else np.nan)
    print(f"sigma={sigma:.4f} done: full={results['Full Model'][-1]:.3f}, double={results['Double Lor.'][-1]:.3f}")

json.dump({'sigma': [float(x) for x in noise_levels], 'results': results}, open('eval_models/mc_sensitivity.json','w'), indent=1)
print("saved eval_models/mc_sensitivity.json")
