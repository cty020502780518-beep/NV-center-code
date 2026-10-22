import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np, torch, torch.nn as nn, warnings
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")
FREQ = np.linspace(2858, 2878, 400)

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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm1d(out_channels))
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
mk = lambda p, ax=None: (lambda m: (m.load_state_dict(torch.load(p, map_location="cpu")), m.eval(), m)[-1])(DualStreamNet(ax))
models = {'Blind': mk('eval_models/model_blind.pth', 0), 'Laser': mk('eval_models/model_laser.pth', 1), 'MW': mk('eval_models/model_mw.pth', 1), 'Full': mk('eval_models/model_full.pth', 2)}
clf = ResNet1D(); clf.load_state_dict(torch.load('eval_models/model_power_classifier_9groups.pth', map_location='cpu')); clf.eval()
std_k, std_b = np.load('eval_models/global_coeffs.npy')

d = torch.load('eval_models/测试集_5k_aligned.pth')
X, Y, Aux = d['X'], d['Y'].squeeze(), d['Aux']
print("synthetic samples:", len(X))
aux_all = Aux.float()

import random
idx = torch.randperm(len(X))
N_FIT = 1500  # Lorentzian fits computed on first N_FIT for speed; models on ALL
fit_idx = idx[:N_FIT]

def single_L(x, y0, A, x0, w): return y0 + A*w*w/((x-x0)**2+w**2)
def double_L(x, y0, A1, x1, w1, A2, x2, w2): return y0 + A1*w1*w1/((x-x1)**2+w1**2) + A2*w2*w2/((x-x2)**2+w2**2)
def fitS(c):
    try:
        y_max, y_min = c.max(), c.min(); dep = y_min-y_max
        p0 = [y_max, dep, FREQ[np.argmin(c)], 3.0]
        b_ = ([y_max-0.05,-1.0,2800,0.5],[y_max+0.05,0.0,2950,20.0])
        p, _ = curve_fit(single_L, FREQ, c, p0=p0, bounds=b_, maxfev=1500)
        return (p[2]-std_b)/std_k
    except Exception: return np.nan
def fitD(c):
    try:
        y_max, y_min = c.max(), c.min(); dep = y_min-y_max
        fd = FREQ[np.argmin(c)]
        p0 = [y_max, dep/2, fd-2, 3.0, dep/2, fd+2, 3.0]
        b_ = ([y_max-0.05,-1.0,2800,0.5,-1.0,2800,0.5],[y_max+0.05,0.0,2950,20.0,0.0,2950,20.0])
        p, _ = curve_fit(double_L, FREQ, c, p0=p0, bounds=b_, maxfev=1500)
        return ((p[2]+p[5])/2-std_b)/std_k
    except Exception: return np.nan

res = {}
Xf = X[fit_idx].numpy().squeeze(); Yf = Y[fit_idx].numpy()
for name, fn, key in [('Single L.', fitS, 'Single L.'), ('Double L.', fitD, 'Double L.')]:
    preds = np.array([fn(c) for c in Xf])
    ok = ~np.isnan(preds); e = np.abs(preds[ok]-Yf[ok])
    res[key] = (float(e.mean()), float(np.sqrt((e**2).mean())), float(np.percentile(e,95)), int(ok.sum()))
    print(key, res[key])
    if name=='Single L.': np.save('/tmp/synth_single_err.npy', e)
    else: np.save('/tmp/synth_double_err.npy', e)

with torch.no_grad():
    names = {}
    bs = 512
    for i0 in range(0, len(X), bs):
        xb = X[i0:i0+bs].float(); ab = aux_all[i0:i0+bs]
        for k, m in models.items():
            a = {'Blind': None, 'Laser': ab[:,0:1], 'MW': ab[:,1:2], 'Full': ab}[k]
            names.setdefault(k, []).append(m(xb, a).squeeze()*60+30)
        names.setdefault('Virtual', []).append(models['Full'](xb, torch.tensor([CLASS_TO_AUX[int(torch.argmax(clf(xb[j:j+1]),1))] for j in range(len(xb))], dtype=torch.float32)).squeeze()*60+30)
    T = Y.numpy()
    for k, v in names.items():
        p = torch.cat(v).numpy(); e = np.abs(p - T)
        key = {'Blind':'Blind AI','Laser':'Laser only','MW':'MW only','Full':'Full (recorded)','Virtual':'Full (virtual)'}[k]
        res[key] = (float(e.mean()), float(np.sqrt((e**2).mean())), float(np.percentile(e,95)), int(len(e)))
        print(key, res[key])
        np.save(f'/tmp/synth_{k}_err.npy', e)
        # accuracy of classifier
logits = []
with torch.no_grad():
    for i0 in range(0, len(X), 512): logits.append(clf(X[i0:i0+512].float()))
pred = torch.cat(logits).argmax(1).numpy()
true_cls = (aux_all[:,0].numpy()*2).round().astype(int)*3 + (aux_all[:,1].numpy()+1).round().astype(int)
acc = (pred == true_cls).mean()
print("Stage-I synthetic accuracy: %.4f" % acc)
res['stage1_acc_synth'] = acc
import json
json.dump(res, open('eval_models/synth_eval.json','w'), indent=2)
print("saved eval_models/synth_eval.json")
