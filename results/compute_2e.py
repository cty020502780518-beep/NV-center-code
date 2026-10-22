import glob, re, os, json
import numpy as np, openpyxl
from scipy.optimize import curve_fit
def double_lorentz(x, y0, A1, x1, w1, A2, x2, w2):
    return y0 + A1*w1*w1/((x-x1)**2+w1*w1) + A2*w2*w2/((x-x2)**2+w2*w2)
def fit_double(f, c):
    y_max, y_min = np.max(c), np.min(c); depth = y_min-y_max
    idx = np.argmin(c); fd = f[idx]
    p0 = [y_max, depth/2, fd-2, 3.0, depth/2, fd+2, 3.0]
    bounds = ([y_max-0.05,-1.0,2800,0.5,-1.0,2800,0.5],
              [y_max+0.05, 0.0, 2950,20.0, 0.0,2950,20.0])
    popt,_ = curve_fit(double_lorentz, f, c, p0=p0, bounds=bounds, maxfev=4000)
    return sorted([popt[2], popt[5]])
pat = re.compile(r'(\d+)%\s*([+-]?\d+)\s*[dD][bB][mM].*?(\d+)\s*')
groups={}; nfail=0
for fp in sorted(glob.glob('fulldata/*.xlsx')):
    if "~$" in fp: continue
    base=os.path.basename(fp)
    m=pat.search(base)
    if not m: continue
    laser,mw,Traw=base.split('%')[0],m.group(2),m.group(3)
    wb=openpyxl.load_workbook(fp,read_only=True); ws=wb['data']
    rr=[r for r in ws.iter_rows(min_row=2,values_only=True) if r[0] is not None]
    f=np.array([r[0] for r in rr],float); c=np.array([r[1] for r in rr],float)
    try:
        f1,f2=fit_double(f,c); split=f2-f1
    except Exception:
        nfail+=1; continue
    T=float(Traw)
    groups.setdefault((laser,int(mw)),[]).append((T,split))
print("fit failures:",nfail)
out={}
for key in sorted(groups):
    arr=np.array(sorted(groups[key])); T=arr[:,0]; S=arr[:,1]
    r=np.corrcoef(T,S)[0,1] if len(T)>=4 else float('nan')
    out[f"{key[0]}%|{key[1]:+d}"]={'n':int(len(T)),'corr':round(float(r),3),'split_mean_MHz':round(float(S.mean()),3),'split_std_MHz':round(float(S.std()),4)}
    print(f"{key}: n={len(T)}, corr(T,2E)={r:+.3f}, mean split={S.mean():.3f} MHz")
json.dump(out,open('analysis/corr_2e_T.json','w'),ensure_ascii=False,indent=1)
print("saved analysis/corr_2e_T.json")
