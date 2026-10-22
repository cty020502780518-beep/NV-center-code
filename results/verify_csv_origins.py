import glob, re, os, json
import numpy as np, pandas as pd
from scipy.optimize import curve_fit

def double_lorentz(x, y0, A1, x1, w1, A2, x2, w2):
    L1 = A1*(w1**2)/((x-x1)**2+w1**2); L2 = A2*(w2**2)/((x-x2)**2+w2**2)
    return y0 + L1 + L2
def get_D(freqs, counts):
    y_max=np.max(counts); y_min=np.min(counts); depth=y_min-y_max
    xc=freqs[np.argmin(counts)]
    p0=[y_max, depth/2, xc-2, 3.0, depth/2, xc+2, 3.0]
    try:
        popt,_=curve_fit(double_lorentz, freqs, counts, p0=p0,
            bounds=([y_max-0.1,-1.0,2800,0.5,-1.0,2800,0.5],[y_max+0.1,0.0,2950,20.0,0.0,2950,20.0]),
            maxfev=5000)
        return (popt[2]+popt[5])/2
    except Exception: return None
def parse(name):
    name=name.replace("（","(").replace("）",")").replace(" ","")
    t=re.search(r'(\d+(\.\d+)?)[°℃]',name); l=re.search(r'(\d+)%',name); m=re.search(r'([-]?\d+)dbm',name,re.I)
    if not t or not l: return None
    return (float(l.group(1)), float(m.group(1)) if m else 0.0, float(t.group(1)))

def scan(folder):
    import openpyxl
    g={}
    for fp in glob.glob(os.path.join(folder,"**","*.xlsx"),recursive=True):
        if "~$" in fp: continue
        p=parse(os.path.basename(fp))
        if not p: continue
        try:
            wb=openpyxl.load_workbook(fp,read_only=True); ws=wb['data']
            rr=[r for r in ws.iter_rows(min_row=2,values_only=True) if r[0] is not None]
        except Exception: continue
        f=np.array([r[0] for r in rr],float); c=np.array([r[1] for r in rr],float)
        D=get_D(f,c)
        if D and 2860<D<2880: g.setdefault((p[0],p[1]),[]).append((p[2],D))
    out={}
    for k,v in g.items():
        a=np.array(v); sl,ic=np.polyfit(a[:,0],a[:,1],1)
        pr=np.polyval([sl,ic],a[:,0]); r2=1-((a[:,1]-pr)**2).sum()/(((a[:,1]-a[:,1].mean())**2).sum()+1e-8)
        out[k]=(sl*1000,r2,len(v))
    return out

csv=pd.read_csv('analysis/全功率组_温度系数标定.csv')
A=scan('fulldata')
try:
    B=scan('/mnt/c/Users/tianyu.chu/Desktop/论文/全数据集/全数据集')
except Exception as e:
    print("desk scan err",e); B={}
print(f"{'group':12} {'CSV_k':>8} {'104_k':>8} {'209_k':>8} | {'CSV_R2':>7} {'104_R2':>7} {'209_R2':>7} | n: CSV? /104 /209")
for _,r in csv.iterrows():
    k=(r['Laser (%)'],r['MW (dBm)'])
    a=A.get(k,(np.nan,np.nan,0)); b=B.get(k,(np.nan,np.nan,0))
    print(f"{str(k):12} {r['Slope']*1000:8.3f} {a[0]:8.3f} {b[0]:8.3f} | {r['R2']:7.4f} {a[1]:7.4f} {b[1]:7.4f} | n: {'?':>3} {a[2]:>3} {b[2]:>3}")
