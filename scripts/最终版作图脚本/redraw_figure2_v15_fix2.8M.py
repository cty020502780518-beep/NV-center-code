# Figure 2 v6 — Stage-II band enlarged / Stage-I band slimmed; elbow composite connectors
# (one trunk that bends once and forks) replace all two-diagonal-arrow fans and merges.
# Six machine checks must all be zero before saving:
#   1) line crossings   2) text-text   3) text-on-image   4) text crossing box edges
#   5) sibling box overlap   6) line interior inside any non-container box
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
import numpy as np

plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9.5,'mathtext.fontset':'stix'})
M = "/tmp/pptx_x/ppt/media/"
W, H = 100.0, 62.0
fig = plt.figure(figsize=(16.2, 10.05))
ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off')

C = {'data':((0.96,0.94,0.88),(0.62,0.42,0.10)), 'set':((1,1,1),(0.62,0.42,0.10)),
     's1':((0.90,0.94,0.99),(0.25,0.42,0.68)), 'router':((0.96,0.92,0.98),(0.45,0.30,0.60)),
     'prior':((1.0,0.92,0.90),(0.75,0.25,0.20)), 's2':((0.90,0.97,0.90),(0.20,0.55,0.28)),
     'white':((1,1,1),(0.35,0.35,0.35)), 'note':((0.95,0.95,0.95),(0.45,0.45,0.45))}
RECTS=[]; CONTAINERS=set(); POLYS=[]
def box(x,y,w,h,fk,ec=None,lw=1.3,pad=0.12,z=3,container=False):
    fc,ec2 = C[fk] if isinstance(fk,str) else (fk,ec)
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f"round,pad={pad},rounding_size=0.5",
                                facecolor=fc,edgecolor=ec2,lw=lw,zorder=z))
    if container: CONTAINERS.add(len(RECTS))
    RECTS.append((x-pad,y-pad,w+2*pad,h+2*pad))
def txt(x,y,s,fs=9.0,col='0.12',bold=False,va='center',ha='center',z=6):
    ax.text(x,y,s,ha=ha,va=va,fontsize=fs,color=col,zorder=z,linespacing=1.42,
            fontweight='bold' if bold else 'normal')
def _bez(p,q,rad,n=26):
    p=np.array(p,float); q=np.array(q,float)
    if rad==0: return [tuple(p),tuple(q)]
    mid=(p+q)/2; d=q-p; ln=np.hypot(*d)+1e-12
    c=mid+np.array([-d[1],d[0]])/ln*rad*ln
    t=np.linspace(0,1,n)
    return [tuple((1-t_)*((1-t_)*p+t_*c)+t_*((1-t_)*c+t_*q)) for t_ in t]
def arr(p,q,rad=0.0,lw=1.7,col='0.22',z=4,fmt='-|>',conn=None,poly=None):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle=fmt,mutation_scale=15,lw=lw,color=col,zorder=z,
                                 shrinkA=2,shrinkB=2,connectionstyle=conn or f"arc3,rad={rad}"))
    if poly is not None: POLYS.append(poly)
    else: POLYS.append(_bez(p,q,rad))
def line(p,q,lw=1.7,col='0.22',z=4):
    ax.plot([p[0],q[0]],[p[1],q[1]],lw=lw,color=col,zorder=z,solid_capstyle='round')
    POLYS.append([tuple(p),tuple(q)])
def img(name,x,y,w,h,z=5,frame=True):
    if frame: box(x-0.16,y-0.16,w+0.32,h+0.32,(1,1,1),'0.55',lw=0.9,pad=0.05,z=z-1)
    r=fig.add_axes([x/W,y/H,w/W,h/H]); r.imshow(Image.open(M+name),aspect='auto'); r.axis('off'); r.set_zorder(z)

# ================= left column =================
box(1.6, 52.0, 16.6, 7.6, 'data', lw=1.4)
txt(9.9,57.7,"Measured seed spectra",fs=9.2,bold=True,col=C['data'][1])
txt(9.9,55.6,"$n=104$,  nine power states",fs=7.8,col='0.35')
img('image13.png', 6.6, 52.45, 1.5, 1.9); img('image12.png', 11.4, 52.45, 1.5, 1.9)

box(1.6, 42.0, 16.6, 7.8, 'data', lw=1.4)
txt(9.9,47.6,"Physics-constrained Sim2Real",fs=8.9,bold=True,col=C['data'][1])
txt(9.9,44.4,"shift along $k_iT{+}b_i$ + noise\nand baseline jitters\n$\\Rightarrow$ 55,000 synthetic",fs=7.6,col='0.25')

box(1.6, 32.6, 7.9, 7.0, 'set', lw=1.2); txt(5.55,36.1,"Training set\n50,000",fs=8.3,bold=True,col='0.2')
box(10.3, 32.6, 7.9, 7.0, 'set', lw=1.2); txt(14.25,36.1,"Testing set\n5,000",fs=8.3,bold=True,col='0.2')
txt(9.9,31.7,"disjoint physical seeds",fs=7.4,col='0.45')

box(1.6, 8.2, 16.6, 22.0, 'note', lw=1.2)
txt(9.9,28.4,"Training & split protocol",fs=8.8,bold=True,col='0.28')
txt(9.9,17.4,"Adam, MSE on $(T{-}30)/60$\nbatch 128; best checkpoint on\nthe disjoint synthetic test split\n\nStage I: 95.7% on 5k synthetic,\n99.1% on the 110 real spectra\n\nall learned layers frozen at\ndeployment",fs=7.7,col='0.3')

box(1.6, 0.6, 16.6, 6.4, 'data', lw=1.4)
txt(9.9,5.6,"110 validation spectra",fs=8.8,bold=True,col=C['data'][1])
txt(9.9,2.9,"9 states $\\times$ 40–80 °C, measured\n(never used for network training)",fs=7.3,col='0.35')

arr((9.9,52.0),(9.9,50.15))
line((9.9,41.9),(9.9,40.0)); line((5.55,40.0),(14.25,40.0))
arr((5.55,40.0),(5.55,39.75)); arr((14.25,40.0),(14.25,39.75))

# frozen-weights brace (clear of everything; above the rail)
ax.plot([19.9,19.9],[8.4,57.2],lw=2.0,color='0.58',zorder=4); POLYS.append([(19.9,8.4),(19.9,57.2)])
ax.plot([19.45,19.9],[57.2,57.2],lw=2.0,color='0.58',zorder=4)
ax.plot([19.45,19.9],[8.4,8.4],lw=2.0,color='0.58',zorder=4)
arr((18.45,36.1),(19.9,36.1),lw=1.4,col='0.5')
arr((19.9,56.0),(21.9,56.0),lw=1.2,col='0.58')
arr((19.9,21.0),(21.9,21.0),lw=1.2,col='0.58')

# ================= row 1: Stage I (slimmer) =================
box(22, 46.2, 76, 13.4, 's1', lw=1.5, pad=0.05, container=True)
txt(60,58.45,"Stage I  ·  ResNet-1D power-state routing  (frozen, 0.96M params)",fs=10.2,bold=True,col=C['s1'][1])
img('image7.png', 23.2, 49.1, 9.6, 6.6); txt(28.0,47.85,"400-pt spectrum",fs=7.5,col='0.35')
box(34.6, 48.9, 13.0, 7.0, 'white', C['s1'][1], lw=1.2)
txt(41.1,52.4,"prep + ResBlocks$\\times$6\nconv 32$\\to$64$\\to$128$\\to$256",fs=7.8,col='0.2')
img('image8.png', 49.4, 49.1, 7.8, 6.6); txt(53.3,47.85,"envelope features",fs=7.5,col='0.35')
box(59.0, 48.9, 8.6, 7.0, 'white', C['s1'][1], lw=1.2); txt(63.3,52.4,"GAP $\\to$ FC\n9-class\nsoftmax",fs=7.9,col='0.2')
img('image9_fixed.png', 68.9, 48.95, 11.07, 6.5, frame=False); txt(74.0,47.6,"softmax probabilities",fs=7.5,col='0.35')
box(81.4, 48.7, 15.2, 7.6, 'prior', lw=1.4)
txt(89.0,54.1,"Virtual power prior",fs=8.8,bold=True,col=C['prior'][1])
txt(89.0,50.7,"$a=[P_L/100,\\,P_{MW}/5]$\nargmax $\\to$ 1 of 9 states",fs=7.9,col='0.3')
arr((33.0,52.4),(34.5,52.4)); arr((47.8,52.4),(49.25,52.4)); arr((57.5,52.4),(58.9,52.4)); arr((67.85,52.4),(68.85,52.4)); arr((80.25,52.4),(81.32,52.4))
arr((89.0,48.58),(89.0,43.7))          # VP down into router band

# ================= row 2: prior router =================
box(22, 31.5, 76, 12.0, 'router', lw=1.5, pad=0.05, container=True)
txt(60,42.15,"Prior router  ·  4 ablation configurations (read right $\\to$ left)",fs=9.8,bold=True,col=C['router'][1])
rcols={'Full Model':(0.13,0.55,0.13),'Laser only':(0.12,0.56,1.00),'MW only':(0.58,0.44,0.86),'Blind':(0.95,0.55,0.10)}
raux={'Full Model':'$[P_L/100,\\,P_{MW}/5]$','Laser only':'$[P_L/100]$','MW only':'$[P_{MW}/5]$','Blind':'$\\varnothing$'}
for i,(nm,cc) in enumerate(rcols.items()):
    x=59.0+i*9.9
    box(x,33.7,9.2,6.3,(1,1,1),cc,lw=1.5,pad=0.05)
    txt(x+4.6,37.8,nm,fs=7.9,bold=True,col=cc); txt(x+4.6,35.5,raux[nm],fs=7.3,col=cc)
box(45.6, 33.9, 10.4, 6.0, 'white', C['router'][1], lw=1.2); txt(50.8,36.9,"prior embed.\nFC 2$\\to$16",fs=8.0,col='0.2')
box(23.0, 33.9, 11.6, 6.0, 'prior', lw=1.3); txt(28.8,36.9,"$f_{aux}\\in\\mathrm{R}^{16}$\n(normalized)",fs=8.2,col=C['prior'][1])
arr((58.85,36.9),(56.15,36.9)); arr((45.45,36.9),(34.85,36.9))
arr((29.0,33.82),(29.0,25.55))          # f_aux down into Stage-II concat

# ================= row 3: Stage II streams (enlarged) =================
box(22, 14.5, 76, 15.3, 's2', lw=1.5, pad=0.05, container=True)
txt(60,28.85,"Stage II  ·  dual-stream feature extraction (read right $\\to$ left)",fs=9.5,bold=True,col=C['s2'][1])
box(23.0, 17.35, 12.0, 8.0, (0.96,0.94,1.0), C['s2'][1], lw=1.3)
txt(29.0,21.35,"high-dim\nconcat\n$64{+}32{+}16$\n$=112$",fs=8.0,bold=True,col='0.15')
box(48.4, 21.7, 12.6, 6.3, 'white', C['s2'][1], lw=1.2)
txt(54.7,24.85,"morphology stream (Conv1D)\n1$\\to$16$\\to$32$\\to$64, GAP\n$f_{shape}\\in\\mathrm{R}^{64}$",fs=7.6,col='0.2')
box(48.4, 14.9, 12.6, 5.9, 'white', C['s2'][1], lw=1.2)
txt(54.7,17.85,"amplitude stream (Dense)\n400$\\to$128$\\to$32\n$f_{amp}\\in\\mathrm{R}^{32}$",fs=7.6,col='0.2')
img('image14.png', 78.4, 18.2, 17.4, 7.4); txt(87.1,16.55,"input spectrum (validation)",fs=7.5,col='0.35')
# composite FAN: one trunk out of the spectrum box, once bent, then two arrow stubs
line((78.25,21.47),(66.0,21.47)); line((66.0,17.85),(66.0,24.85))
arr((66.0,24.85),(61.15,24.85)); arr((66.0,17.85),(61.15,17.85))
# composite MERGE: two stubs into one bus, one arrow into concat
line((48.3,24.85),(44.5,24.85)); line((48.3,17.85),(44.5,17.85)); line((44.5,17.85),(44.5,24.85))
arr((44.5,21.35),(35.2,21.35))

# ================= row 4: readout + evaluation =================
box(22, 3.6, 76, 9.4, 'white', C['s2'][1], lw=1.3, pad=0.05, container=True)
txt(60,12.4,"Readout  ·  regression & real-set evaluation",fs=9.8,bold=True,col=C['s2'][1])
box(24.0, 4.8, 17.6, 6.8, 'white', C['s2'][1], lw=1.2)
txt(32.8,8.2,"MLP regressor\n112$\\to$64$\\to$32$\\to$1",fs=8.3,col='0.2')
box(46.0, 4.8, 17.0, 6.8, 'white', C['prior'][1], lw=1.4)
img('image5.png', 46.9, 5.2, 4.9, 4.9)
txt(56.6,8.2,"$\\hat T$ (°C)",fs=8.8,bold=True,col=C['prior'][1])
box(71.8, 4.2, 24.6, 7.4, (0.94,0.97,0.94), C['s2'][1], lw=1.4)
txt(84.2,9.7,"Evaluation on the 110 real spectra",fs=8.6,bold=True,col=C['s2'][1])
txt(84.2,6.3,"MAE 0.31$^\\circ$C  |  $P_{95}$ 0.55$^\\circ$C  |  $U_{tot}\\ \\pm0.84\\,^\\circ$C",fs=7.5,col='0.3')
arr((29.0,17.23),(29.0,11.7))                    # concat -> MLP
arr((41.85,8.2),(45.85,8.2)); arr((63.3,8.2),(71.7,8.2))
arr((18.35,2.05),(84.3,3.8),col='0.55',lw=1.6,conn="angle,angleA=0,angleB=90,rad=0",
    poly=[(18.35,2.05),(84.3,2.05),(84.3,3.8)])   # single elbow arrow pointing up at the eval box
txt(45.0,0.75,"validation spectra $\\rightarrow$ real-set evaluation",fs=7.5,col='0.5')

# ================= automated checks =================
def segx(a,b,c,d):
    def o(p,q,r): return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    d1,d2,d3,d4=o(c,d,a),o(c,d,b),o(a,b,c),o(a,b,d)
    return ((d1>1e-4)!=(d2>1e-4)) and ((d3>1e-4)!=(d4>1e-4))
K=[]
for P in POLYS: K.extend(zip(P[:-1],P[1:]))
cross=[]
for i,(a,b) in enumerate(K):
    for c,d in K[i+1:]:
        if min(abs(a[0]-c[0])+abs(a[1]-c[1]),abs(a[0]-d[0])+abs(a[1]-d[1]),
               abs(b[0]-c[0])+abs(b[1]-c[1]),abs(b[0]-d[0])+abs(b[1]-d[1]))>0.08 and segx(a,b,c,d):
            def onseg(p,u,v):
                return abs((v[0]-u[0])*(p[1]-u[1])-(v[1]-u[1])*(p[0]-u[0]))<1e-6 and \
                       min(u[0],v[0])-1e-6<=p[0]<=max(u[0],v[0])+1e-6 and min(u[1],v[1])-1e-6<=p[1]<=max(u[1],v[1])+1e-6
            if onseg(a,c,d) or onseg(b,c,d) or onseg(c,a,b) or onseg(d,a,b): continue
            cross.append((a,c))
print("1) LINE CROSSINGS:", len(cross))

fig.canvas.draw(); r=fig.canvas.get_renderer()
texts=[]
for t in ax.texts:
    bb=t.get_window_extent(r); i=ax.transData.inverted()
    (X0,Y0)=i.transform((bb.x0,bb.y0)); (X1,Y1)=i.transform((bb.x1,bb.y1))
    texts.append(((X0,X1,Y0,Y1),t.get_text()[:36]))
ov=[(t1,t2) for i,((a1,b1,c1,d1),t1) in enumerate(texts) for (a2,b2,c2,d2),t2 in texts[i+1:]
    if max(0,min(b1,b2)-max(a1,a2))>0.05 and max(0,min(d1,d2)-max(c1,c2))>0.05]
print("2) TEXT-TEXT:", len(ov), ov[:6])
imgs=[(a.get_position().x0*100,a.get_position().x1*100,a.get_position().y0*62,a.get_position().y1*62) for a in fig.axes[1:]]
ov2=[t for (x0,x1,y0,y1),t in texts for (ix0,ix1,iy0,iy1) in imgs
     if max(0,min(x1,ix1)-max(x0,ix0))>0.15 and max(0,min(y1,iy1)-max(y0,iy0))>0.15]
print("3) TEXT-ON-IMAGE:", len(ov2), ov2[:5])
tb=[]
for (x0,x1,y0,y1),t in texts:
    cx,cy=(x0+x1)/2,(y0+y1)/2
    for (bx,by,bw,bh) in RECTS:
        inside = bx<=cx<=bx+bw and by<=cy<=by+bh
        ox=max(0,min(x1,bx+bw)-max(x0,bx)); oy=max(0,min(y1,by+bh)-max(y0,by))
        if not inside and ox>0.08 and oy>0.08: tb.append((t[:22],(round(bx,1),round(by,1))))
print("4) TEXT CROSSING BOX EDGES:", len(tb), tb[:6])
bb=[]
for i,(x1,y1,w1,h1) in enumerate(RECTS):
    for j,(x2,y2,w2,h2) in enumerate(RECTS[i+1:],i+1):
        cx1,cy1,cx2,cy2=x1+w1/2,y1+h1/2,x2+w2/2,y2+h2/2
        ox=max(0,min(x1+w1,x2+w2)-max(x1,x2)); oy=max(0,min(y1+h1,y2+h2)-max(y1,y2))
        nest=(x1<=cx2<=x1+w1 and y1<=cy2<=y1+h1) or (x2<=cx1<=x2+w2 and y2<=cy1<=y2+h2)
        if ox>0.12 and oy>0.12 and not nest: bb.append(((round(x1,1),round(y1,1)),(round(x2,1),round(y2,1))))
print("5) SIBLING BOX OVERLAPS:", len(bb), bb[:6])
# 6) line interior inside non-container boxes
insidec=0; worst=[]
for k,(p,q) in enumerate(K):
    for f in np.linspace(0.2,0.8,13):
        x=p[0]+(q[0]-p[0])*f; y=p[1]+(q[1]-p[1])*f
        for m,(bx,by,bw,bh) in enumerate(RECTS):
            if m in CONTAINERS: continue
            if bx+0.05<x<bx+bw-0.05 and by+0.05<y<by+bh-0.05: insidec+=1; worst.append((round(x,1),round(y,1),k)); break
print("6) LINE-INTERIOR-IN-BOX:", insidec, worst[:8])

for ext,kw in [('png',dict(dpi=500)),('pdf',{})]:
    plt.savefig(f"/mnt/c/Users/tianyu.chu/Desktop/论文/论文新改/工作区/Figure2.{ext}",**kw)
print("Figure2 v15 saved (2.8M->0.96M, seeding->network training)")
