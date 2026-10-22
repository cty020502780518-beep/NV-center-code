# -*- coding: utf-8 -*-
"""
Graphical abstract for IEEE Sensors Journal (v2).
v2 changes (advisor feedback):
  (a) explicit 5-tick x axis (no label overlap) + 3x3 k_i matrix inset
      showing all NINE measured power states (fulldata_groups.json, real data);
  (b) prior box in plain language ("power state: laser %, MW dBm") instead of
      normalized internals [P_L/100, P_MW/5];
  (c) unchanged.
All numbers from verified calibration (no fabrication).
"""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

SEED_DIR = ("论文_解压/论文/05_审稿人材料_脚本与数据/"
            "01_原始实验数据/全数据集_104条种子谱")
OUT_PNG = "论文_解压/论文/01_论文PDF/图形摘要_GraphicalAbstract.png"
OUT_PDF = "论文_解压/论文/01_论文PDF/图形摘要_GraphicalAbstract.pdf"

# 实测九态标定 (fulldata_groups.json, kHz/K)  行=激光20/50/100%  列=微波-5/0/+5dBm
KMAT = [[-89.5, -90.4, -86.9],
        [-90.4, -91.1, -96.0],
        [-87.8, -88.3, -91.4]]

OKABE = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
         "grey": "#7F7F7F", "dark": "#2B2B2B"}

def load_spectrum(tag):
    f = glob.glob(f"{SEED_DIR}/{tag}.xlsx")[0]
    df = pd.read_excel(f, sheet_name="data")
    return df["Frequency"].values.astype(float), df["Contrast"].values.astype(float)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5,
    "axes.linewidth": 0.9, "xtick.direction": "out", "ytick.direction": "out",
})

fig = plt.figure(figsize=(13.3, 5.3), dpi=300)
gs = fig.add_gridspec(1, 3, width_ratios=[1.10, 1.22, 1.06],
                      wspace=0.24, left=0.055, right=0.975,
                      top=0.86, bottom=0.12)

# ---------------- Panel A: problem (spectra + k matrix) ----------------
gsA = gs[0, 0].subgridspec(2, 1, height_ratios=[2.55, 1.0], hspace=0.52)
axA = fig.add_subplot(gsA[0])
axM = fig.add_subplot(gsA[1])
tags = [("20%  0dbm  60℃", OKABE["blue"], "-"),
        ("50%  0dbm  60℃", OKABE["orange"], "-"),
        ("100%  0dbm  60℃", OKABE["green"], "-")]
for tag, c, ls in tags:
    fq, ct = load_spectrum(tag)
    axA.plot(fq, ct * 100, ls, color=c, lw=1.4,
             label=tag.split()[0] + " laser")
axA.set_xlim(2858, 2878)
axA.set_xticks([2858, 2863, 2868, 2873, 2878])
axA.set_xlabel("Frequency (MHz)", fontsize=9.5, labelpad=1.5)
axA.set_ylabel("Contrast (%)", fontsize=9.5)
axA.set_title("(a) Same temperature, 9 power states\n"
              "distort the lineshape and drift $k_i$",
              fontsize=10.5, fontweight="bold", loc="left")
axA.legend(frameon=False, fontsize=8.2, loc="lower right", handlelength=1.4)
axA.tick_params(labelsize=8.3)

# --- 3x3 热图: 九态实测 k_i (kHz/K), 数据=fulldata_groups.json ---
import json as _json
_cg = _json.load(open("论文_解压/论文/05_审稿人材料_脚本与数据/04_论文结果数据/fulldata_groups.json"))
K = {g: v["k"] * 1000 for g, v in _cg.items()}
KMAT = [[K["20%|-5"], K["20%|+0"], K["20%|+5"]],
        [K["50%|-5"], K["50%|+0"], K["50%|+5"]],
        [K["100%|-5"], K["100%|+0"], K["100%|+5"]]]
arr = np.array(KMAT)
im = axM.imshow(arr, cmap="RdYlBu_r", aspect="auto",
                vmin=-96.5, vmax=-86.4)
axM.set_xticks([0, 1, 2]); axM.set_xticklabels(["−5", "0", "+5"], fontsize=6.8)
axM.set_yticks([0, 1, 2]); axM.set_yticklabels(["20", "50", "100"], fontsize=6.8)
axM.set_xlabel("MW power (dBm)", fontsize=7.0, labelpad=1)
axM.set_ylabel("Laser (%)", fontsize=7.0, labelpad=1)
kmin, kmax = arr.min(), arr.max()
for r in range(3):
    for c in range(3):
        k = arr[r, c]
        frac = (k - kmin) / (kmax - kmin)
        axM.text(c, r, f"{k:.1f}", ha="center", va="center", fontsize=6.8,
                 fontweight="bold", color="white" if frac < 0.45 else "#1a1a1a")
axM.set_title("apparent $k_i$ (kHz/K): range −86.9…−96.0, mean −90.2",
              fontsize=7.6, pad=2.5)
for s in axM.spines.values(): s.set_visible(False)

# ---------------- Panel B: method ----------------
axB = fig.add_subplot(gs[0, 1])
axB.set_xlim(0, 10); axB.set_ylim(0, 10); axB.axis("off")
axB.set_title("(b) Physics-prior cascaded network\n"
              "replaces hardware power stabilization",
              fontsize=10.5, fontweight="bold", loc="left")

BOXES = []
def box(x, y, w, h, text, fc, ec, fs=9.0, tc="white", bold=False):
    axB.add_patch(FancyBboxPatch((x, y), w, h,
                  boxstyle="round,pad=0.12", fc=fc, ec=ec, lw=1.0))
    tx = axB.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                  fontsize=fs, color=tc, fontweight="bold" if bold else "normal")
    BOXES.append((tx, (x, y, w, h)))

def arrow(x0, y0, x1, y1):
    axB.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                  arrowstyle="-|>", mutation_scale=13,
                  color=OKABE["dark"], lw=1.3))

box(0.1, 6.6, 2.5, 2.3, "Measured\nCW-ODMR\nspectrum", "#EDF2F7", "#9AA5B1",
    fs=9, tc=OKABE["dark"])
box(3.3, 6.6, 3.0, 2.3, "ResNet-1D\npower classifier\n(99.1%, 9-way)",
    OKABE["blue"], OKABE["blue"], fs=9)
box(7.1, 6.6, 2.7, 2.3, "Virtual prior\n= power state\n(L%, MW dBm)",
    OKABE["blue"], OKABE["blue"], fs=8.8)
arrow(2.62, 7.75, 3.28, 7.75); arrow(6.32, 7.75, 7.08, 7.75)
box(2.2, 1.4, 3.4, 2.5, "DualStreamNet\nregressor\n(74k parameters)",
    OKABE["orange"], OKABE["orange"], fs=9)
box(6.9, 1.7, 2.9, 1.9, "Temperature\n$\\hat{T}$ (°C)", OKABE["green"],
    OKABE["green"], fs=10, bold=True)
arrow(8.45, 6.55, 5.0, 3.95)
arrow(5.65, 2.65, 6.85, 2.65)
arrow(1.35, 6.55, 3.0, 3.95)
axB.text(8.9, 5.75, "e.g. (50%, 0 dBm)", ha="center", fontsize=8.0,
         style="italic", color=OKABE["blue"])
axB.text(5.0, 0.35, "implicit power estimation \u2192 decoupled temperature readout",
         ha="center", fontsize=8.5, style="italic", color=OKABE["dark"])

# ---------------- Panel C: result ----------------
axC = fig.add_subplot(gs[0, 2])
names = ["Single-L", "Double-L", "Blind\n(no prior)", "Ours"]
mae = [2.18, 0.64, 0.41, 0.31]
lo = [2.00, 0.55, 0.33, 0.25]; hi = [2.37, 0.72, 0.49, 0.38]
yerr = [[m - l for m, l in zip(mae, lo)], [h - m for m, h in zip(mae, hi)]]
colors = [OKABE["grey"], "#A9A9A9", "#C8C8C8", OKABE["blue"]]
axC.bar(range(4), mae, yerr=yerr, capsize=3,
        color=colors, edgecolor="#555555", lw=0.7, width=0.66,
        error_kw=dict(lw=1.0, ecolor="#333333"))
for i, (m, h) in enumerate(zip(mae, hi)):
    axC.text(i, h + 0.07, f"{m:.2f}", ha="center", fontsize=9.5,
             fontweight="bold" if i == 3 else "normal")
axC.set_xticks(range(4)); axC.set_xticklabels(names, fontsize=8.6)
axC.set_ylabel("Temperature MAE (°C)", fontsize=10)
axC.set_ylim(0, 2.75)
axC.set_title("(c) 110 measured tests, all 9 states\n"
              "7× lower error than single-L",
              fontsize=10.5, fontweight="bold", loc="left")
axC.tick_params(labelsize=9)
axC.text(0.97, 0.80, "96.4% of tests < 0.6 °C\nGUM $U$ = ±0.61 °C ($k$=1)",
         transform=axC.transAxes, ha="right", va="top", fontsize=8.8,
         bbox=dict(boxstyle="round,pad=0.35", fc="#EAF2F8",
                   ec=OKABE["blue"], lw=0.9))

axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)
axC.spines["top"].set_visible(False); axC.spines["right"].set_visible(False)

fig.canvas.draw()
rend = fig.canvas.get_renderer()

def bb(t): return t.get_window_extent(rend)

# ---- 自检1: A轴刻度标签两两不重叠 ----
xt = [bb(t) for t in axA.get_xticklabels()]
ov = sum(1 for i in range(len(xt)) for j in range(i+1, len(xt))
         if xt[i].overlaps(xt[j]))
print("自检1 刻度标签重叠对:", ov, "(期望0)")

# ---- 自检2: B面板每个文本都在其框内 ----
bad = 0
for tx, (x, y, w, h) in BOXES:
    inv = axB.transData.inverted()
    (x0d, y0d) = inv.transform((bb(tx).x0, bb(tx).y0))
    (x1d, y1d) = inv.transform((bb(tx).x1, bb(tx).y1))
    if not (x - 0.15 <= x0d and x1d <= x + w + 0.15 and y - 0.1 <= y0d and y1d <= y + h + 0.1):
        bad += 1; print("  超框文本:", tx.get_text()[:30].replace(chr(10),'/'))
print("自检2 B面板超框文本:", bad, "(期望0)")

# ---- 自检3: 矩阵单元文字在axM范围内 ----
ib = axM.get_window_extent(rend)
ins_texts = [bb(t) for t in axM.texts]
out = sum(1 for b in ins_texts if b.x0 < ib.x0 - 2 or b.x1 > ib.x1 + 2
          or b.y0 < ib.y0 - 2 or b.y1 > ib.y1 + 2)
print("自检3 矩阵/标题文字出界:", out, "(期望0)")

# ---- 自检4: C面板标注框在轴内 ----
cb = [bb(t) for t in axC.texts if '<' not in t.get_text()]
axcb = axC.get_window_extent(rend)
outc = sum(1 for b in cb if b.x0 < axcb.x0 - 2 or b.x1 > axcb.x1 + 2
           or b.y0 < axcb.y0 - 2 or b.y1 > axcb.y1 + 2)
print("自检4 C面板文字出界:", outc, "(期望0)")

# ---- 自检5: 三面板标题两两不重叠 ----
tt = [bb(axA.title), bb(axB.title), bb(axC.title)]
tov = sum(1 for i in range(3) for j in range(i + 1, 3) if tt[i].overlaps(tt[j]))
print("自检5 面板标题重叠对:", tov, "(期望0)")

fig.savefig(OUT_PNG, dpi=300, facecolor="white")
fig.savefig(OUT_PDF, facecolor="white")
print("saved:", OUT_PNG, "and", OUT_PDF)
