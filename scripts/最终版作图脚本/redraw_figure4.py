import json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
warnings.filterwarnings("ignore")

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 9.5,
                     'axes.linewidth': 0.8, 'mathtext.fontset': 'stix'})

# ---- real-data per-spectrum predictions ----
rows = list(csv.DictReader(open('eval_models/真实数据对比结果_110谱.csv', encoding='utf-8-sig')))
T = np.array([float(r['True_T_C']) for r in rows])
pred_virtual = np.array([float(r['VirtualCascade']) for r in rows])
pred_fullrec = np.array([float(r['Full_true_aux']) for r in rows])

# ---- numbers (reproduced with frozen weights & code) ----
real = {'Textbook\n(ideal)': 3.82, 'Single\nLor.': 2.18, 'Double\nLor.': 0.64, 'Blind\nAI': 0.41,
        'Laser\nonly': 0.33, 'MW\nonly': 0.41, 'Full\n(record.)': 0.32, 'Full\n(virtual)': 0.31}
synth = {'Single\nLor.': 4.89, 'Double\nLor.': 0.886, 'Blind\nAI': 0.346, 'Laser\nonly': 0.089,
         'MW\nonly': 0.343, 'Full\n(record.)': 0.077, 'Full\n(virtual)': 0.119}
real_exact = {'Full\n(record.)': 0.3204, 'Full\n(virtual)': 0.3066}
synth_exact = {'Full\n(record.)': 0.0766, 'Full\n(virtual)': 0.1191}

colors = {'Textbook\n(ideal)': '0.45', 'Single\nLor.': (0.45,0.47,0.52), 'Double\nLor.': (0.70,0.21,0.17),
          'Blind\nAI': (0.95,0.55,0.10), 'Laser\nonly': (0.12,0.56,1.00), 'MW\nonly': (0.58,0.44,0.86),
          'Full\n(record.)': (0.20,0.59,0.55), 'Full\n(virtual)': (0.13,0.55,0.13)}

fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))

def bar_panel(ax, data, title, ylabel='Mean absolute error ($^{\circ}$C)'):
    names = list(data); vals = [data[k] for k in names]
    bars = ax.bar(range(len(names)), vals, color=[colors[k] for k in names],
                  width=0.66, alpha=0.92, zorder=3)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals)*0.015, f'{v:.2f}', ha='center', fontsize=7.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.4)
    ax.set_title(title, loc='left', fontsize=10.5, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.28, ls='--')
    ax.set_ylim(0, max(vals)*1.16)

# (a) synthetic
bar_panel(axes[0,0], synth, '(a) MAE on synthetic test set (5,000 spectra)')
# (b) real
bar_panel(axes[0,1], real, '(b) MAE on real validation set (110 spectra)')

# (c) virtual vs recorded
ax = axes[1,0]
labels2 = ['Synthetic test', 'Real validation']
v_rec = [synth_exact['Full\n(record.)'], real_exact['Full\n(record.)']]
v_vir = [synth_exact['Full\n(virtual)'], real_exact['Full\n(virtual)']]
x = np.arange(2); w = 0.32
b1 = ax.bar(x-w/2, v_rec, w, color=colors['Full\n(record.)'], alpha=0.92, zorder=3, label='Recorded hardware prior')
b2 = ax.bar(x+w/2, v_vir, w, color=colors['Full\n(virtual)'], alpha=0.92, zorder=3, label='Classifier-derived virtual prior')
for i, (a, b) in enumerate(zip(v_rec, v_vir)):
    ax.text(i-w/2, a+0.008, f'{a:.3f}', ha='center', fontsize=8.2)
    ax.text(i+w/2, b+0.008, f'{b:.3f}', ha='center', fontsize=8.2)
ax.set_xticks(x); ax.set_xticklabels(labels2)
ax.set_ylim(0, 0.45)
ax.set_title('(c) Virtual prior vs. hardware prior', loc='left', fontsize=10.5, fontweight='bold')
ax.set_ylabel('MAE ($^{\circ}$C)')
ax.legend(loc='upper left', fontsize=8)
ax.grid(axis='y', alpha=0.28, ls='--')
mae_v = np.abs(pred_virtual-T).mean()
# (d) scatter
ax = axes[1,1]
ax.scatter(T, pred_virtual, s=26, facecolor='none', edgecolor=(0.13,0.55,0.13), linewidth=1.1, zorder=4,
           label=f'Cascaded model ($N={len(T)}$)')
lims = [33, 87]
ax.plot(lims, lims, 'k--', lw=1.6, zorder=3, label='1:1 reference')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel(r'Reference temperature $T_{\mathrm{ref}}$ ($^{\circ}$C)')
ax.set_ylabel(r'Predicted temperature ($^{\circ}$C)')
ax.set_title('(d) Regression on the real validation set', loc='left', fontsize=10.5, fontweight='bold')
ss_res = ((pred_virtual-T)**2).sum(); r2 = 1-ss_res/((T-T.mean())**2).sum()
ax.text(0.045, 0.945, f'MAE $= {mae_v:.3f}^{{\\circ}}$C\nRMSE $= {np.sqrt(((pred_virtual-T)**2).mean()):.3f}^{{\\circ}}$C\n$R^2 = {r2:.4f}$',
        transform=ax.transAxes, va='top', ha='left', fontsize=8.6,
        bbox=dict(facecolor='white', edgecolor='0.7', alpha=0.95, boxstyle='round,pad=0.38'))
ax.legend(loc='lower right', fontsize=8.2, framealpha=0.95)
ax.grid(alpha=0.28, ls='--')
ax.set_aspect('equal', adjustable='box')

plt.tight_layout(w_pad=2.4, h_pad=2.2)
for ext, kw in [('png', dict(dpi=600)), ('pdf', {})]:
    plt.savefig(f"/mnt/c/Users/tianyu.chu/Desktop/论文/NV论文/Figure4.{ext}", bbox_inches='tight', **kw)
    plt.savefig(f"Figure4.{ext}", bbox_inches='tight', **kw)
print("Figure4 saved. MAE virtual =", mae_v)

# ================= Figure 5: CDF =================
def errs(col):
    return np.abs(np.array([float(r[col]) for r in rows]) - T)
series = [('Single L.', errs('Single L.')), ('Double L.', errs('Double L.')),
          ('Blind AI', errs('Blind')), ('Laser only', errs('LaserOnly')),
          ('MW only', errs('MWOnly')), ('Full (recorded)', errs('Full_true_aux')),
          ('Full (virtual)', errs('VirtualCascade'))]
fig2, ax = plt.subplots(figsize=(6.2, 3.6))
plt.rcParams['font.size']=10
styles = {'Single L.': ('0.40', '-', 1.4), 'Double L.': ((0.70,0.21,0.17), '-.', 1.4),
          'Blind AI': ((0.95,0.55,0.10), '-', 1.4), 'Laser only': ((0.12,0.56,1.00), '-', 1.4),
          'MW only': ((0.58,0.44,0.86), '-', 1.4), 'Full (recorded)': ((0.20,0.59,0.55), '--', 1.6),
          'Full (virtual)': ((0.13,0.55,0.13), '-', 2.4)}
for name, e in series:
    e = np.sort(e); y = np.linspace(0, 100, len(e), endpoint=True)
    c, ls, lw = styles[name]
    ax.plot(e, y, color=c, ls=ls, lw=lw, label=name, zorder=4 if 'virtual' in name.lower() else 3)
ax.axhline(90, color='red', ls=':', alpha=0.6, zorder=2)
ax.text(0.05, 91.2, '90% pass threshold', color='red', fontsize=8)
ax.set_xlim(0, 2.0); ax.set_ylim(0, 103)
ax.set_xlabel('Error tolerance threshold ($^{\circ}$C)')
ax.set_ylabel('Cumulative percentage of samples (%)')
ax.set_title('Error CDF on real validation set', fontsize=10.5, fontweight='bold', loc='left')
ax.legend(loc='center left', bbox_to_anchor=(1.015, 0.5), fontsize=8.6, framealpha=0.95, edgecolor='0.75')
ax.grid(alpha=0.28, ls='--')
plt.tight_layout()
for ext, kw in [('png', dict(dpi=600)), ('pdf', {})]:
    plt.savefig(f"/mnt/c/Users/tianyu.chu/Desktop/论文/NV论文/Figure5.{ext}", bbox_inches='tight', **kw)
    plt.savefig(f"Figure5.{ext}", bbox_inches='tight', **kw)
print("Figure5 saved")
for name, e in series:
    e=np.array(e); print(f"  {name:16s} P(<0.6°C)={100*(e<0.6).mean():5.1f}%  P95={np.percentile(e,95):.2f}")
