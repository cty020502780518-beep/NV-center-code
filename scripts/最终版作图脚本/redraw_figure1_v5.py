import os, json, csv, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

groups = json.load(open('analysis/fulldata_groups.json'))
CALI_CSV = "analysis/全功率组_温度系数标定.csv"  # canonical source, keep inside workspace
cali = {}
with open(CALI_CSV, encoding='utf-8-sig') as fh:
    for row in csv.DictReader(fh):
        cali[(float(row['Laser (%)']), float(row['MW (dBm)']))] = (float(row['Slope']), float(row['Intercept']), float(row['R2']))

gkey = ('50%', '+0')
# panel (a) data: 50% laser, 0 dBm seeds (n=21)
T = np.array(groups['50%|+0']['t']); D = np.array(groups['50%|+0']['D'])
k_a = float(groups['50%|+0']['k']); b_a = float(groups['50%|+0']['b']); r2_a = float(groups['50%|+0']['r2'])
n_a = len(T)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 10, 'axes.linewidth': 0.8, 'mathtext.fontset': 'stix',
})
fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.7))

# ---------------- (a) ----------------
ax = axes[0]
ax.scatter(T, D, s=26, facecolor='none', edgecolor=(0.16,0.45,0.70), linewidth=1.15, zorder=5, label='Measured spectra')
xl = np.linspace(36, 86, 60)
ax.plot(xl, k_a*xl+b_a, color=(0.84,0.16,0.16), lw=2.0, zorder=6, label='Linear fit')
ax.set_xlabel(r'Reference temperature $T_{\mathrm{ref}}$ ($^{\circ}$C)')
ax.set_ylabel('Doublet center $D_0=(f_1{+}f_2)/2$ (MHz)')
ax.set_title('(a) Calibration within one power group', loc='left', fontsize=10.5, fontweight='bold')
ax.set_xlim(36, 86)
ax.set_ylim(2864.2, 2870.8)
# info box placed in the empty upper-right quadrant (data line falls lower-right)
info = (f'Slope $k_i = {k_a*1000:.1f}$ kHz/K\n'
        f'$R^2 = {r2_a:.4f}$,  $n = {n_a}$ states\n'
        'Laser 50% / MW 0 dBm')
ax.text(0.965, 0.955, info, transform=ax.transAxes, ha='right', va='top', fontsize=8.6,
        bbox=dict(facecolor='white', edgecolor='0.72', alpha=0.95, boxstyle='round,pad=0.38'))
ax.legend(loc='upper left', fontsize=8.2, framealpha=0.95, edgecolor='0.75')
ax.grid(alpha=0.28, ls='--')

# ---------------- (b) ----------------
ax = axes[1]
order = [(l, m) for l in (20.0, 50.0, 100.0) for m in (-5.0, 0.0, 5.0)]
slopes = np.array([abs(cali[g][0])*1000 for g in order])
labels = [f'{int(l)}%\n{int(m):+d} dBm' for l, m in order]
ax.bar(range(9), slopes, color=(0.29,0.44,0.57), width=0.62, alpha=0.92, zorder=3)
ax.axhline(74.0, color=(0.84,0.16,0.16), ls='--', lw=1.7, zorder=4)
ax.axhline(slopes.mean(), color='0.15', ls=':', lw=1.5, zorder=4)
# labels placed in the free band far above all bar tops (bars < 97)
ax.text(4.0, 104.8, 'ideal: $-74$ kHz/K (red dashed)', fontsize=8.4,
        color=(0.84,0.16,0.16), ha='center', va='center')
ax.text(4.0, 110.6, f'measured mean: $-{slopes.mean():.1f}$ kHz/K (dotted)', fontsize=8.4,
        color='0.15', ha='center', va='center')
for i, v in enumerate(slopes):
    ax.text(i, v+1.0, f'{v:.0f}', ha='center', fontsize=7.2)
ax.set_xticks(range(9)); ax.set_xticklabels(labels, fontsize=7.0, rotation=28, ha='right')
ax.set_xlabel('Power state (laser power % | MW power dBm)')
ax.set_ylabel('Apparent slope $|k_i|$ (kHz/K)')
ax.set_ylim(0, 116)
ax.set_title('(b) Apparent slopes across 9 power states', loc='left', fontsize=10.5, fontweight='bold')
ax.grid(axis='y', alpha=0.28, ls='--')

# ---------------- (c) ----------------
ax = axes[2]
rng = np.random.default_rng(42)
t_aug = rng.uniform(35, 85, 2000)
# visible envelope = per-measurement repeatability spread of the seeds (measured sigma ~0.05-0.09 MHz over 104 seeds)
jitter = rng.normal(0, 0.121, 2000)  # matches measured intra-group residual rms of 50%/0dBm
d_aug = k_a*t_aug + b_a + jitter
ax.scatter(t_aug, d_aug, s=8, color=(0.42,0.68,0.90), alpha=0.5, zorder=2, label='Generated virtual cloud')
ax.scatter(T, D, s=28, color=(0.84,0.16,0.16), edgecolor='k', linewidth=0.45, zorder=5, label='Real seed data (n=21)')
ax.plot(xl, k_a*xl+b_a, 'k--', lw=1.6, zorder=4, label=f'Physical law  ($k_i = {k_a*1000:.1f}$ kHz/K)')
ax.set_xlabel(r'Temperature $T$ ($^{\circ}$C)')
ax.set_ylabel('Doublet center (MHz)')
ax.set_title('(c) Physics-constrained Sim2Real', loc='left', fontsize=10.5, fontweight='bold')
ax.set_xlim(33, 87)
ax.set_ylim(2862.2, 2871.0)
# power-group tag placed in the empty upper-right quadrant (same as (a)/legend zone is fine below legend)
ax.text(0.965, 0.71, 'Power group:\nLaser 50% / MW 0 dBm', transform=ax.transAxes,
        ha='right', va='top', fontsize=8.6, fontweight='bold',
        bbox=dict(facecolor='white', edgecolor='0.72', alpha=0.96, boxstyle='round,pad=0.38'))
ax.legend(loc='upper right', fontsize=8.0, framealpha=0.95, edgecolor='0.75')
ax.grid(alpha=0.28, ls='--')

plt.tight_layout(w_pad=2.2)
for ext, kw in [('png', dict(dpi=600)), ('pdf', {})]:
    out = f"/mnt/c/Users/tianyu.chu/Desktop/论文/NV论文/Figure1.{ext}"
    plt.savefig(out, bbox_inches='tight', **kw)
    plt.savefig(f"Figure1.{ext}", bbox_inches='tight', **kw)
print("Figure1 v5 saved (cloud sigma=0.121 MHz, canonical CSV)")
