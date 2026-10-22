import json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9.5,
                     'axes.linewidth':0.8, 'mathtext.fontset':'stix'})

d = json.load(open('eval_models/mc_sensitivity.json'))
sigma = np.array(d['sigma']); R = d['results']

style = {
 'Single Lor.':  ('0.45','x',':', 1.4, 'Single Lor. (high bias, stable)'),
 'Double Lor.':  ((0.70,0.21,0.17),'o','--', 1.6, 'Double Lor. (low noise OK, diverges)'),
 'Blind':        ((0.95,0.55,0.10),None,'-', 1.6, 'Blind AI'),
 'MW Only':      ((0.58,0.44,0.86),None,'-', 1.6, 'MW Only'),
 'Laser Only':   ((0.12,0.56,1.00),None,'-', 1.6, 'Laser Only'),
 'Full Model':   ((0.13,0.55,0.13),None,'-', 2.6, 'Full Model (proposed)'),
}

fig, ax = plt.subplots(figsize=(7.4, 4.4))
for k in ['Single Lor.','Double Lor.','Blind','MW Only','Laser Only','Full Model']:
    c, mk, ls, lw, lab = style[k]
    ax.plot(sigma, R[k], color=c, ls=ls, lw=lw, label=lab, marker=mk, ms=4.5, markevery=3,
            alpha=0.95 if k=='Full Model' else 0.85,
            zorder=5 if k=='Full Model' else 3)

# measured per-point error bar as a marker on the noise axis
y_lo = min(min(R[k]) for k in R)
ax.axvline(0.0017, color='k', ls=':', lw=1.1, alpha=0.6)
ax.annotate('measured per-point\nerror bar $\\approx1.7\\times10^{-3}$',
            xy=(0.0017, y_lo), xytext=(0.0035, y_lo+0.02), fontsize=8,
            arrowprops=dict(arrowstyle='->', lw=0.9, color='0.3'), color='0.25')

ax.set_xlabel(r'Injected Gaussian noise level $\sigma_{\mathrm{noise}}$ (contrast units)')
ax.set_ylabel(r'Mean prediction uncertainty $\sigma_{T}$ ($^{\circ}$C)')
ax.set_title('Monte Carlo noise stress test (110 spectra $\\times$ 15 realizations per level)',
             fontsize=10.5, fontweight='bold', loc='left')
ax.set_xlim(0, 0.02)
allmax = max(max(R[k]) for k in R)
ax.set_ylim(0, np.ceil(allmax*1.12*2)/2)
ax.grid(alpha=0.28, ls='--')
ax.legend(loc='upper left', fontsize=8.1, framealpha=0.95, edgecolor='0.75')

# zoom inset: 0 -- 0.006 (near operating floor), full y auto
axins = ax.inset_axes([0.47, 0.14, 0.50, 0.44])
for k in ['Single Lor.','Double Lor.','Blind','MW Only','Laser Only','Full Model']:
    c, mk, ls, lw, lab = style[k]
    axins.plot(sigma, R[k], color=c, ls=ls, lw=lw*0.8, alpha=0.9)
axins.set_xlim(0, 0.006)
sub = {k: [v for s,v in zip(sigma,R[k]) if s<=0.006] for k in style}
ymax = max(max(v) for v in sub.values())
axins.set_ylim(0, np.ceil(ymax*1.15*20)/20)
axins.grid(alpha=0.28, ls='--')
axins.set_title('zoom: near measured noise floor', fontsize=8)
axins.tick_params(labelsize=7.5)
ax.indicate_inset_zoom(axins, edgecolor='0.55', lw=1.0)

plt.tight_layout()
for ext, kw in [('png', dict(dpi=600)), ('pdf', {})]:
    plt.savefig(f"/mnt/c/Users/tianyu.chu/Desktop/论文/NV论文/Figure6.{ext}", bbox_inches='tight', **kw)
    plt.savefig(f"Figure6.{ext}", bbox_inches='tight', **kw)
print("Figure6 saved")
for k in ['Single Lor.','Double Lor.','Blind','MW Only','Laser Only','Full Model']:
    print(f"  {k:12s} sigma_temp@{sigma[-1]:.3f} = {R[k][-1]:.3f} °C  | @{sigma[3]:.4f} = {R[k][3]:.3f} °C")
