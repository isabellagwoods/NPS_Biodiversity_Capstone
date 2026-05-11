"""
make_summary_figures.py — generate all figures for research_summary.md
Run with: python make_summary_figures.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

# ── colours ──────────────────────────────────────────────────────────────────
C_P1  = '#4C72B0'   # phase 1 — two-stage leaky
C_P2  = '#DD8452'   # phase 2 — dim-reduction park-split
C_P3  = '#55A868'   # phase 3 — data structure study
C_P4  = '#C44E52'   # phase 4 — single-stage rarefied

KEEP_C    = '#2ca02c'
DISCARD_C = '#d62728'

# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════
res = pd.read_csv('results.tsv', sep='\t')
res['exp'] = range(1, len(res) + 1)

# Phase labels (rows in results.tsv)
# Phase 1: rows 0-14  (Exp 001–015)
# Phase 2: rows 15-24 (Exp 016–024, dim-reduction park-split)
# Phase 4: rows 25-34 (Exp 032–041, single-stage rarefied)
phases = (
    [1] * 15 +   # Exp 001-015
    [2] * 9  +   # Exp 016-024
    [4] * 10     # Exp 032-041
)
res['phase'] = phases
res['phase_color'] = res['phase'].map({1: C_P1, 2: C_P2, 4: C_P4})

# Data structure study (phase 3) — separate tsv, no crash entries
struct_labels = ['A\nTaxon/park', 'B\nMonthly+geo', 'C\nFirst-diff',
                 'D\nYearly', 'E\nLag AR(1)', 'F\nTaxon×month']
struct_r2     = [0.7679, 0.3086, 0.0492, 0.6152, 0.5960, 0.8444]
struct_rmse   = [0.7138, 0.9470, 0.6109, 0.6292, 0.5927, 0.4427]

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — metric trajectory  (tall, wide)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(18, 10),
                         gridspec_kw={'height_ratios': [3, 1]})
ax = axes[0]
ax2 = axes[1]

phase_name = {1: 'Phase 1 — two-stage (leaky split)', 2: 'Phase 2 — dim-reduction (park split)',
              4: 'Phase 4 — single-stage rarefied'}

for ph, clr in [(1, C_P1), (2, C_P2), (4, C_P4)]:
    sub = res[res['phase'] == ph]
    ax.plot(sub['exp'], sub['s1_val_rmse'], '-', color=clr, linewidth=1.4, zorder=2)
    keep_m  = sub[sub['status'] == 'keep']
    drop_m  = sub[sub['status'] == 'discard']
    ax.scatter(keep_m['exp'],  keep_m['s1_val_rmse'],
               color=KEEP_C,    s=80, zorder=4, marker='o')
    ax.scatter(drop_m['exp'],  drop_m['s1_val_rmse'],
               color=DISCARD_C, s=60, zorder=4, marker='x')

# Phase boundary shading
ax.axvspan( 0.5, 15.5, alpha=0.06, color=C_P1)
ax.axvspan(15.5, 25.5, alpha=0.06, color=C_P2)
ax.axvspan(25.5, 35.5, alpha=0.06, color=C_P4)

# Phase 3 — show as horizontal reference band (best structure F val_rmse = 0.443)
ax.axhline(0.443, color=C_P3, linewidth=1.4, linestyle='--', zorder=3,
           label='Phase 3 best (Structure F, RMSE=0.443)')
ax.text(35.6, 0.443, 'Phase 3\nStruct-F', color=C_P3, fontsize=8, va='center')

# Key annotations
annotations = [
    (1,  0.697, 'Exp 001\nbaseline\n0.697'),
    (9,  0.477, 'Stage 2\nHistGBM\n0.477'),
    (16, 0.936, 'Park split\n(harder)\n0.936'),
    (26, 0.334, 'New model\n0.334'),
    (31, 0.326, 'Best:\nPCA+Ridge\n0.326'),
]
for xp, yp, txt in annotations:
    ax.annotate(txt, xy=(xp, yp), xytext=(xp + 0.6, yp + 0.06),
                fontsize=7.5, ha='left', color='#333333',
                arrowprops=dict(arrowstyle='->', color='#888888', lw=0.8))

ax.set_xlim(0.5, 36)
ax.set_ylim(0.0, 1.15)
ax.set_ylabel('val RMSE (s1_val_rmse)', fontsize=11)
ax.set_title('Metric Trajectory — All 35 Experiments\n'
             '(circles = keep, × = discard; shading = phase)', fontsize=13)
ax.grid(axis='y', alpha=0.3)

legend_handles = [
    mpatches.Patch(color=C_P1, alpha=0.7, label='Phase 1: two-stage, leaky split (Exp 1–15)'),
    mpatches.Patch(color=C_P2, alpha=0.7, label='Phase 2: dim-reduction, park split (Exp 16–24)'),
    mpatches.Patch(color=C_P3, alpha=0.7, label='Phase 3: data structure study (Exp 25–31)'),
    mpatches.Patch(color=C_P4, alpha=0.7, label='Phase 4: single-stage rarefied (Exp 32–41)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=KEEP_C,    markersize=9, label='keep'),
    plt.Line2D([0], [0], marker='x', color=DISCARD_C, markersize=9,        label='discard', markeredgewidth=2),
]
ax.legend(handles=legend_handles, loc='upper right', fontsize=8.5, framealpha=0.9)

# Bottom panel — R²
for ph, clr in [(1, C_P1), (2, C_P2), (4, C_P4)]:
    sub = res[res['phase'] == ph]
    ax2.plot(sub['exp'], sub['s1_val_r2'], '-', color=clr, linewidth=1.4)
    ax2.scatter(sub[sub['status']=='keep']['exp'],
                sub[sub['status']=='keep']['s1_val_r2'],
                color=KEEP_C, s=55, zorder=4, marker='o')
    ax2.scatter(sub[sub['status']=='discard']['exp'],
                sub[sub['status']=='discard']['s1_val_r2'],
                color=DISCARD_C, s=40, zorder=4, marker='x')

ax2.axhspan(0.5, 15.5, alpha=0.06, color=C_P1, transform=ax2.get_xaxis_transform())
ax2.axhline(0.0, color='#aaaaaa', linewidth=0.7, linestyle=':')
ax2.axhline(0.7, color='green', linewidth=0.9, linestyle='--', label='target R²=0.70')
ax2.set_xlim(0.5, 36)
ax2.set_ylabel('val R²', fontsize=10)
ax2.set_xlabel('Experiment #', fontsize=10)
ax2.legend(fontsize=8, loc='upper right')
ax2.grid(axis='y', alpha=0.3)
ax2.axvspan( 0.5, 15.5, alpha=0.06, color=C_P1)
ax2.axvspan(15.5, 25.5, alpha=0.06, color=C_P2)
ax2.axvspan(25.5, 35.5, alpha=0.06, color=C_P4)

plt.tight_layout()
plt.savefig(OUT / 'summary_metric_trajectory.png', dpi=160, bbox_inches='tight')
plt.close()
print("Saved summary_metric_trajectory.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — 3-panel: keep/discard | data structure R² | best vs baseline
# ═══════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(18, 6))
gs   = GridSpec(1, 3, figure=fig2, wspace=0.38)
axA  = fig2.add_subplot(gs[0])
axB  = fig2.add_subplot(gs[1])
axC  = fig2.add_subplot(gs[2])

# ── Panel A: keep / discard per phase ────────────────────────────────────
phase_labels = ['Phase 1\n(Exp 1–15)\ntwo-stage leaky',
                'Phase 2\n(Exp 16–24)\ndim-reduction',
                'Phase 3\n(Exp 25–31)\ndata structures',
                'Phase 4\n(Exp 32–41)\nsingle-stage']
keeps    = [res[res['phase']==ph]['status'].eq('keep').sum()    for ph in [1,2,4]]
discards = [res[res['phase']==ph]['status'].eq('discard').sum() for ph in [1,2,4]]
keeps.insert(2, 6)     # Phase 3 — all 6 structures kept
discards.insert(2, 0)  # Phase 3 — no discards
crashes = [0, 0, 0, 0]  # no crashes

x = np.arange(4)
w = 0.28
bars_k = axA.bar(x - w, keeps,    w, label='keep',    color=KEEP_C,    alpha=0.85)
bars_d = axA.bar(x,     discards, w, label='discard', color=DISCARD_C, alpha=0.85)
bars_c = axA.bar(x + w, crashes,  w, label='crash',   color='#888888', alpha=0.85)

for bar in bars_k:
    if bar.get_height() > 0:
        axA.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)
for bar in bars_d:
    if bar.get_height() > 0:
        axA.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)

axA.set_xticks(x)
axA.set_xticklabels(phase_labels, fontsize=8.5)
axA.set_ylabel('Experiments', fontsize=10)
axA.set_title('Keep / Discard / Crash\nby Phase', fontsize=11)
axA.legend(fontsize=9)
axA.set_ylim(0, 13)
axA.grid(axis='y', alpha=0.3)
axA.text(0.02, 0.97, '0 crashes total', transform=axA.transAxes,
         fontsize=8, va='top', color='#555555', style='italic')

# ── Panel B: data structure study R² ────────────────────────────────────
colors_struct = [C_P3] * 6
colors_struct[5] = '#1f7a1f'  # structure F — best, darker
bars_s = axB.bar(struct_labels, struct_r2, color=colors_struct, alpha=0.85, edgecolor='white')
for bar, r2 in zip(bars_s, struct_r2):
    axB.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{r2:.2f}', ha='center', va='bottom', fontsize=9.5)
axB.axhline(0.70, color='#e07000', linewidth=1.2, linestyle='--', label='target R²=0.70')
axB.set_ylim(0, 1.0)
axB.set_ylabel('val R²', fontsize=10)
axB.set_title('Data Structure Study (Phase 3)\nval R² by Data Framing', fontsize=11)
axB.legend(fontsize=9)
axB.grid(axis='y', alpha=0.3)

# ── Panel C: best vs. very first run ─────────────────────────────────────
metrics = ['val RMSE', 'val R²', 'CV RMSE']
first   = [0.697, -0.043, 0.761]   # Exp 001
best    = [0.326,  0.629, 0.309]   # Exp 037

x3 = np.arange(len(metrics))
w3 = 0.33
b_first = axC.bar(x3 - w3/2, first, w3, label='Exp 001 — first run\nRidge(α=1), two-stage, leaky',
                  color='#888888', alpha=0.85)
b_best  = axC.bar(x3 + w3/2, best,  w3, label='Exp 037 — best result\nPCA(n=10)+Ridge(α=100)',
                  color='#2a7a2a', alpha=0.85)

for bar, v in zip(b_first, first):
    ypos = bar.get_height()
    offset = 0.03 if ypos >= 0 else -0.07
    axC.text(bar.get_x() + bar.get_width()/2, ypos + offset,
             f'{v:.3f}', ha='center', fontsize=9, color='#333333')
for bar, v in zip(b_best, best):
    ypos = bar.get_height()
    offset = 0.03 if ypos >= 0 else -0.07
    axC.text(bar.get_x() + bar.get_width()/2, ypos + offset,
             f'{v:.3f}', ha='center', fontsize=9, color='#333333')

axC.axhline(0, color='#aaaaaa', linewidth=0.7)
axC.set_xticks(x3)
axC.set_xticklabels(metrics, fontsize=10)
axC.set_title('Best Result vs. Very First Run', fontsize=11)
axC.set_ylabel('Metric value', fontsize=10)
axC.legend(fontsize=8.5, loc='upper right')
axC.grid(axis='y', alpha=0.3)

# delta annotations
deltas = [f'Δ {best[i]-first[i]:+.3f}' for i in range(len(metrics))]
delta_colors = ['green' if (best[i] < first[i] and i != 1) or (best[i] > first[i] and i == 1)
                else 'red' for i in range(len(metrics))]
for i, (d, dc) in enumerate(zip(deltas, delta_colors)):
    axC.text(x3[i], max(first[i], best[i]) + 0.12, d, ha='center',
             fontsize=9, color=dc, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT / 'summary_panels.png', dpi=160, bbox_inches='tight')
plt.close()
print("Saved summary_panels.png")

print("All figures done.")
