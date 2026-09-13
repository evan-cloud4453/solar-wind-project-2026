# -*- coding: utf-8 -*-
"""Generate the portfolio figures into <repo>/figures."""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

DST = r'C:\2026-1 My Project\solar-wind-project-2026'
FIG = os.path.join(DST, 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'font.size': 10,
})

OURS = '#d1495b'
BASE = '#4c6ef5'
GREY = '#9aa0a6'
GOOD = '#2a9d8f'


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


# --------------------------------------------------------------------------
# Final-round public leaderboard, 33 teams
# --------------------------------------------------------------------------
FINAL_SCORES = [
    55.0411, 55.5576, 55.5931, 55.9010, 56.0239, 56.0308, 56.1123, 56.2854,
    56.4122, 56.4965, 56.5416, 56.6192, 56.6293, 56.8108, 57.0893, 57.1820,
    57.2308, 57.2975, 57.3357, 57.4904, 57.6581, 57.6724, 57.7067, 58.0603,
    58.0914, 58.2225, 58.2227, 58.7370, 58.7501, 58.7759, 58.8028, 59.9971,
    60.0171,
]
OUR_RANK = 31                       # 1-indexed
OUR_SCORE = 58.8028


def fig_final_leaderboard():
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ranks = np.arange(1, len(FINAL_SCORES) + 1)
    colors = [OURS if r == OUR_RANK else GREY for r in ranks]
    ax.bar(ranks, FINAL_SCORES, color=colors, width=0.72)
    ax.set_ylim(54.5, 60.6)
    ax.set_xlim(0.3, len(FINAL_SCORES) + 0.7)
    ax.set_xlabel('final public leaderboard rank')
    ax.set_ylabel('RMSE  [km/s]   (lower is better)')
    ax.set_title('Final round — public leaderboard, 33 scored teams')
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 33])

    ax.annotate('team Sion  #%d   %.4f' % (OUR_RANK, OUR_SCORE),
                xy=(OUR_RANK, OUR_SCORE), xytext=(OUR_RANK - 8.5, 60.0),
                color=OURS, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=OURS, lw=1.4))
    ax.annotate('#1  %.4f' % FINAL_SCORES[0], xy=(1, FINAL_SCORES[0]),
                xytext=(2.2, 59.4), color='#333',
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.1))

    span = FINAL_SCORES[-1] - FINAL_SCORES[0]
    ax.text(0.99, 0.06,
            'whole field spans %.2f km/s  ·  one rank = %.3f km/s' %
            (span, span / (len(FINAL_SCORES) - 1)),
            transform=ax.transAxes, ha='right', fontsize=9.5, color='#444',
            bbox=dict(boxstyle='round,pad=0.4', fc='#f4f4f6', ec='#ddd'))
    save(fig, '01-final-leaderboard.png')


# --------------------------------------------------------------------------
# Submission history: local validation vs public score
# --------------------------------------------------------------------------
SUBS = [
    # label,          val,    public,  note
    ('P1',           68.408, 62.3393, 'baseline + fundamentals'),
    ('P3',           64.203, 58.8028, 'CH grid + ballistic  (best)'),
    ('P6-T1',        63.357, 60.1955, '7-model NNLS ensemble'),
    ('P9',           66.978, 59.2246, 'paired CV instrument'),
    ('N1',           None,   59.7313, 'baseline + SpeedNet'),
    ('P10',          66.772, 61.7853, 'SpeedNet + Stonyhurst'),
    ('P11-S1',       65.622, 59.7332, 'no-fit ensemble'),
]


def fig_submissions():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                   gridspec_kw={'width_ratios': [1.15, 1]})

    labels = [s[0] for s in SUBS]
    pubs = [s[2] for s in SUBS]
    x = np.arange(len(SUBS))
    cols = [OURS if l == 'P3' else BASE for l in labels]
    ax1.bar(x, pubs, color=cols, width=0.62)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylim(57.5, 63)
    ax1.set_ylabel('public RMSE  [km/s]')
    ax1.set_title('7 submissions — nothing ever beat P3')
    ax1.axhline(58.8028, color=OURS, ls='--', lw=1.2, alpha=0.8)
    for xi, p in zip(x, pubs):
        ax1.text(xi, p + 0.07, '%.2f' % p, ha='center', fontsize=9)
    ax1.text(len(SUBS) - 0.55, 58.30, 'best submission  P3 = 58.80', color=OURS,
             fontsize=9, ha='right', va='bottom')

    # slope graph: validation ranking vs public ranking
    scored = [s for s in SUBS if s[1] is not None]
    by_val = sorted(scored, key=lambda s: s[1])
    by_pub = sorted(scored, key=lambda s: s[2])
    for s in scored:
        i = by_val.index(s); j = by_pub.index(s)
        c = OURS if s[0] == 'P3' else (GOOD if s[0] == 'P9' else GREY)
        lw = 2.4 if s[0] in ('P3', 'P9', 'P6-T1') else 1.2
        ax2.plot([0, 1], [i, j], color=c, lw=lw, marker='o', ms=6,
                 zorder=3 if lw > 2 else 2)
        ax2.text(-0.05, i, '%s  %.2f' % (s[0], s[1]), ha='right', va='center',
                 fontsize=9.5, color=c, fontweight='bold' if lw > 2 else 'normal')
        ax2.text(1.05, j, '%.2f  %s' % (s[2], s[0]), ha='left', va='center',
                 fontsize=9.5, color=c, fontweight='bold' if lw > 2 else 'normal')
    ax2.set_xlim(-0.75, 1.75); ax2.set_ylim(len(scored) - 0.4, -0.6)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['local validation\n(best  →  worst)',
                         'public leaderboard\n(best  →  worst)'])
    ax2.set_yticks([]); ax2.grid(False)
    for sp in ('left', 'bottom'):
        ax2.spines[sp].set_visible(False)
    ax2.set_title('The ranking inverts')
    save(fig, '02-submissions-and-inversion.png')


# --------------------------------------------------------------------------
# Leaderboard drift: our score froze, the field moved
# --------------------------------------------------------------------------
def fig_drift():
    days = [0, 1, 2]
    day_lbl = ['Aug 19', 'Aug 20', 'final (Aug 21)']
    leader = [57.6114, 55.5866, 55.0411]
    ours = [58.8028, 58.8028, 58.8028]
    our_rank = [2, 22, 31]
    field = [None, 31, 33]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax1.plot(days, leader, 'o-', color=BASE, lw=2, label='leader')
    ax1.plot(days, ours, 'o-', color=OURS, lw=2.4, label='team Sion (P3)')
    ax1.fill_between(days, leader, ours, color=BASE, alpha=0.08)
    for d, v in zip(days, leader):
        ax1.text(d, v - 0.22, '%.2f' % v, ha='center', color=BASE, fontsize=9)
    ax1.text(1, 58.98, 'frozen at 58.8028 — we submitted nothing better',
             ha='center', color=OURS, fontsize=9.5)
    ax1.set_xticks(days); ax1.set_xticklabels(day_lbl)
    ax1.set_ylim(54.6, 59.6)
    ax1.set_ylabel('RMSE  [km/s]')
    ax1.set_title('Our score stood still; the field improved')
    ax1.legend(frameon=False, loc='center left')

    ax2.plot(days, our_rank, 'o-', color=OURS, lw=2.4, ms=9)
    for d, r, n in zip(days, our_rank, field):
        tail = ' / %d' % n if n else ''
        ax2.text(d, r - 2.2, '#%d%s' % (r, tail), ha='center',
                 color=OURS, fontweight='bold')
    ax2.invert_yaxis()
    ax2.set_xticks(days); ax2.set_xticklabels(day_lbl)
    ax2.set_ylim(36, -3)
    ax2.set_ylabel('public rank')
    ax2.set_title('Rank: 2nd  →  22nd  →  31st')
    save(fig, '03-leaderboard-drift.png')


# --------------------------------------------------------------------------
# Measurement resolution vs leaderboard density
# --------------------------------------------------------------------------
def fig_resolution():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ranks = np.arange(1, len(FINAL_SCORES) + 1)
    lo, hi = OUR_SCORE - 3.0, OUR_SCORE + 3.0
    inside = [(lo <= s <= hi) for s in FINAL_SCORES]
    n_in = sum(inside)

    ax.axhspan(lo, hi, color=BASE, alpha=0.10, zorder=0)
    ax.scatter(ranks, FINAL_SCORES,
               c=[BASE if i else GREY for i in inside], s=42, zorder=3)
    ax.scatter([OUR_RANK], [OUR_SCORE], c=OURS, s=140, zorder=4,
               edgecolor='white', linewidth=1.5)
    ax.axhline(OUR_SCORE, color=OURS, ls='--', lw=1.2)

    ax.set_xlabel('final public leaderboard rank')
    ax.set_ylabel('RMSE  [km/s]')
    ax.set_title('Why local validation could not steer the ranking')
    ax.set_ylim(54.3, 62.3)
    ax.annotate('', xy=(2.0, lo), xytext=(2.0, hi),
                arrowprops=dict(arrowstyle='<->', color=BASE, lw=1.6))
    ax.text(2.9, OUR_SCORE + 1.5,
            'our measured noise floor  ±3 km/s\n'
            'covers %d of %d teams' % (n_in, len(FINAL_SCORES)),
            color=BASE, fontsize=10, fontweight='bold', va='center')
    step = (OUR_SCORE - FINAL_SCORES[0]) / (OUR_RANK - 1)
    ax.text(0.99, 0.05,
            'ranks 1-%d are %.2f km/s apart in total   ·   one rank = %.3f km/s'
            % (OUR_RANK, OUR_SCORE - FINAL_SCORES[0], step),
            transform=ax.transAxes, ha='right', fontsize=9.5, color='#444',
            bbox=dict(boxstyle='round,pad=0.4', fc='#f4f4f6', ec='#ddd'))
    save(fig, '04-resolution-gap.png')


# --------------------------------------------------------------------------
# Per-horizon error
# --------------------------------------------------------------------------
def fig_horizon():
    h = np.arange(6, 73, 6)
    model = np.array([27.847, 44.479, 54.495, 60.998, 66.154, 70.028,
                      72.546, 74.231, 75.275, 76.414, 77.973, 79.510])
    persist = np.array([30.134, 49.613, 63.040, 73.255, 82.029, 89.150,
                        94.699, 99.062, 102.730, 105.954, 108.661, 110.933])
    clim = np.full_like(h, 93.5, dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.5))

    ax1.plot(h, persist, 's--', color='#e07a5f', label='persistence')
    ax1.plot(h, clim, ':', color=GREY, lw=2, label='climatology')
    ax1.plot(h, model, 'o-', color=BASE, lw=2.2, label='our model')
    ax1.axvline(42, color='#999', lw=1, ls='-.')
    ax1.text(43, 34, 'past 42 h persistence\nis worse than climatology',
             fontsize=9, color='#555')
    ax1.set_xlabel('forecast horizon  [hour]')
    ax1.set_ylabel('RMSE  [km/s]')
    ax1.set_title('Error grows with lead time')
    ax1.legend(frameon=False, loc='lower right')

    gain = (persist - model) / persist * 100
    ax2.bar(h, gain, width=4.2, color=GOOD)
    ax2.set_xlabel('forecast horizon  [hour]')
    ax2.set_ylabel('improvement over persistence  [%]')
    ax2.set_title('Where the model actually earns its keep')
    for xi, g in zip(h, gain):
        ax2.text(xi, g + 0.6, '%.0f' % g, ha='center', fontsize=8.5)
    ax2.set_ylim(0, 34)
    save(fig, '05-horizon-error.png')


# --------------------------------------------------------------------------
# Literature comparison
# --------------------------------------------------------------------------
def fig_literature():
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    h = [6, 24, 72]
    son = [37.4, 57.6, 68.2]
    ours = [28.3, 60.9, 74.4]
    x = np.arange(3); w = 0.36
    ax.bar(x - w / 2, son, w, color=GREY, label='Son et al. 2023')
    ax.bar(x + w / 2, ours, w, color=BASE, label='ours (P6, validation)')
    for xi, a, b in zip(x, son, ours):
        d = b - a
        ax.text(xi + w / 2, b + 1.2, '%+.1f' % d, ha='center', fontsize=9.5,
                color=(GOOD if d < 0 else OURS), fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(['6 h', '24 h', '72 h'])
    ax.set_ylabel('RMSE  [km/s]')
    ax.set_ylim(0, 88)
    ax.set_title('We win the short horizon and lose the long one')
    ax.legend(frameon=False, loc='upper left')
    save(fig, '06-literature-comparison.png')


# --------------------------------------------------------------------------
# P3 pipeline diagram
# --------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc, ec, fs=9.5, tc='#111', weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.012,rounding_size=0.02',
                                fc=fc, ec=ec, lw=1.3, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc, zorder=3, fontweight=weight, linespacing=1.45)


def arrow(ax, p0, p1, color='#666'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=13,
                                 color=color, lw=1.3, zorder=1,
                                 shrinkA=2, shrinkB=2))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off'); ax.grid(False)

    IMG, PHY, NET, OUT = '#eaf1fe', '#fdf0e6', '#e9f6f2', '#fdeaee'
    IMGE, PHYE, NETE, OUTE = '#7fa5f0', '#e8a06a', '#66bfae', '#e08497'

    ax.text(0.5, 0.965, 'P3 — best submission (public 58.8028)',
            ha='center', fontsize=13.5, fontweight='bold')
    ax.text(0.5, 0.928,
            'no CNN image branch · low-dimensional physical features only',
            ha='center', fontsize=10, color='#666')

    box(ax, 0.04, 0.80, 0.34, 0.085,
        '193 Å  +  211 Å\n20 frames × 6 h  (past 120 h)', IMG, IMGE, 10)
    box(ax, 0.62, 0.80, 0.32, 0.085,
        'solar wind speed\n20 observations  (past 120 h)', OUT, OUTE, 10)

    box(ax, 0.04, 0.655, 0.34, 0.085,
        'disk detection on the train mean image\nradius shrunk 5 % to avoid the limb',
        IMG, IMGE, 9.2)
    box(ax, 0.04, 0.505, 0.34, 0.10,
        'coronal-hole mask\ndarker than 0.45 × disk median\nin BOTH 193 Å and 211 Å',
        PHY, PHYE, 9.2, weight='bold')
    box(ax, 0.04, 0.375, 0.34, 0.078,
        '3 × 5 grid  →  15 area fractions', PHY, PHYE, 9.5)
    box(ax, 0.04, 0.24, 0.34, 0.075, 'CH  GRU', NET, NETE, 10.5, weight='bold')

    box(ax, 0.42, 0.345, 0.28, 0.205,
        'ballistic alignment\n\n'
        r'source time  $=T_0 + h - \tau$' + '\n'
        r'$\tau = 1\,\mathrm{AU}/v$,   $v \in \{350,500,700\}$'
        '\n\nthe 5-day window covers\nevery horizon\'s source time',
        PHY, PHYE, 9.2, weight='bold')

    box(ax, 0.62, 0.655, 0.32, 0.085,
        'diff + 9 summary statistics', OUT, OUTE, 9.5)
    box(ax, 0.62, 0.24, 0.32, 0.075, 'wind  GRU', NET, NETE, 10.5, weight='bold')

    box(ax, 0.19, 0.115, 0.62, 0.078,
        'weight-sharing head  +  horizon embedding   '
        '(12× fewer parameters than per-horizon heads)', NET, NETE, 9.8)
    box(ax, 0.19, 0.005, 0.62, 0.072,
        r'residual output:  $\hat y_h = \mathrm{wind}_{19} + \Delta_h$,'
        '   h = 6, 12, …, 72 h   (12 values)', OUT, OUTE, 10, weight='bold')

    arrow(ax, (0.21, 0.80), (0.21, 0.742))
    arrow(ax, (0.21, 0.655), (0.21, 0.607))
    arrow(ax, (0.21, 0.505), (0.21, 0.455))
    arrow(ax, (0.21, 0.375), (0.21, 0.317))
    arrow(ax, (0.38, 0.435), (0.42, 0.448))
    arrow(ax, (0.78, 0.80), (0.78, 0.742))
    arrow(ax, (0.78, 0.655), (0.78, 0.317))
    arrow(ax, (0.21, 0.24), (0.35, 0.194))
    arrow(ax, (0.56, 0.345), (0.56, 0.194))
    arrow(ax, (0.78, 0.24), (0.66, 0.194))
    arrow(ax, (0.5, 0.115), (0.5, 0.078))

    ax.text(0.015, 0.155,
            'loss is the official metric itself:\n'
            r'$\mathrm{mean}_h\,\mathrm{RMSE}_h$' + '   (not pooled RMSE)',
            ha='left', va='center', fontsize=9.3, color='#555',
            bbox=dict(boxstyle='round,pad=0.4', fc='#f7f7f9', ec='#ddd'))
    save(fig, '07-p3-pipeline.png')


# --------------------------------------------------------------------------
# What survived / what died
# --------------------------------------------------------------------------
def fig_ideas():
    kept = [
        ('residual target  (y − wind_19)', 'P1'),
        ('loss aligned to the official metric', 'P1'),
        ('global normalisation, never per-image', 'P1'),
        ('no left-right flip augmentation', 'P2'),
        ('longitude-preserving pooling', 'P2'),
        ('CH grid area features', 'P3'),
        ('193 AND 211 dual-channel mask', 'P3'),
        ('ballistic alignment', 'P3'),
        ('weight-sharing horizon head', 'P3'),
        ('CNN image branch turned OFF', 'P3'),
        ('extract CH from the 512 px original', 'P6'),
        ('chain-grouped CV', 'P7'),
        ('paired-sample decision rule', 'P9'),
    ]
    dropped = [
        ('limb-brightening correction', 'P4', 'val +3.8; the rim was read as a hole'),
        ('12th-percentile threshold', 'P4', 'regressed together with the above'),
        ('NNLS ensemble weights', 'P6', 'public −1.39; 84 dof fitted on val'),
        ('CNN branch revived', 'P6', 'best on val, refuted on public'),
        ('Hampel outlier filter', 'P7', '1.47 % contamination — no effect'),
        ('adaptive transit time τ', 'P9', 'lost the paired comparison'),
        ('SpeedNet CNN-LSTM ×3', 'P10', 'all three lost to the feature control'),
        ('rotation longitude extrapolation', 'P10', 'OFF beat ON'),
        ('missing-value imputation', 'audit', 'there are no missing values'),
        ('186 h ballistic lag', 'audit', 'edge-of-range artefact, found twice'),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2),
                                   gridspec_kw={'width_ratios': [1, 1.32]})
    for ax in (ax1, ax2):
        ax.axis('off'); ax.grid(False); ax.set_xlim(0, 1)

    ax1.set_ylim(len(dropped) + 0.5, -1.2)
    ax1.text(0, -0.8, 'KEPT — carried through to the end',
             fontsize=12, fontweight='bold', color=GOOD)
    for i, (name, ver) in enumerate(kept):
        ax1.text(0.02, i * 0.78, '✓', color=GOOD, fontsize=11, fontweight='bold')
        ax1.text(0.09, i * 0.78, name, fontsize=10, va='center')
        ax1.text(0.98, i * 0.78, ver, fontsize=9, color=GREY, ha='right', va='center')

    ax2.set_ylim(len(dropped) + 0.5, -1.2)
    ax2.text(0, -0.8, 'DROPPED — tried, measured, abandoned',
             fontsize=12, fontweight='bold', color=OURS)
    for i, (name, ver, why) in enumerate(dropped):
        ax2.text(0.02, i - 0.13, '✗', color=OURS, fontsize=11, fontweight='bold')
        ax2.text(0.07, i - 0.13, name, fontsize=10, va='center')
        ax2.text(0.07, i + 0.26, why, fontsize=8.6, color='#777', va='center')
        ax2.text(0.99, i - 0.13, ver, fontsize=9, color=GREY, ha='right', va='center')
    save(fig, '08-ideas-kept-and-dropped.png')


if __name__ == '__main__':
    fig_final_leaderboard()
    fig_submissions()
    fig_drift()
    fig_resolution()
    fig_horizon()
    fig_literature()
    fig_pipeline()
    fig_ideas()
