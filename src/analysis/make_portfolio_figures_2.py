# -*- coding: utf-8 -*-
"""Second figure pass: lift the EDA plots out of the executed notebooks and
build the coronal-hole mask comparison from the P5 run screenshots."""
import os, sys, io, json, base64
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DST = r'C:\2026-1 My Project\solar-wind-project-2026'
FIG = os.path.join(DST, 'figures')
os.makedirs(FIG, exist_ok=True)


def nb_png(nb_rel, cell_index, out_name):
    """Pull one PNG output out of an executed notebook."""
    path = os.path.join(DST, nb_rel)
    nb = json.load(open(path, encoding='utf-8'))
    cell = nb['cells'][cell_index]
    for o in cell.get('outputs', []):
        data = o.get('data', {})
        if 'image/png' in data:
            dest = os.path.join(FIG, out_name)
            open(dest, 'wb').write(base64.b64decode(data['image/png']))
            print('wrote', out_name)
            return
    print('NO IMAGE in', nb_rel, 'cell', cell_index)


EDA = 'experiments/p07-cv-instrument/eda_p7_executed.ipynb'
nb_png(EDA, 6,  '09-eda-horizon-baselines.png')
nb_png(EDA, 8,  '10-eda-disk-geometry.png')
nb_png(EDA, 12, '11-eda-lag-correlation.png')
nb_png(EDA, 14, '12-eda-longitude-latitude.png')
nb_png(EDA, 16, '13-eda-ch-area-timeseries.png')
nb_png('experiments/p09-paired-cv/code_p9.ipynb', 20, '14-paired-cv-comparison.png')


# --------------------------------------------------------------------------
# Coronal-hole mask: limb correction ON (P4/P5-T1) vs OFF (P5-T2)
# The screenshots are 1287 px wide; the mask thumbnails sit in a 2x3 block.
# --------------------------------------------------------------------------
SHOTS = {
    'on':  'experiments/p05-threshold-sweep/trial1-limb-on/output-ch-detection.png',
    'off': 'experiments/p05-threshold-sweep/trial2-limb-off/output-ch-detection.png',
}
# (left, top, right, bottom) of each mask thumbnail, measured on the sources
CROPS = {
    'on':  [(382, 255, 597, 471), (690, 255, 906, 471), (382, 502, 597, 718)],
    'off': [(379, 253, 593, 467), (685, 253, 900, 467), (379, 499, 593, 713)],
}
TITLES = ['frame #0', 'frame #3379', 'frame #6758']


def fig_mask_comparison():
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.9))
    for row, key in enumerate(('off', 'on')):
        im = Image.open(os.path.join(DST, SHOTS[key]))
        for col, boxc in enumerate(CROPS[key]):
            ax = axes[row][col]
            ax.imshow(im.crop(boxc))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_edgecolor('#ccc')
            if row == 0:
                ax.set_title(TITLES[col], fontsize=10)
    axes[0][0].set_ylabel('P5-T2   limb OFF,  4 % threshold\n'
                          'CH area 3.1 %   ·   kept',
                          fontsize=10, color='#2a9d8f', fontweight='bold')
    axes[1][0].set_ylabel('P5-T1   limb ON,  8 % threshold\n'
                          'CH area 4.6 %   ·   dropped',
                          fontsize=10, color='#d1495b', fontweight='bold')
    fig.suptitle('The threshold, not the limb correction, is the dominant knob',
                 fontsize=13, fontweight='bold', y=0.975)
    fig.text(0.5, 0.923,
             'P5 separated the two knobs of the failed P4.  Loosening the threshold '
             'swells the mask into quiet Sun and rim;\n'
             "P4's own setting (limb correction ON + 12 % threshold) cost 3.8 km/s "
             'on validation.',
             ha='center', fontsize=9.3, color='#666', linespacing=1.5)
    fig.tight_layout(rect=[0, 0, 1, 0.895])
    out = os.path.join(FIG, '15-ch-mask-limb-comparison.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote 15-ch-mask-limb-comparison.png')


fig_mask_comparison()
