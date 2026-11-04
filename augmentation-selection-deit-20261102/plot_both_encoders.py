"""Post-result descriptive plot; every frozen setting, no fitting or selection."""
from pathlib import Path
import os, json
ROOT = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(ROOT / '.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

s = json.loads((ROOT / 'BOTH_ENCODERS_SUMMARY.json').read_text())
fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), sharex=True, sharey=True)
for i, (model, title) in enumerate([('dinov2_small', 'DINOv2-Small'), ('deit_tiny', 'DeiT-Tiny')]):
    for j, B in enumerate([8, 16]):
        ax = axes[i, j]
        rows = sorted((r for r in s['all_48_settings'] if r['encoder'] == model and r['B'] == B), key=lambda r: r['episode'])
        assert len(rows) == 12
        for name, label, color, marker, dx in [
            ('selected_minus_weak_pp', 'CV minus fixed weak', '#0072B2', 'o', -.1),
            ('selected_minus_ra_pp', 'CV minus fixed RA', '#D55E00', '^', .1)]:
            ax.scatter([r['episode'] + dx for r in rows], [r[name] for r in rows], c=color, marker=marker, s=30, label=label)
        ax.axhline(0, color='#444444', lw=1)
        ax.set_title(f'{title}, B={B} labels/class')
        ax.set_xticks(range(12))
        ax.grid(axis='y', alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
axes[0, 1].legend(loc='upper right', fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel('Paired accuracy difference (pp)')
for ax in axes[1]:
    ax.set_xlabel('Support episode')
fig.suptitle('EuroSAT: all 48 related exploratory settings', fontsize=13)
fig.text(.5, .015, 'Same query identities; overlapping supports; native clean preprocessing differs. No independence claim.', ha='center', fontsize=8)
fig.tight_layout(rect=[0, .04, 1, .96])
for extension in ['png', 'svg']:
    fig.savefig(ROOT / ('both_encoders_paired.' + extension), dpi=180)
