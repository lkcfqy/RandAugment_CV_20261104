"""Render coordinate inspection figures from fixed official exemplar boxes."""
from portable_dev.paths import project_path as _portable_project_path
from pathlib import Path
import os, json
ROOT=_portable_project_path(__file__, 'research-counting-exemplar-20261031')
os.environ['MPLCONFIGDIR']=str(ROOT/'reports/mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
rows=json.loads((ROOT/'reports/selection_frozen.json').read_text())['images']
out=ROOT/'reports/exemplar_inspection';out.mkdir(exist_ok=True)
for group in ['OTR_INTRA_BUT1_BUT2','OTR_INTRA_BEA1_BEA2','HOU_INTRA_HCP1_HCP2','HOU_INTRA_TPK1_TPK2']:
    selected=[x for x in rows if x['pair_id']==group]
    fig,axes=plt.subplots(2,6,figsize=(14,5.8),dpi=130)
    for row,x in enumerate(selected):
        im=Image.open(ROOT/'data/images'/x['filename'])
        for col,b in enumerate(x['A_boxes_xyxy']+x['B_boxes_xyxy']):
            ax=axes[row,col];x1,y1,x2,y2=b;pad=max(x2-x1,y2-y1)*.15
            ax.imshow(im,interpolation='nearest');ax.set_xlim(x1-pad,x2+pad);ax.set_ylim(y2+pad,y1-pad)
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='#008fce' if col<3 else '#d35210',linewidth=1))
            ax.set_title(('A' if col<3 else 'B')+str(col%3+1)+' source: '+x['A_prompt_source' if col<3 else 'B_prompt_source'],fontsize=8,wrap=True)
            ax.set_xticks([]);ax.set_yticks([])
        axes[row,0].set_ylabel(x['filename'],fontsize=6,rotation=90)
    fig.suptitle(group+' | fixed official boxes; no prediction; nearest pixel display',fontsize=12)
    fig.tight_layout();fig.savefig(out/(group+'.png'));plt.close(fig)
print('Created 4 coordinate-only inspection figures; source files unchanged.')
