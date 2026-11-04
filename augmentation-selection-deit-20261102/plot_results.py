"""Display every paired setting; descriptive figure, no selection or fitting."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"]=str(ROOT/".mplconfig")
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
s=json.loads((ROOT/"SUMMARY.json").read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True,constrained_layout=True)
for ax,B in zip(axes,[8,16]):
    rows=sorted([r for r in s["all_settings"] if r["B"]==B],key=lambda r:r["episode"])
    for name,label,color,marker,offset in [
        ("selected_minus_weak_pp","CV selected minus fixed weak","#0072B2","o",-.10),
        ("selected_minus_ra_pp","CV selected minus fixed RA","#D55E00","^",.10)]:
        ax.scatter([r["episode"]+offset for r in rows],[r[name] for r in rows],label=label,c=color,marker=marker,s=38)
    ax.axhline(0,color="#444444",lw=1)
    ax.set_title(f"B={B} labels/class");ax.set_xlabel("Support episode (fixed before utility)");ax.set_xticks(range(12))
    ax.grid(axis="y",alpha=.18);ax.spines[["top","right"]].set_visible(False)
axes[0].set_ylabel("Paired query accuracy difference (percentage points)")
axes[1].legend(loc="upper right",fontsize=8)
fig.suptitle("EuroSAT / frozen DeiT-Tiny: all 24 exploratory settings",fontsize=12)
fig.savefig(ROOT/"paired_differences.png",dpi=180)
fig.savefig(ROOT/"paired_differences.svg")

