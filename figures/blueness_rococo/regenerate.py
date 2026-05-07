"""Render the blueness reward single-row grid for the rococo prompt."""
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "blueness_rococo"

for ttf in (REPO / "assets" / "fonts").glob("Lato-*.ttf"):
    fm.fontManager.addfont(str(ttf))

plt.style.use(str(REPO / "assets" / "paper.mplstyle"))

COLS = [
    ("Unguided", "unguided"),
    ("Guided", "gns50"),
    (r"Guided (lower $\lambda$)", "gns30"),
    (r"Guided ($k=8$)", "gns50_k8"),
    ("Guided (damped)", "gns100_damp0.1"),
]
IMG_IDX = 13
PROMPT = r"``Artist painting in the center of a cluttered room lit by candlelight, rococo''"


def cache_pngs(cond_dir: Path):
    sub = next(p for p in cond_dir.iterdir() if p.is_dir())
    return sorted(sub.glob("[0-9]*.png"))


fig, axes = plt.subplots(
    1,
    len(COLS),
    figsize=(10, 2.55),
    gridspec_kw={"wspace": 0.02},
)
fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.74)
fig.text(0.5, 0.965, PROMPT, ha="center", va="top", fontsize=12,
         color="#666666", style="italic")

for col_idx, (label, key) in enumerate(COLS):
    pngs = cache_pngs(DATA / key)
    ax = axes[col_idx]
    ax.imshow(Image.open(pngs[IMG_IDX]))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(label, pad=8)

fig.savefig(HERE / "blueness_rococo.pdf")
fig.savefig(HERE / "blueness_rococo.png")
print(f"saved -> {HERE}/blueness_rococo.{{pdf,png}}")
