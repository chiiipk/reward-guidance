"""Render the masked-brightness reward grid for the welder prompt (3 rows x 5 cols)."""
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "masked_brightness_welder"

for ttf in (REPO / "assets" / "fonts").glob("Lato-*.ttf"):
    fm.fontManager.addfont(str(ttf))

plt.style.use(str(REPO / "assets" / "paper.mplstyle"))

COLS = [
    ("Unguided", "unguided"),
    ("Guided", "gns100"),
    (r"Guided (lower $\lambda$)", "gns50"),
    (r"Guided ($k=8$)", "gns50_k8"),
    ("Guided (damped)", "gns100_damp0.1"),
]
ROWS = [0, 4, 12]
PROMPT = r"``Photorealistic worm's-eye view of a welder mid-spark inside a rusted ship hull, sweat, smoke, orange backlight''"


def cache_pngs(cond_dir: Path):
    sub = next(p for p in cond_dir.iterdir() if p.is_dir())
    return sorted(sub.glob("[0-9]*.png"))


fig, axes = plt.subplots(
    len(ROWS),
    len(COLS),
    figsize=(10, 6.65),
    gridspec_kw={"wspace": 0.02, "hspace": 0.02},
)
fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.90)
fig.text(0.5, 0.985, PROMPT, ha="center", va="top", fontsize=12,
         color="#666666", style="italic")

for col_idx, (label, key) in enumerate(COLS):
    pngs = cache_pngs(DATA / key)
    for row_idx, img_idx in enumerate(ROWS):
        ax = axes[row_idx, col_idx]
        ax.imshow(Image.open(pngs[img_idx]))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if row_idx == 0:
            ax.set_title(label, pad=8)

fig.savefig(HERE / "masked_brightness_welder.pdf")
fig.savefig(HERE / "masked_brightness_welder.png")
print(f"saved -> {HERE}/masked_brightness_welder.{{pdf,png}}")
