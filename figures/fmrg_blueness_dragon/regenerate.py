"""Render the FMRG vs. plug-in comparison on the dragon prompt with blueness reward.

3 rows x 4 cols: Unguided, Guided (k = 1) plug-in, Guided (FMRG). All three
rows use the same prompt and seed; the methods differ in their lookahead
(Diamond renoise for the plug-in, deterministic flow-map endpoint for FMRG).
"""
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "fmrg_blueness_dragon"

for ttf in (REPO / "assets" / "fonts").glob("Lato-*.ttf"):
    fm.fontManager.addfont(str(ttf))
plt.style.use(str(REPO / "assets" / "paper.mplstyle"))


def cache_pngs(cond_dir: Path):
    sub = next(p for p in cond_dir.iterdir() if p.is_dir())
    return sorted(sub.glob("[0-9]*.png"))


ROWS = [
    ("Unguided",          "dragon_unguided"),
    (r"Guided ($k = 1$)", "plugin"),
    ("Guided (FMRG)",     "fmrg"),
]
N_COLS = 4
PROMPT = r"``a massive dragon perched on basalt cliffs above lava waterfalls, volcanic ash, crimson sunset, ultra-detailed fantasy''"

fig, axes = plt.subplots(
    len(ROWS), N_COLS,
    figsize=(10, 6.85),
    gridspec_kw={"wspace": 0.02, "hspace": 0.02},
)
fig.subplots_adjust(left=0.16, right=0.998, bottom=0.005, top=0.96)
fig.text(0.5, 0.985, PROMPT, ha="center", va="top", fontsize=12,
         color="#666666", style="italic")

for r, (label, key) in enumerate(ROWS):
    pngs = cache_pngs(DATA / key)
    for c in range(N_COLS):
        ax = axes[r, c]
        ax.imshow(Image.open(pngs[c]))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

for r, (label, _) in enumerate(ROWS):
    pos = axes[r, 0].get_position()
    y = 0.5 * (pos.y0 + pos.y1)
    x = pos.x0 - 0.025
    fig.text(x, y, label, ha="right", va="center", fontsize=20)

fig.savefig(HERE / "fmrg_blueness_dragon.pdf")
fig.savefig(HERE / "fmrg_blueness_dragon.png")
print(f"saved -> {HERE}/fmrg_blueness_dragon.{{pdf,png}}")
