"""Render the VLM-reward best-of-n grid for the NEXT TRAIN MARS prompt (rows = settings, cols = best-of-n)."""
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "vlm_subway_mars"

for ttf in (REPO / "assets" / "fonts").glob("Lato-*.ttf"):
    fm.fontManager.addfont(str(ttf))

plt.style.use(str(REPO / "assets" / "paper.mplstyle"))

UNGUIDED_PERM = [19, 5, 1, 0, 6, 10, 11, 12, 8, 7, 18, 17, 13, 15, 4, 2, 3, 9, 14, 16]

ROW_GROUPS = [
    ("Unguided", "unguided", UNGUIDED_PERM),
    ("Guided", "gns100", 17),
    (r"Guided ($k=8$)", "gns50_k8", 17),
    ("Guided (damped)", "gns100_damp0.1", 23),
]
COL_GROUPS = [("Best-of-1", 1), ("Best-of-2", 2), ("Best-of-4", 4)]

INNER_WSPACE = 0.02  # gap within a best-of-n pair (inner cell width fraction)
# inter-group gap should be 3x inner gap, in inches.
# inner gap = INNER_WSPACE * (cell_w_fig / 2) * fig_w
# = 0.02 * (0.293/2) * 10 = 0.0293 in
# target group gap = 0.088 in
# row: HSPACE * cell_h_fig * fig_h = 0.088 -> HSPACE = 0.088/(0.234*6.4) = 0.0588
# col: WSPACE * cell_w_fig * fig_w = 0.088 -> WSPACE = 0.088/(0.293*10) = 0.030
SECTION_HSPACE = 0.0588
SECTION_WSPACE = 0.030

PROMPT = r"``cyberpunk subway platform with a holographic display that says NEXT TRAIN MARS, teal neon, commuters in silhouette''"


def load_condition(cond_dir: Path, order):
    sub = next(p for p in cond_dir.iterdir() if p.is_dir())
    pngs = sorted(sub.glob("[0-9]*.png"))
    rewards = np.load(sub / "rewards.npy")
    if isinstance(order, int):
        perm = np.random.RandomState(order).permutation(len(pngs))
    else:
        perm = np.asarray(order)
    return [pngs[i] for i in perm], rewards[perm]


def best_of(pngs, rewards, candidates):
    chosen = candidates[int(np.argmax(rewards[candidates]))]
    return Image.open(pngs[chosen])


fig = plt.figure(figsize=(10, 6.7))
fig.subplots_adjust(left=0.115, right=0.995, bottom=0.005, top=0.90)
fig.text(0.5, 0.985, PROMPT, ha="center", va="top", fontsize=12,
         color="#666666", style="italic")

outer = GridSpec(
    len(ROW_GROUPS),
    len(COL_GROUPS),
    figure=fig,
    wspace=SECTION_WSPACE,
    hspace=SECTION_HSPACE,
)

axes_grid = [[None] * len(COL_GROUPS) for _ in range(len(ROW_GROUPS))]

for ri, (_, key, order) in enumerate(ROW_GROUPS):
    pngs, rewards = load_condition(DATA / key, order)
    for ci, (_, n) in enumerate(COL_GROUPS):
        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[ri, ci], wspace=INNER_WSPACE, hspace=0)
        cell_axes = []
        for k in range(2):
            candidates = np.array([k + 2 * m for m in range(n)])
            ax = fig.add_subplot(inner[0, k])
            ax.imshow(best_of(pngs, rewards, candidates))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            cell_axes.append(ax)
        axes_grid[ri][ci] = cell_axes

fig.canvas.draw()

for ci, (label, _) in enumerate(COL_GROUPS):
    top_axes = axes_grid[0][ci]
    left_pos = top_axes[0].get_position()
    right_pos = top_axes[1].get_position()
    x = 0.5 * (left_pos.x0 + right_pos.x1)
    y = max(left_pos.y1, right_pos.y1) + 0.012
    fig.text(x, y, label, ha="center", va="bottom", fontsize=16)

for ri, (label, _, _) in enumerate(ROW_GROUPS):
    left_axes = axes_grid[ri][0]
    pos = left_axes[0].get_position()
    y = 0.5 * (pos.y0 + pos.y1)
    # match the row-gap (~0.088 inches => 0.0088 figure-x-fraction)
    x = pos.x0 - 0.009
    fig.text(x, y, label, ha="right", va="center", fontsize=16)

fig.savefig(HERE / "vlm_subway_mars.pdf")
fig.savefig(HERE / "vlm_subway_mars.png")
print(f"saved -> {HERE}/vlm_subway_mars.{{pdf,png}}")
