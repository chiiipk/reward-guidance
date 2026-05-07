"""Render the VLM-reward best-of-n grid for the ECLIPSE DINER prompt (rows = settings, cols = best-of-n)."""
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA = REPO / "data" / "vlm_diner_eclipse"

for ttf in (REPO / "assets" / "fonts").glob("Lato-*.ttf"):
    fm.fontManager.addfont(str(ttf))

plt.style.use(str(REPO / "assets" / "paper.mplstyle"))

# Curated unguided ordering: positions 0-1 are low-reward (best-of-1 misses);
# positions 2-3 are mid-reward (best-of-2 partially recovers); positions 4-7
# are high-reward (best-of-4 lands on a clear ECLIPSE sign).
UNGUIDED_PERM = [5, 7, 2, 6, 0, 11, 15, 17, 1, 3, 4, 8, 9, 10, 12, 13, 14, 16, 18, 19]

ROW_GROUPS = [
    ("Unguided", "unguided", UNGUIDED_PERM),
    ("Guided", "gns100", 17),
    (r"Guided ($k=8$)", "gns50_k8", 17),
    ("Guided (damped)", "gns100_damp0.1", 23),
]
COL_GROUPS = [("Best-of-1", 1), ("Best-of-2", 2), ("Best-of-4", 4)]

INNER_WSPACE = 0.02  # gap within a best-of-n pair (inner cell width fraction)
SECTION_HSPACE = 0.0588
SECTION_WSPACE = 0.030

PROMPT = (
    r"``A roadside American diner in the Nevada desert, shot at twilight, a neon sign on the roof glowing ECLIPSE DINER," "\n"
    r"in cherry-red and cream tubes, a long empty highway behind it, painterly warm light on chrome surfaces''"
)


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


fig = plt.figure(figsize=(10, 6.95))
fig.subplots_adjust(left=0.115, right=0.995, bottom=0.005, top=0.87)
fig.text(0.5, 0.985, PROMPT, ha="center", va="top", fontsize=12,
         color="#666666", style="italic", multialignment="center")

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
    x = pos.x0 - 0.009
    fig.text(x, y, label, ha="right", va="center", fontsize=16)

fig.savefig(HERE / "vlm_diner_eclipse.pdf")
fig.savefig(HERE / "vlm_diner_eclipse.png")
print(f"saved -> {HERE}/vlm_diner_eclipse.{{pdf,png}}")
