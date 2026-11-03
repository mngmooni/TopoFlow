"""Shared plotting helpers."""
from matplotlib.patches import Circle as MplCircle, Rectangle

from .env import Circle


def draw_env(ax, env, show_cuts=True):
    for j, o in enumerate(env.obstacles):
        if isinstance(o, Circle):
            ax.add_patch(MplCircle((o.cx, o.cy), o.r, color="0.25", zorder=3))
        else:
            ax.add_patch(Rectangle((o.cx - o.hw, o.cy - o.hh), 2 * o.hw, 2 * o.hh, color="0.25", zorder=3))
        if show_cuts:
            ax.plot([o.cx, o.cx], [o.cut_top(), -1.2], ls="--", lw=1, color="crimson", zorder=2)
            ax.text(o.cx + 0.02, -1.12, "abcdefg"[j], color="crimson", fontsize=9)
    ax.plot(*env.start, "o", color="k", ms=6, zorder=4)
    ax.plot(*env.goal, "*", color="k", ms=10, zorder=4)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
