"""Page-1 hero figure.
Top: the exact 2-D example of Appendix A (Sigma = diag(25, 1), rotated 30 degrees for layout; rotation preserves
cosines and co-activation). Shaded: inputs on which both probes fire. Monte-Carlo shares are checked against the
orthant law.
Bottom: Bias in Bios case study (real results): conditional score correlation of gender vs profession readouts.
Usage: python make_hero.py OUTDIR case_bios.json
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT, CASE = sys.argv[1], json.load(open(sys.argv[2]))
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e6e5e0"
import os
C_ORIG, C_EU, C_SG, C_SV = os.environ.get("PAL", "#0072B2,#D55E00,#009E73,#CC79A7").split(",")
C_LR = "#8a8984"
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})

th = np.deg2rad(30); Rm = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
S0 = np.diag([25.0, 1.0]); S = Rm @ S0 @ Rm.T
w1 = Rm @ np.array([1.0, -1.0]); w2s = Rm @ np.array([0.04, 1.0])
w2e = w2s - (w2s @ w1) / (w1 @ w1) * w1                     # Euclidean Gram-Schmidt of w2 against w1
rng = np.random.default_rng(0)
X = rng.multivariate_normal([0, 0], S, 20000)
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
rs = lambda a, b: (a @ S @ b) / np.sqrt((a @ S @ a) * (b @ S @ b))

fig = plt.figure(figsize=(3.15, 3.05))
gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.62, wspace=0.08, left=0.03, right=0.97, top=0.95, bottom=0.125)
panels = [(w2e, C_EU, "Euclidean-orthogonal"), (w2s, C_SG, r"$\Sigma$-orthogonal")]
for j, (w2, col, ttl) in enumerate(panels):
    ax = fig.add_subplot(gs[0, j]); ax.set_aspect("equal"); ax.axis("off")
    both = (X @ w1 > 0) & (X @ w2 > 0)
    pts = X[:1500]; b = both[:1500]
    ax.scatter(pts[~b, 0], pts[~b, 1], s=1.2, color="#c9c8c3", linewidths=0, rasterized=True)
    ax.scatter(pts[b, 0], pts[b, 1], s=1.2, color=col, linewidths=0, rasterized=True)
    L = 7.5
    for w, lab in ((w1, r"$\mathbf{w}_1$"), (w2, r"$\mathbf{w}_2$")):
        u = w / np.linalg.norm(w) * 5.2
        ax.add_patch(FancyArrowPatch((0, 0), tuple(u), arrowstyle="-|>", mutation_scale=7, lw=1.1, color=INK))
        ax.text(*(u * 1.18), lab, ha="center", va="center", fontsize=7.5, color=INK)
    ax.set_xlim(-L, L); ax.set_ylim(-L, L)
    share = both.mean(); law = 0.25 + np.arcsin(rs(w1, w2)) / (2 * np.pi)
    assert abs(share - law) < 0.01, (share, law)
    ax.set_title(ttl, fontsize=7.5, color=INK, pad=2)
    ax.text(0, -L - 0.4, f"cos = {abs(cos(w1, w2)) if abs(cos(w1, w2)) < 5e-3 else cos(w1, w2):.2f}\nboth fire on {100 * law:.0f}%", ha="center", va="top", fontsize=7, color=INK,
            linespacing=1.25)

ax = fig.add_subplot(gs[1, :]); pos = ax.get_position(); ax.set_position([0.34, pos.y0, 0.63, pos.height])
rows = [("dom|original", "DoM", C_ORIG), ("dom|euclid_orth", "DoM, Eucl. (cos 0)", C_EU),
        ("dom|sigma_orth", r"DoM, $\Sigma$-orth.", C_SG), ("lr|original", "LR probe", C_LR)]
for i, (k, lab, col) in enumerate(rows):
    y = len(rows) - 1 - i
    xs = np.array([d[k]["arc"] for d in CASE.values()])
    ax.scatter(xs, np.full(len(xs), y), marker="o", s=18, color=col, edgecolors="white", linewidths=0.6, zorder=3)
ax.set_xlim(-0.02, 0.92); ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows][::-1], fontsize=7); ax.tick_params(axis="y", length=0)
for t, r in zip(ax.get_yticklabels(), rows[::-1]): t.set_color(r[2] if r[2] != C_LR else INK)
ax.spines["left"].set_visible(False); ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_xlabel(r"gender–profession $|r_{\mathrm{cond}}|$ (one dot per model)", fontsize=7, labelpad=1.5)
ax.set_title("Bias in Bios (gender, profession independent)", fontsize=7.5, color=INK, pad=3, x=0.30)
fig.savefig(f"{OUT}/fig_hero.pdf"); fig.savefig(f"{OUT}/fig_hero.png", dpi=250)
print("ok")
