"""Figures for the revised paper, generated from figdata/ written by gen_numbers.py."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
C = {"original": "#0072B2", "euclid_orth": "#D55E00", "sigma_orth": "#009E73", "random_orth": "#8f8e86"}
LBL = {"original": "Original probes", "euclid_orth": "Euclidean-orthogonalized", "sigma_orth": "$\\Sigma$-orthogonalized",
       "random_orth": "Random directions"}
MK = {"original": "o", "euclid_orth": "s", "sigma_orth": "^", "random_orth": "x"}
LS = {"original": "-", "euclid_orth": "-", "sigma_orth": "--", "random_orth": ":"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False})

# ---- Fig 1: predicted vs observed co-activation
d = np.loadtxt(f"{OUT}/figdata/orthant.dat", skiprows=1)
pred, obs, cos = d[:, 0], d[:, 1], d[:, 2]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), sharey=True)
ax = axes[0]
ax.scatter(cos, obs, s=6, color=C["original"], alpha=0.45, linewidths=0)
ax.set_xlabel(r"Cosine of probe directions $\rho_{\mathrm{dir}}$"); ax.set_ylabel(r"Observed co-activation $\rho_{\mathrm{mass}}$")
ax.axhline(0.25, color=MUTED, lw=0.8, ls=":"); ax.text(-0.66, 0.40, "dotted: independence (0.25)", ha="left", va="bottom", color=MUTED, fontsize=7)
ax.set_title("(a) Against cosine of probe directions", loc="left", fontsize=8, color=INK)
ax = axes[1]
ax.scatter(pred, obs, s=6, color=C["euclid_orth"], alpha=0.45, linewidths=0)
lo, hi = min(pred.min(), obs.min()), max(pred.max(), obs.max())
ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8); ax.text(hi, hi, "  $y=x$", va="center", fontsize=7, color=INK)
ax.axhline(0.25, color=MUTED, lw=0.8, ls=":")
ax.set_xlabel(r"Orthant law: $\frac{1}{4}+\frac{1}{2\pi}\arcsin r$ (held-out score correlation)")
ax.set_title("(b) Against the orthant law", loc="left", fontsize=8, color=INK)
fig.tight_layout(w_pad=2.0); fig.savefig(f"{OUT}/fig_orthant.pdf"); fig.savefig(f"{OUT}/fig_orthant.png", dpi=200)
# single-column version (orthant law only) for the main text
fig1, ax1 = plt.subplots(figsize=(3.1, 2.4))
ax1.scatter(pred, obs, s=5, color="#5a5955", alpha=0.45, linewidths=0, rasterized=True)
ax1.plot([lo, hi], [lo, hi], color=INK, lw=0.8); ax1.text(hi, hi, "  $y=x$", va="center", fontsize=7, color=INK)
ax1.axhline(0.25, color=MUTED, lw=0.8, ls=":")
ax1.set_xlabel(r"Orthant law $\frac{1}{4}+\frac{1}{2\pi}\arcsin r$", fontsize=7.5)
ax1.set_ylabel(r"Observed co-activation $\rho_{\mathrm{mass}}$", fontsize=7.5)
fig1.tight_layout(); fig1.savefig(f"{OUT}/fig_orthant1.pdf"); fig1.savefig(f"{OUT}/fig_orthant1.png", dpi=200)

# ---- Fig 2: independent (factorial) labels, pair-averaged |O_excess| by layer, with sampling-noise floor
fl = json.load(open(f"{OUT}/figdata/factorial_layers.json"))
from scipy.stats import hypergeom
_K = 448; _ks = np.arange(0, _K + 1); NOISE = float(np.sum(hypergeom(896, _K, _K).pmf(_ks) * np.abs(_ks / 896 - 0.25)))  # exact single-seed null
names = {"roberta-base": "RoBERTa-base", "gpt2": "GPT-2", "EleutherAI/pythia-160m": "Pythia-160M"}
panels = [m for m in names if m in fl]
if panels:
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels) + 1.2, 2.1), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], panels):
        ax.axhspan(0, NOISE, color="#ecebe6", lw=0)
        for c in ("original", "euclid_orth", "sigma_orth", "random_orth"):
            sdat = fl[m].get(c, {})
            if not sdat: continue
            L = sorted(int(k) for k in sdat); y = [sdat[str(k)] for k in L]
            ax.plot(L, y, color=C[c], ls=LS[c], marker=MK[c], ms=3, lw=1.4, label=LBL[c])
        ax.set_title(names[m], loc="left", fontsize=8, color=INK); ax.set_xlabel("Layer")
    top = max(v for m in panels for c in fl[m].values() for v in c.values()) * 1.08
    for ax in axes[0]: ax.set_ylim(0, top)
    axes[0][0].set_ylabel(r"mean $|O_{\mathrm{excess}}|$ (factorial labels)")
    axes[0][-1].text(axes[0][-1].get_xlim()[1], NOISE, " sampling\n noise", va="center", fontsize=6.5, color=MUTED)
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.12, 1.0), fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_layers.pdf"); fig.savefig(f"{OUT}/fig_layers.png", dpi=200)
print("figures written")
