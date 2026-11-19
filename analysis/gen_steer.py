"""Phase-4 steering results -> numbers_steer.tex macros, tables/tab_steer.tex, fig_steer.pdf.
Usage: python gen_steer.py OUTDIR steer1.json [steer2.json ...]
"""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
R = {}
for p in sys.argv[2:]:
    if os.path.exists(p):
        R.update(json.load(open(p)))
NICE = {"gpt2": "GPT-2", "gpt2-medium": "GPT-2-medium", "EleutherAI/pythia-160m": "Pythia-160M", "EleutherAI/pythia-410m": "Pythia-410M", "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B": "Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct", "tiny": "tiny", "Qwen/Qwen2.5-7B": "Qwen2.5-7B", "Qwen/Qwen2.5-14B": "Qwen2.5-14B", "meta-llama/Llama-3.1-8B": "Llama-3.1-8B"}
VN_ALL = ["original", "euclid_orth", "sigma_orth", "sigma_vec", "euclid_big"]
macros = {}
def M(k, v, f="{:.2f}"): macros[k] = f.format(v) if isinstance(v, (float, int, np.floating)) else str(v)

VN = [v for v in VN_ALL if any(f"|{v}|" in k for res in R.values() for k in res)]
HDR = {"euclid_big": r"Eucl.@$\|\Sigma\mathbf{w}^{\Sigma}\|$", "random_big": r"Rand.@$\|\Sigma\mathbf{w}^{\Sigma}\|$", "original": "Orig.", "euclid_orth": "Eucl.", "sigma_orth": r"$\Sigma$ (covector)", "sigma_vec": r"$\Sigma\mathbf{w}^{\Sigma}$"}
# collect: per model, per steering block, per vector: (steer-layer dP, final dG, final dP, mean downstream |dP|)
recs = []
for m, res in R.items():
    meta = res.get("_meta", {})
    for b in meta.get("steer_blocks", []):
        for vn in VN + ["random", "random_big"]:
            ks = sorted([k for k in res if k.startswith(f"block{b}|{vn}|")], key=lambda k: int(k.split("|h")[1]))
            if not ks: continue
            first, last = res[ks[0]], res[ks[-1]]
            down = [res[k] for k in ks[1:]] or [last]
            recs.append(dict(model=m, block=b, vn=vn, dP0=first["d_prof"], dG0=first["d_gender"], dGf=last["d_gender"], dPf=last["d_prof"],
                             spillf=abs(last["d_prof"]) / max(abs(last["d_gender"]), 1e-9),
                             mean_absdP=np.mean([abs(x["d_prof"]) for x in down]), mean_absdG=np.mean([abs(x["d_gender"]) for x in down])))

def agg(vn, key, models=None):
    a = [r[key] for r in recs if r["vn"] == vn and (models is None or r["model"] in models)]
    return float(np.mean(a)) if a else float("nan")

models = [m for m in NICE if m in R]
M("steerNModels", len(models), "{:d}")
for vn, nm in (("original", "Orig"), ("euclid_orth", "Eucl"), ("sigma_orth", "Sigma"), ("sigma_vec", "SigVec"), ("random", "Rand")):
    if not any(r["vn"] == vn for r in recs): continue
    M(f"steerSpill{nm}", agg(vn, "spillf")); M(f"steerAbsdPZero{nm}", np.mean([abs(r["dP0"]) for r in recs if r["vn"] == vn]) if recs else float("nan"))
    M(f"steerDownAbsdP{nm}", agg(vn, "mean_absdP")); M(f"steerFinaldG{nm}", agg(vn, "dGf"))
eu = [r for r in recs if r["vn"] == "euclid_orth"]; sg = {(r["model"], r["block"]): r for r in recs if r["vn"] == "sigma_orth"}
pairs = [(r, sg[(r["model"], r["block"])]) for r in eu if (r["model"], r["block"]) in sg]
if pairs:
    M("steerPctEuclLess", 100 * np.mean([e["mean_absdP"] < s["mean_absdP"] for e, s in pairs]), "{:.0f}")
    M("steerNCases", len(pairs), "{:d}")
for vn, nm in (("original", "Orig"), ("euclid_orth", "Eucl"), ("sigma_orth", "Sigma"), ("sigma_vec", "SigVec")):
    per = [agg(vn, "mean_absdP", [m]) for m in models if any(r["vn"] == vn and r["model"] == m for r in recs)]
    if per: M(f"steerLo{nm}", min(per)); M(f"steerHi{nm}", max(per))
sv = {(r["model"], r["block"]): r for r in recs if r["vn"] == "sigma_vec"}
pv = [(e, sv[(e["model"], e["block"])]) for e in eu if (e["model"], e["block"]) in sv]
if pv:
    M("steerPctSigVecLess", 100 * np.mean([s_["mean_absdP"] < e["mean_absdP"] for e, s_ in pv]), "{:.0f}")
    M("steerNCasesSigVec", len(pv), "{:d}")
for k in ["steerPctEuclLess", "steerNCases", "steerPctSigVecLess", "steerNCasesSigVec", "steerLoSigVec", "steerHiSigVec", "steerAbsdPZeroSigVec",
          "nSigVecBetter", "sigVecWorseModels", "worseRatioSigVec", "worseRatioEucl", "gainLoBetter", "gainHiBetter", "ratioEuclBetterLo", "ratioEuclBetterHi", "ratioSigVecBetterLo", "ratioSigVecBetterHi"]:
    macros.setdefault(k, r"\textbf{[pending]}")

# ratio of downstream spillover to downstream target effect, per model and vector
def ratio(vn, m):
    rs = [r for r in recs if r["vn"] == vn and r["model"] == m]
    return np.mean([r["mean_absdP"] for r in rs]) / max(np.mean([r["mean_absdG"] for r in rs]), 1e-9) if rs else float("nan")
def dgd(vn, m):
    rs = [r for r in recs if r["vn"] == vn and r["model"] == m]
    return np.mean([r["mean_absdG"] for r in rs]) if rs else float("nan")
if "sigma_vec" in VN:
    rv = {vn: [ratio(vn, m) for m in models] for vn in VN}
    for vn, nm in (("original", "Orig"), ("euclid_orth", "Eucl"), ("sigma_orth", "Sigma"), ("sigma_vec", "SigVec")):
        M(f"ratioLo{nm}", min(rv[vn])); M(f"ratioHi{nm}", max(rv[vn]))
    better = [m for m in models if ratio("sigma_vec", m) < ratio("euclid_orth", m)]
    M("nSigVecBetter", len(better), "{:d}")
    worse = [NICE[m] for m in models if m not in better]
    M("sigVecWorseModels", ", ".join(worse) if worse else "none")
    wm = [m for m in models if m not in better]
    if wm: M("worseRatioSigVec", ratio("sigma_vec", wm[0])); M("worseRatioEucl", ratio("euclid_orth", wm[0]))
    gains = [dgd("sigma_vec", m) / dgd("euclid_orth", m) for m in models]
    M("gainLo", min(gains), "{:.1f}"); M("gainHi", max(gains), "{:.1f}")
    gb = [dgd("sigma_vec", m) / dgd("euclid_orth", m) for m in models if m in better]
    if gb: M("gainLoBetter", min(gb), "{:.1f}"); M("gainHiBetter", max(gb), "{:.1f}")
    rb = {vn: [ratio(vn, m) for m in better] for vn in ("euclid_orth", "sigma_vec")}
    if better: M("ratioEuclBetterLo", min(rb["euclid_orth"])); M("ratioEuclBetterHi", max(rb["euclid_orth"])); M("ratioSigVecBetterLo", min(rb["sigma_vec"])); M("ratioSigVecBetterHi", max(rb["sigma_vec"]))

# table: downstream |dGender| and |dProfession| for every vector incl. random, and the spillover ratio
VT = [v for v in ("euclid_orth", "sigma_vec", "euclid_big", "random_big") if any(r["vn"] == v for r in recs)]
VR = [v for v in ("original", "euclid_orth", "sigma_orth", "sigma_vec") if v in VN]   # rescaling leaves a ratio unchanged
HDR2 = dict(HDR, random="Random")
def dpd(vn, m):
    rs = [r for r in recs if r["vn"] == vn and r["model"] == m]
    return np.mean([r["mean_absdP"] for r in rs]) if rs else float("nan")
L = [r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{3.5pt}",
     r"\caption{Steering along gender: downstream $|\Delta g|$\,/\,$|\Delta p|$ (score SDs) and spillover ratio $|\Delta p|/|\Delta g|$.}",
     r"\label{tab:steer}", r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{l" + "c" * len(VT) + "|" + "c" * len(VR) + "}", r"\toprule",
     r" & \multicolumn{%d}{c|}{$|\Delta g|$ / $|\Delta p|$} & \multicolumn{%d}{c}{Spillover ratio} \\" % (len(VT), len(VR)),
     "Model & " + " & ".join(HDR2[v] for v in VT) + " & " + " & ".join(HDR[v] for v in VR) + r" \\", r"\midrule"]
for m in models:
    L.append(f"{NICE[m]} & " + " & ".join(f"{dgd(v, m):.2f} / {dpd(v, m):.2f}" for v in VT) + " & " +
             " & ".join((r"\textbf{%.2f}" if ratio(v, m) == min(ratio(u, m) for u in VR) else "%.2f") % ratio(v, m) for v in VR) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}"]
# covector vs original
cov_better = [NICE[m] for m in models if ratio("sigma_orth", m) < ratio("original", m)]
M("nCovBeatsOrig", len(cov_better), "{:d}")
cw = [m for m in models if all(ratio("sigma_orth", m) >= ratio(v, m) for v in VN)]
M("covWorstModels", ", ".join(NICE[m] for m in cw)); M("nCovWorst", len(cw), "{:d}")
M("randAbsdGLo", min(dgd("random", m) for m in models)); M("randAbsdGHi", max(dgd("random", m) for m in models))
M("randAbsdPLo", min(dpd("random", m) for m in models)); M("randAbsdPHi", max(dpd("random", m) for m in models))
M("absdPSigVecLo", min(dpd("sigma_vec", m) for m in models)); M("absdPSigVecHi", max(dpd("sigma_vec", m) for m in models))
M("absdPEuclLo", min(dpd("euclid_orth", m) for m in models)); M("absdPEuclHi", max(dpd("euclid_orth", m) for m in models))
nlow = sum(dpd("sigma_vec", m) < dpd("euclid_orth", m) for m in models); M("nSigVecLowerAbsdP", nlow, "{:d}")
os.makedirs(f"{OUT}/tables", exist_ok=True)
open(f"{OUT}/tables/tab_steer.tex", "w").write("\n".join(L))

# figure: profession shift by relative downstream depth, averaged over steering blocks, per model
C = {"original": "#0072B2", "euclid_orth": "#D55E00", "sigma_orth": "#009E73", "sigma_vec": "#CC79A7"}
LBL = {"original": "Original", "euclid_orth": "Euclidean-orth.", "sigma_orth": "$\\Sigma$-orth. covector", "sigma_vec": "$\\Sigma\\mathbf{w}^{\\Sigma}$ vector"}
MK = {"original": "o", "euclid_orth": "s", "sigma_orth": "^", "sigma_vec": "D"}
if models:
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "legend.frameon": False})
    fig, axes = plt.subplots(1, len(models), figsize=(1.9 * len(models) + 0.9, 2.0), sharey=True, squeeze=False)
    bins = np.linspace(0, 1, 6)
    for ax, m in zip(axes[0], models):
        res = R[m]
        for vn in [v for v in VN if v in C]:
            pts = [(v["rel_depth"], v["d_prof"]) for k, v in res.items() if not k.startswith("_") and f"|{vn}|" in k]
            if not pts: continue
            x, y = np.array(pts).T
            idx = np.digitize(x, bins[1:-1]); xs = [x[idx == i].mean() for i in range(len(bins) - 1) if np.any(idx == i)]
            ys = [y[idx == i].mean() for i in range(len(bins) - 1) if np.any(idx == i)]
            ax.plot(xs, ys, color=C[vn], marker=MK[vn], ms=3, lw=1.4, label=LBL[vn])
        ax.axhline(0, color="#1f1f1e", lw=0.6); ax.set_title(NICE[m], loc="left", fontsize=8); ax.set_xlabel("Depth after steering layer")
    axes[0][0].set_ylabel("$\\Delta$ profession readout (SD)")
    axes[0][-1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_steer.pdf"); fig.savefig(f"{OUT}/fig_steer.png", dpi=200)

with open(f"{OUT}/numbers_steer.tex", "w") as f:
    f.write("% AUTO-GENERATED by gen_steer.py\n")
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in macros.items()))
# absolute spillover in the models where the pattern is best by ratio
if "sigma_vec" in VN and better:
    extra = {}
    extra["absdPSigVecBetterLo"] = min(dpd("sigma_vec", m) for m in better); extra["absdPSigVecBetterHi"] = max(dpd("sigma_vec", m) for m in better)
    extra["absdPEuclBetterLo"] = min(dpd("euclid_orth", m) for m in better); extra["absdPEuclBetterHi"] = max(dpd("euclid_orth", m) for m in better)
    with open(f"{OUT}/numbers_steer.tex", "a") as f:
        for k, v in extra.items(): f.write(f"\\newcommand{{\\{k}}}{{{v:.2f}}}\n")
    print(extra)

# equal-norm comparison (pattern vs Euclidean rescaled to the pattern's norm)
if any(r["vn"] == "euclid_big" for r in recs):
    eb = [m for m in models if any(r["vn"] == "euclid_big" and r["model"] == m for r in recs)]
    win = [m for m in eb if ratio("sigma_vec", m) < ratio("euclid_big", m)]
    ex = {"nEqNormBetter": len(win), "nEqNorm": len(eb),
          "eqNormLoser": ", ".join(NICE[m] for m in eb if m not in win) or "none",
          "eqPSigLo": min(dpd("sigma_vec", m) for m in win), "eqPSigHi": max(dpd("sigma_vec", m) for m in win),
          "eqPEuLo": min(dpd("euclid_big", m) for m in win), "eqPEuHi": max(dpd("euclid_big", m) for m in win),
          "eqGEuLo": min(dgd("euclid_big", m) for m in win), "eqGEuHi": max(dgd("euclid_big", m) for m in win),
          "eqGSigLo": min(dgd("sigma_vec", m) for m in win), "eqGSigHi": max(dgd("sigma_vec", m) for m in win),
          "eqRandG": max(dgd("random_big", m) for m in win),
          "eqRandGLoser": max([dgd("random_big", m) for m in eb if m not in win] or [float("nan")]),
          "eqNormRatioLo": min(ratio("sigma_vec", m) / ratio("euclid_big", m) for m in win),
          "eqNormRatioHi": max(ratio("sigma_vec", m) / ratio("euclid_big", m) for m in win)}
    with open(f"{OUT}/numbers_steer.tex", "a") as f:
        for k, v in ex.items(): f.write(f"\\newcommand{{\\{k}}}{{{v if isinstance(v, (int, str)) else f'{v:.2f}'}}}\n")
    print(ex)
