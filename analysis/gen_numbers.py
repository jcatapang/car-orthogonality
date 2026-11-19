"""Single source of truth: results JSON(s) -> numbers.tex (macros) + tables/*.tex + figure data.
Usage: python gen_numbers.py OUTDIR results.json [results_full.json ...]
"""
import json, sys, os, math
import numpy as np

OUT = sys.argv[1]
os.makedirs(f"{OUT}/tables", exist_ok=True); os.makedirs(f"{OUT}/figdata", exist_ok=True)
R = {}
for p in sys.argv[2:]:
    if os.path.exists(p):
        R.update(json.load(open(p)))

PAIRS = ["G-P", "G-R", "R-P"]
CONDS = ["original", "euclid_orth", "sigma_orth", "random_orth", "shuffled_orth"]
NICE = {"roberta-base": "RoBERTa-base", "roberta-large": "RoBERTa-large", "bert-base-uncased": "BERT-base",
        "answerdotai/ModernBERT-base": "ModernBERT-base", "gpt2": "GPT-2", "gpt2-medium": "GPT-2-medium",
        "EleutherAI/pythia-160m": "Pythia-160M", "EleutherAI/pythia-410m": "Pythia-410M", "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B",
        "Qwen/Qwen2.5-3B": "Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B": "Qwen2.5-7B", "Qwen/Qwen2.5-14B": "Qwen2.5-14B", "meta-llama/Llama-3.1-8B": "Llama-3.1-8B"}
ORDER = list(NICE)
DEFN = {"presence": "topic", "stereo": "stereotype", "factorial": "factorial"}

def g(m, k):
    v = m.get(k)
    return v["mean"] if isinstance(v, dict) else float("nan")

# ---------------------------------------------------------------- flatten
rows = []      # one row per (ds, model, layer, def, cond, pair)
layers_seen = {}
skipped = []
EXCLUDED_EMB = []
for key, v in R.items():
    ds, model, l = key.split("|"); l = int(l)
    layers_seen.setdefault(model, set()).add(l)
    if l == 0:   # embedding output: input-independent (encoders) or last-token-identity only (decoders)
        EXCLUDED_EMB.append(model)
        continue
    if not v.get("analysis"):
        skipped.append((model, l, v.get("layer_input_std", float("nan")), v.get("note", v.get("error", ""))))
        continue
    for df, cd in v["analysis"].items():
        for c in CONDS + ["leace"]:
            m = cd.get(c, {})
            if not m or "skipped" in m:
                continue
            for p in PAIRS:
                if f"mass_{p}" not in m:
                    continue
                rows.append(dict(ds=ds, model=model, layer=l, df=df, cond=c, pair=p, mass=g(m, f"mass_{p}"),
                                 excess=g(m, f"excess_{p}"), r=g(m, f"scorecorr_{p}"), pred=g(m, f"gausspred_{p}"),
                                 cos=g(m, f"cos_{p}"), mass3=g(m, "mass_3way"), ex3=g(m, "excess_3way"),
                                 auc=np.nanmean([g(m, f"auc_{c2}") for c2 in ("gender", "race", "profession")]),
                                 aucg=g(m, "auc_gender")))

def sel(**kw):
    return [r for r in rows if all((r[k] in v) if isinstance(v, (list, tuple, set)) else r[k] == v for k, v in kw.items())]

def arr(rs, k):
    a = np.array([r[k] for r in rs], float); return a[np.isfinite(a)]

macros = {}
def M(name, val, fmt="{:.2f}"):
    macros[name] = fmt.format(val) if isinstance(val, (float, int, np.floating, np.integer)) else str(val)

models_done = [m for m in ORDER if any(r["model"] == m for r in rows)]
M("nModels", len(models_done), "{:d}")
M("nLayerConfigs", len({(r["model"], r["layer"]) for r in rows}), "{:d}")
M("nSkippedLayers", len(skipped), "{:d}")

# ---------------------------------------------------------------- predictors of co-activation
P = [r for r in rows if r["cond"] in CONDS and np.isfinite(r["pred"]) and np.isfinite(r["mass"])]
obs, pred, cos = arr(P, "mass"), arr(P, "pred"), np.array([r["cos"] for r in P])
def r2(y, yhat): return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
cos_fit = np.polyval(np.polyfit(cos, obs, 1), cos)
M("nPredPoints", len(obs), "{:,d}")
M("rTwoOrthant", r2(obs, pred)); M("rTwoCos", r2(obs, cos_fit)); M("maeOrthant", np.mean(np.abs(obs - pred)), "{:.3f}")
# within the Euclidean-orthogonal condition, cosine is ~0 by construction: show it carries no information
E = [r for r in P if r["cond"] == "euclid_orth"]
def sci(x):
    m, e_ = f"{x:.0e}".split("e"); return f"\\ensuremath{{{m}\\times10^{{{int(e_)}}}}}"
M("maxAbsCosEuclid", sci(max(abs(r["cos"]) for r in E)))
M("massRangeEuclidLo", min(r["mass"] for r in E)); M("massRangeEuclidHi", max(r["mass"] for r in E))
np.savetxt(f"{OUT}/figdata/orthant.dat", np.c_[pred, obs, cos], header="pred obs cos", comments="", fmt="%.4f")

# ---------------------------------------------------------------- Euclidean vs Sigma orthogonalization
def paired(df_set, ds_set=("stereoset", "factorial")):
    out = []
    keys = {(r["ds"], r["model"], r["layer"], r["df"], r["pair"]) for r in rows if r["df"] in df_set and r["ds"] in ds_set}
    idx = {(r["ds"], r["model"], r["layer"], r["df"], r["pair"], r["cond"]): r for r in rows}
    for k in keys:
        try:
            out.append({c: idx[k + (c,)] for c in CONDS})
        except KeyError:
            pass
    return out
PA = paired({"presence", "stereo", "factorial"})
inc = np.array([abs(p["euclid_orth"]["excess"]) - abs(p["original"]["excess"]) for p in PA])
M("pctEuclidIncrease", 100 * np.mean(inc > 0), "{:.0f}")
M("maxEuclidIncrease", inc.max(), "{:+.2f}")
M("meanAbsExOrig", np.mean([abs(p["original"]["excess"]) for p in PA]), "{:.3f}")
M("nPairedCases", len(PA), "{:d}")
dE = np.array([abs(p["euclid_orth"]["excess"]) for p in PA]); dS = np.array([abs(p["sigma_orth"]["excess"]) for p in PA])
M("pctSigmaBetter", 100 * np.mean(dS < dE), "{:.0f}")
M("meanAbsExEuclid", dE.mean(), "{:.3f}"); M("meanAbsExSigma", dS.mean(), "{:.3f}")
M("medianExReduction", 100 * (1 - np.median(dS) / max(np.median(dE), 1e-9)), "{:.0f}")
rE = np.array([abs(p["euclid_orth"]["r"]) for p in PA]); rS = np.array([abs(p["sigma_orth"]["r"]) for p in PA])
M("meanAbsREuclid", np.nanmean(rE)); M("meanAbsRSigma", np.nanmean(rS))
# headline example: largest Euclidean excess
best = max(PA, key=lambda p: p["euclid_orth"]["excess"])
b = best["euclid_orth"]
M("exModel", NICE.get(b["model"], b["model"])); M("exLayer", b["layer"], "{:d}"); M("exDef", DEFN[b["df"]]); M("exPair", b["pair"])
M("exROrig", best["original"]["r"]); M("exREucl", b["r"]); M("exRSigma", best["sigma_orth"]["r"])
M("exMassOrig", best["original"]["mass"]); M("exMassEucl", b["mass"]); M("exMassSigma", best["sigma_orth"]["mass"])
M("exCosEucl", sci(abs(b["cos"])))

# controls
for c, nm in (("random_orth", "Rand"), ("shuffled_orth", "Shuf")):
    a = arr(sel(cond=c), "excess"); M(f"meanEx{nm}", a.mean(), "{:+.3f}"); M(f"maxAbsEx{nm}", np.abs(a).max(), "{:.3f}")
a = arr(sel(cond="random_orth"), "ex3"); M("meanExThreeRand", a.mean(), "{:+.3f}")

# three-way after Euclidean orthogonalization (the original paper's E5 quantity, with correct baseline)
t3 = {}
for r in sel(cond="euclid_orth", pair="G-P"):
    t3[(r["ds"], r["model"], r["layer"], r["df"])] = r["mass3"]
a = np.array(list(t3.values())); a = a[np.isfinite(a)]
M("massThreeEuclLo", a.min()); M("massThreeEuclHi", a.max()); M("massThreeEuclMean", a.mean())

# LEACE (pairwise G-P, correct 0.25 baseline)
L = arr(sel(cond="leace", pair="G-P"), "excess")
if len(L):
    M("meanExLeace", L.mean(), "{:+.3f}"); M("maxAbsExLeace", np.abs(L).max(), "{:.3f}")
    M("aucGLeace", np.nanmean(arr(sel(cond="leace", pair="G-P"), "aucg")))

# probe validity after orthogonalization
for c, nm in (("original", "Orig"), ("euclid_orth", "Eucl"), ("sigma_orth", "Sigma")):
    for df in ("presence", "stereo", "factorial"):
        a = arr(sel(cond=c, pair="G-P", df=df), "auc")
        if len(a): M(f"auc{nm}{df.capitalize()}", a.mean())

# factorial (labels independent by construction)
F = paired({"factorial"}, ("factorial",))
if F:
    for c, nm in (("original", "Orig"), ("euclid_orth", "Eucl"), ("sigma_orth", "Sigma"), ("random_orth", "Rand")):
        a = np.array([p[c]["excess"] for p in F]); M(f"factEx{nm}", a.mean(), "{:+.3f}"); M(f"factMaxEx{nm}", np.abs(a).max(), "{:.3f}")
    a = np.array([abs(p["euclid_orth"]["excess"]) - abs(p["original"]["excess"]) for p in F]); M("factPctEuclidIncrease", 100 * np.mean(a > 0), "{:.0f}")
    for c, nm in (("original", "Orig"), ("euclid_orth", "Eucl"), ("sigma_orth", "Sigma"), ("random_orth", "Rand")):
        M(f"factAbsEx{nm}", np.mean([abs(p[c]["excess"]) for p in F]), "{:.3f}")

# constant layers
M("constLayers", "; ".join(f"{NICE.get(m, m)} layer {l} (relative spread ${s:.0e}$)".replace("e-0", "\\times10^{-").replace("$)", "}$)") for m, l, s, _ in sorted(set(skipped))) or "none")

# embedding-layer diagnostics (excluded from all analyses above)
emb = []
for key, v in R.items():
    ds, model, l = key.split("|")
    if int(l) != 0: continue
    if not v.get("analysis"):
        emb.append((model, "constant", v.get("layer_input_std", float("nan")), None, None)); continue
    for df, cd in v["analysis"].items():
        for c in ("sigma_orth", "random_orth"):
            for p in PAIRS:
                m = cd[c]; emb.append((model, c, None, g(m, f"mass_{p}"), g(m, f"scorecorr_{p}")))
analysed_models = {m for m, k, *_ in emb if k != "constant"}
const_models = sorted({NICE.get(m, m) for m, k, *_ in emb if k == "constant" and m not in analysed_models})
M("embConstModels", ", ".join(const_models) if const_models else "none")
dec = [(m, mass, r) for m, k, _, mass, r in emb if k != "constant" and np.isfinite(mass)]
if dec:
    M("embMassMax", max(x[1] for x in dec)); M("embAbsRMax", max(abs(x[2]) for x in dec if np.isfinite(x[2])))
    M("embModels", ", ".join(sorted({NICE.get(m, m) for m, *_ in dec})))

# instruction tuning: Qwen2.5-3B base vs Instruct
for m, nm in (("Qwen/Qwen2.5-3B", "QBase"), ("Qwen/Qwen2.5-3B-Instruct", "QInst")):
    Pm = [r for r in P if r["model"] == m]
    if len(Pm) > 5:
        o, p_, c_ = arr(Pm, "mass"), arr(Pm, "pred"), np.array([r["cos"] for r in Pm])
        M(f"rTwoOrth{nm}", r2(o, p_)); M(f"rTwoCos{nm}", r2(o, np.polyval(np.polyfit(c_, o, 1), c_)))
        for c, cn in (("original", "Orig"), ("euclid_orth", "Eucl"), ("sigma_orth", "Sigma")):
            M(f"absEx{nm}{cn}", np.mean([abs(r["excess"]) for r in rows if r["model"] == m and r["cond"] == c and r["df"] == "presence"]), "{:.3f}")
for k in ["rTwoOrthQBase", "rTwoOrthQInst", "rTwoCosQBase", "rTwoCosQInst"] + [f"absEx{a}{b}" for a in ("QBase", "QInst") for b in ("Orig", "Eucl", "Sigma")]:
    macros.setdefault(k, r"\textbf{[pending]}")

# per-model share of topic-label excess removed by Sigma-orth and by Euclid-orth
red_s, red_e = [], []
for m in models_done:
    o = np.mean([abs(r["excess"]) for r in rows if r["model"] == m and r["df"] == "presence" and r["cond"] == "original"] or [np.nan])
    e = np.mean([abs(r["excess"]) for r in rows if r["model"] == m and r["df"] == "presence" and r["cond"] == "euclid_orth"] or [np.nan])
    g_ = np.mean([abs(r["excess"]) for r in rows if r["model"] == m and r["df"] == "presence" and r["cond"] == "sigma_orth"] or [np.nan])
    if np.isfinite(o): red_s.append(1 - g_ / o); red_e.append(1 - e / o)
M("sigmaRedLo", 100 * min(red_s), "{:.0f}"); M("sigmaRedHi", 100 * max(red_s), "{:.0f}")
M("euclRedLo", 100 * min(red_e), "{:.0f}"); M("euclRedHi", 100 * max(red_e), "{:.0f}")

# ---- per-label-set statistics with cluster bootstrap over models (conservative) + sign tests
from math import comb
_rng = np.random.default_rng(0)
def sign_p(k, n):
    lo = min(k, n - k); return min(1.0, 2 * sum(comb(n, i) for i in range(lo + 1)) / 2 ** n)
def cluster_boot(vals_by_model, B=4000):
    ms = list(vals_by_model); out = []
    for _ in range(B):
        pick = _rng.choice(len(ms), len(ms))
        out.append(np.mean(np.concatenate([vals_by_model[ms[i]] for i in pick])))
    return np.percentile(out, [2.5, 97.5])
for df, tag in (("presence", "Top"), ("factorial", "Fac"), ("stereo", "Ste")):
    PD = [p for p in PA if p["original"]["df"] == df]
    if not PD: continue
    dEO = {};
    for p in PD:
        dEO.setdefault(p["original"]["model"], []).append(abs(p["euclid_orth"]["excess"]) - abs(p["original"]["excess"]))
    dEO = {k: np.array(v) for k, v in dEO.items()}
    allv = np.concatenate(list(dEO.values())); k = int((allv > 0).sum()); n = len(allv)
    lo, hi = cluster_boot(dEO)
    M(f"dEO{tag}", allv.mean(), "{:+.4f}"); M(f"dEO{tag}Lo", lo, "{:+.4f}"); M(f"dEO{tag}Hi", hi, "{:+.4f}")
    M(f"pctEO{tag}", 100 * k / n, "{:.0f}"); M(f"nEO{tag}", n, "{:d}")
    pv = sign_p(k, n); ex = int(np.floor(np.log10(max(pv, 1e-300))))
    M(f"signP{tag}", f"\\ensuremath{{<10^{{{ex + 1}}}}}" if pv < 1e-3 else f"{pv:.2f}")
    for c, cn in (("original", "Orig"), ("euclid_orth", "Eucl"), ("sigma_orth", "Sigma"), ("random_orth", "Rand"), ("shuffled_orth", "Shuf")):
        M(f"abs{tag}{cn}", np.mean([abs(p[c]["excess"]) for p in PD]), "{:.3f}")
    dOR = {}
    for p in PD:
        dOR.setdefault(p["original"]["model"], []).append(abs(p["original"]["excess"]) - abs(p["random_orth"]["excess"]))
    dOR = {k: np.array(v) for k, v in dOR.items()}
    lo, hi = cluster_boot(dOR)
    M(f"dOR{tag}", np.concatenate(list(dOR.values())).mean(), "{:+.4f}"); M(f"dOR{tag}Lo", lo, "{:+.4f}"); M(f"dOR{tag}Hi", hi, "{:+.4f}")
    lst = [p for p in PD]
    M(f"pctSE{tag}", 100 * np.mean([abs(p["sigma_orth"]["excess"]) < abs(p["euclid_orth"]["excess"]) for p in lst]), "{:.0f}")

# orthant-law fit by label set and condition
for df, tag in (("presence", "Top"), ("factorial", "Fac"), ("stereo", "Ste")):
    Pm = [r for r in P if r["df"] == df]
    if len(Pm) > 5: M(f"rTwoOrth{tag}", r2(arr(Pm, "mass"), arr(Pm, "pred")))
Pc = [r for r in P if r["df"] == "presence" and abs(r["pred"] - 0.25) > 0.05]
if Pc: M("extremeBias", np.mean(arr(Pc, "mass") - arr(Pc, "pred")) * 1, "{:+.3f}")

# factorial layer trajectories, pair-averaged |excess|, for Figure 2
fl = {}
for r in rows:
    if r["df"] != "factorial" or r["cond"] not in ("original", "euclid_orth", "sigma_orth", "random_orth"): continue
    fl.setdefault(r["model"], {}).setdefault(r["cond"], {}).setdefault(r["layer"], []).append(abs(r["excess"]))
json.dump({m: {c: {l: float(np.mean(v)) for l, v in d2.items()} for c, d2 in d1.items()} for m, d1 in fl.items()},
          open(f"{OUT}/figdata/factorial_layers.json", "w"))

# ---- review-driven statistics
from scipy.stats import hypergeom
def null_abs(n):
    K = n // 2; ks = np.arange(0, K + 1); p_ = hypergeom(n, K, K).pmf(ks); return float(np.sum(p_ * np.abs(ks / n - 0.25)))
M("floorTop", null_abs(2536), "{:.4f}"); M("floorFac", null_abs(896), "{:.4f}")
# fair cosine test: original probes only (cosine not fixed at 0 by construction)
for df, tag in (("presence", "Top"), ("factorial", "Fac"), ("stereo", "Ste"), (None, "All")):
    Po = [r for r in P if r["cond"] == "original" and (df is None or r["df"] == df)]
    if len(Po) > 5:
        x = np.array([r["cos"] for r in Po]); y = arr(Po, "mass")
        M(f"cosRtwoOrig{tag}", r2(y, np.polyval(np.polyfit(x, y, 1), x))); M(f"orthRtwoOrig{tag}", r2(y, arr(Po, "pred")))
# signed topic excess per pair (original probes)
for p_, nm in (("G-P", "GP"), ("G-R", "GR"), ("R-P", "RP")):
    M(f"signTop{nm}", np.mean([r["excess"] for r in rows if r["df"] == "presence" and r["cond"] == "original" and r["pair"] == p_]), "{:+.3f}")
# orthant-law error on strongly dependent topic pairs
Ph = [r for r in P if r["df"] == "presence" and abs(r["r"]) > 0.3]
if Ph:
    M("hiRabsDev", np.mean(np.abs(arr(Ph, "mass") - arr(Ph, "pred"))), "{:.3f}"); M("hiRn", len(Ph), "{:d}")
# factorial three-way
for c, cn in (("original", "Orig"), ("euclid_orth", "Eucl"), ("sigma_orth", "Sigma"), ("random_orth", "Rand")):
    t = {}
    for r in rows:
        if r["df"] == "factorial" and r["cond"] == c: t[(r["model"], r["layer"])] = abs(r["ex3"])
    if t: M(f"triFac{cn}", np.mean(list(t.values())), "{:.4f}")
# model-level counts and exact sign-test counts
for df, tag in (("presence", "Top"), ("factorial", "Fac")):
    cnt = 0; tie = 0; nm_ = 0
    for m in models_done:
        o = np.mean([abs(r["excess"]) for r in rows if r["model"] == m and r["df"] == df and r["cond"] == "original"] or [np.nan])
        e = np.mean([abs(r["excess"]) for r in rows if r["model"] == m and r["df"] == df and r["cond"] == "euclid_orth"] or [np.nan])
        if np.isfinite(o): nm_ += 1; cnt += round(e, 3) > round(o, 3); tie += round(e, 3) == round(o, 3)
    M(f"modelsEgtO{tag}", cnt, "{:d}"); M(f"modelsTie{tag}", tie, "{:d}"); M(f"modelsN{tag}", nm_, "{:d}")
    PD = [p_ for p_ in PA if p_["original"]["df"] == df]
    M(f"kEO{tag}", int(sum(abs(p_["euclid_orth"]["excess"]) > abs(p_["original"]["excess"]) for p_ in PD)), "{:d}")
# AUC means for every condition and label set (case-weighted)
for c, cn in (("original", "Orig"), ("euclid_orth", "Eucl"), ("sigma_orth", "Sigma")):
    for df, tag in (("presence", "Top"), ("factorial", "Fac"), ("stereo", "Ste")):
        a = arr(sel(cond=c, pair="G-P", df=df), "auc")
        if len(a): M(f"auc{cn}{tag}", a.mean())
# LEACE: specify label set
for df, tag in (("presence", "Top"), ("factorial", "Fac")):
    a = [abs(r["excess"]) for r in rows if r["cond"] == "leace" and r["pair"] == "G-P" and r["df"] == df]
    g_ = [r["aucg"] for r in rows if r["cond"] == "leace" and r["pair"] == "G-P" and r["df"] == df]
    if a: M(f"leaceAbs{tag}", np.mean(a), "{:.3f}"); M(f"leaceAucG{tag}", np.nanmean(g_))

# ---------------------------------------------------------------- tables
def fmt(x, s="{:+.3f}"):
    return "--" if not np.isfinite(x) else s.format(x)

def main_table(dfs, caption, label, star=True):
    env = "table*" if star else "table"
    L = [r"\begin{%s}[t]" % env, r"\centering\small", caption, label, r"\begin{adjustbox}{max width=\textwidth}", r"\begin{tabular}{llcccccccc}", r"\toprule",
         r"Model & Labels & Original & Euclid-orth & $\Sigma$-orth & Random & Shuffled & LEACE & $|r|$ E\,/\,$\Sigma$ & AUC O\,/\,E\,/\,$\Sigma$ \\", r"\midrule"]
    for m in models_done:
        first = True
        for df in dfs:
            rs = [r for r in rows if r["model"] == m and r["df"] == df]
            if not rs: continue
            cell = lambda c: fmt(np.nanmean([abs(r["excess"]) for r in rs if r["cond"] == c]) if any(r["cond"] == c for r in rs) else float("nan"), "{:.3f}")
            lea = [abs(r["excess"]) for r in rs if r["cond"] == "leace" and r["pair"] == "G-P"]
            rr = lambda c: np.nanmean([abs(r["r"]) for r in rs if r["cond"] == c])
            L.append(f"{NICE.get(m, m) if first else ''} & {DEFN[df]} & {cell('original')} & {cell('euclid_orth')} & {cell('sigma_orth')} & "
                     f"{cell('random_orth')} & {cell('shuffled_orth')} & {fmt(np.nanmean(lea) if lea else float('nan'), '{:.3f}')} & "
                     f"{rr('euclid_orth'):.2f}\\,/\\,{rr('sigma_orth'):.2f} & " + "\\,/\\,".join(f"{np.nanmean([r['auc'] for r in rs if r['cond'] == c_]):.2f}" for c_ in ('original', 'euclid_orth', 'sigma_orth')) + " \\\\")
            first = False
        L.append(r"\addlinespace")
    L[-1:] = [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{%s}" % env]
    return "\n".join(L)

open(f"{OUT}/tables/tab_full.tex", "w").write(main_table(("presence", "factorial"),
    r"\caption{Full results, topic and factorial labels: mean $|O_{\text{excess}}|$ by condition, $|r|$ and AUC.}",
    r"\label{tab:full}"))

def compact():
    L = [r"\begin{table*}[t]", r"\centering\small",
         r"\caption{Mean $|O_{\text{excess}}|$ per model.}",
         r"\label{tab:main}", r"\begin{tabular}{lcccccccc}", r"\toprule",
         r" & \multicolumn{4}{c}{Topic labels (StereoSet)} & \multicolumn{4}{c}{Factorial labels (independent)} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Model & Original & Euclid. & $\Sigma$ & Random & Original & Euclid. & $\Sigma$ & Random \\", r"\midrule"]
    for m in models_done:
        cells = []
        for df in ("presence", "factorial"):
            for c in ("original", "euclid_orth", "sigma_orth", "random_orth"):
                a = [abs(r["excess"]) for r in rows if r["model"] == m and r["df"] == df and r["cond"] == c]
                cells.append(f"{np.mean(a):.3f}" if a else "--")
        L.append(f"{NICE.get(m, m)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)
open(f"{OUT}/tables/tab_main.tex", "w").write(compact())
open(f"{OUT}/tables/tab_stereo.tex", "w").write(main_table(("stereo",),
    r"\caption{As Table~\ref{tab:full}, stereotype labels.}",
    r"\label{tab:stereo}"))

# per-model predictor comparison
_orth_pm = []
lines = [r"\begin{table}[!htb]", r"\centering\small",
         r"\caption{Variance in co-activation explained ($R^2$) by cosine and the orthant law.}",
         r"\label{tab:predict}", r"\begin{adjustbox}{max width=\columnwidth}", r"\begin{tabular}{lrccc}", r"\toprule", r" & & \multicolumn{2}{c}{Original probes} & All \\", r"\cmidrule(lr){3-4}",
         r"Model & $n$ & Cosine & Orthant & Orthant \\", r"\midrule"]
for m in models_done:
    Pm = [r for r in P if r["model"] == m]; Po = [r for r in Pm if r["cond"] == "original"]
    if len(Po) < 5: continue
    xo = np.array([r["cos"] for r in Po]); yo = arr(Po, "mass")
    lines.append(f"{NICE.get(m, m)} & {len(Pm)} & {r2(yo, np.polyval(np.polyfit(xo, yo, 1), xo)):.2f} & {r2(yo, arr(Po, 'pred')):.2f} & {r2(arr(Pm, 'mass'), arr(Pm, 'pred')):.2f} \\\\")
    _orth_pm.append(r2(yo, arr(Po, 'pred')))
Po = [r for r in P if r["cond"] == "original"]; xo = np.array([r["cos"] for r in Po]); yo = arr(Po, "mass")
lines += [r"\midrule", f"All & {len(obs)} & {r2(yo, np.polyval(np.polyfit(xo, yo, 1), xo)):.2f} & {r2(yo, arr(Po, 'pred')):.2f} & {r2(obs, pred):.2f} \\\\", r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
open(f"{OUT}/tables/tab_predict.tex", "w").write("\n".join(lines))
M("orthRtwoLo", min(_orth_pm)); M("orthRtwoHi", max(_orth_pm))
M("nPredOrig", len([r for r in P if r["cond"] == "original"]), "{:,d}")

# layer trajectories for figure: per model, presence/stereo, G-P excess by condition
with open(f"{OUT}/figdata/layers.json", "w") as f:
    traj = {}
    for r in rows:
        if r["pair"] != "G-P": continue
        traj.setdefault(r["model"], {}).setdefault(r["df"], {}).setdefault(r["cond"], {})[r["layer"]] = r["excess"]
    json.dump(traj, f)

REQUIRED = ["embMassMax", "embAbsRMax", "embModels", "factAbsExOrig", "factAbsExEucl", "factAbsExSigma", "factAbsExRand", "factExOrig", "factExEucl", "factExSigma", "factExRand", "factMaxExOrig", "factMaxExEucl", "factMaxExSigma",
            "factMaxExRand", "factPctEuclidIncrease", "meanExLeace", "maxAbsExLeace", "aucGLeace"]
for k in REQUIRED:
    macros.setdefault(k, r"\textbf{[pending]}")
with open(f"{OUT}/numbers.tex", "w") as f:
    f.write("% AUTO-GENERATED by gen_numbers.py -- do not edit by hand\n")
    for k, v in macros.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in macros.items()))
