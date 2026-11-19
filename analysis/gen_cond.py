"""Tier A+B conditional-entanglement results -> numbers_cond.tex macros + tables/tab_cond.tex.
Usage: python gen_cond.py OUTDIR main.json [main_qwen3b.json] --tpl tpl.json [tpl_qwen3b.json]
Factorial numbers come from the template-conditioned run (cells = template x labels); all other label sets
from the main run (cells = label combinations).
"""
import json, sys, os, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("main", nargs="+"); ap.add_argument("--tpl", nargs="+", default=[])
ap.add_argument("--bios", nargs="+", default=[], help="Bios-only runs with the projection audit; used for the case study")
ap.add_argument("--both", nargs="+", default=[], help="files whose factorial is already template-conditioned; used for all label sets")
a = ap.parse_args()
D, T = {}, {}
for f in a.main: D.update(json.load(open(f)))
for f in a.tpl: T.update(json.load(open(f)))
B = {}
for f in a.both: B.update(json.load(open(f)))
D_notpl = dict(D)
D.update(B); T.update({k: v for k, v in B.items() if k.startswith("factorial|")})
NICE = {"roberta-base": "RoBERTa-base", "roberta-large": "RoBERTa-large", "bert-base-uncased": "BERT-base", "answerdotai/ModernBERT-base": "ModernBERT-base",
        "gpt2": "GPT-2", "gpt2-medium": "GPT-2-medium", "EleutherAI/pythia-160m": "Pythia-160M", "EleutherAI/pythia-410m": "Pythia-410M",
        "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B", "Qwen/Qwen2.5-3B": "Qwen2.5-3B", "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B": "Qwen2.5-7B", "Qwen/Qwen2.5-14B": "Qwen2.5-14B", "meta-llama/Llama-3.1-8B": "Llama-3.1-8B"}
DSN = {"presence": "Topic", "stereo": "Stereotype", "factorial": "Factorial", "bios": "Bios"}
CONDS = ["original", "euclid_orth", "sigma_orth", "sigmaW_orth", "sigmaLW_orth", "shuffled_orth", "random_euclid", "random_sigma"]


def rows_from(src, keep):
    out = []
    for key, v in src.items():
        ds, model, l = key.split("|")
        for dn, kinds in (v.get("analysis") or {}).items():
            if dn not in keep: continue
            for kind, conds in kinds.items():
                for cn, m in conds.items():
                    pairs = sorted({k.split("_")[-1] for k in m if k.startswith("median_r_cond_") and "_z_" not in k})
                    for p in pairs:
                        g = lambda k: m[k]["mean"] if k in m else np.nan
                        out.append(dict(model=model, layer=int(l), dn=dn, kind=kind, cond=cn, pair=p,
                                        raw=g(f"median_excess_{p}"), ce=g(f"median_ce_{p}"), cep=g(f"median_ce_p_{p}"),
                                        rc=g(f"median_r_cond_{p}"), rcz=g(f"median_r_cond_z_{p}"), r=g(f"median_r_{p}"),
                                        auc=np.nanmean([g(k) for k in m if k.startswith("auc_")]),
                                        bshare=np.nanmean([g(k) for k in m if k.startswith("bshare_")]) if any(k.startswith("bshare_") for k in m) else np.nan))
    return out


R = rows_from(D, {"presence", "stereo", "bios"}) + rows_from(T if T else D, {"factorial"})
macros = {}
def M(k, v, f="{:.2f}"):
    macros[k] = f.format(v) if isinstance(v, (float, int, np.floating, np.integer)) else str(v)


def sel(dn=None, kind=None, cond=None, model=None):
    # random directions are stored under the first probe kind only (lr)
    return [r for r in R if (dn is None or r["dn"] == dn) and (kind is None or r["kind"] == kind or r["cond"].startswith("random"))
            and (cond is None or r["cond"] == cond) and (model is None or r["model"] == model)]


def mabs(rs, k):
    v = np.array([x[k] for x in rs], float); return float(np.nanmean(np.abs(v))) if len(v) else float("nan")


def permodel(dn, kind, cond, k="rc"):
    return {m: mabs(strict(dn, kind, cond, m), k) for m in NICE if strict(dn, kind, cond, m)}


def strict(dn, kind, cond, model=None):
    """rows for exactly this probe kind (random directions live under lr)"""
    k = "lr" if cond.startswith("random") else kind
    return [x for x in sel(dn, k, cond, model) if x["kind"] == k]


def mm(dn, kind, cond, key="rc", absval=True, fn=None):
    """mean over models of the per-model mean (each model weighted equally)"""
    vals = []
    for m in NICE:
        rs = strict(dn, kind, cond, m)
        if not rs: continue
        if fn is not None: v = fn(rs)
        else:
            a_ = np.array([x[key] for x in rs], float)
            v = np.nanmean(np.abs(a_)) if absval else np.nanmean(a_)
        vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def cep_share(rs):
    cp = np.array([x["cep"] for x in rs], float); cp = cp[np.isfinite(cp)]
    return 100 * float(np.mean(cp < 0.05)) if len(cp) else float("nan")


def boot_models(dn, kind, cond, k="rc", B=5000, seed=0):
    pm = np.array(list(permodel(dn, kind, cond, k).values()))
    rng = np.random.default_rng(seed)
    bs = [rng.choice(pm, len(pm)).mean() for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


models = [m for m in NICE if any(r["model"] == m for r in R)]
M("condNModels", len(models), "{:d}")
M("condModelList", ", ".join(NICE[m] for m in models))
tag = {"presence": "Top", "stereo": "Ste", "factorial": "Fac", "bios": "Bio"}
ctag = {"original": "Orig", "euclid_orth": "Eucl", "sigma_orth": "Sigma", "sigmaW_orth": "SigW", "sigmaLW_orth": "SigLW", "shuffled_orth": "Shuf", "random_euclid": "RandE", "random_sigma": "RandS"}
ktag = {"lr": "LR", "dom": "DoM"}
for dn in DSN:
    for kind in ("lr", "dom"):
        for cn in CONDS:
            if cn.startswith("random") and kind == "dom": continue
            if not strict(dn, kind, cn): continue
            t_ = f"{ktag[kind]}{ctag[cn]}{tag[dn]}"
            M(f"rc{t_}", mm(dn, kind, cn, "rc"))
            M(f"ce{t_}", mm(dn, kind, cn, "ce"), "{:.3f}")
            M(f"raw{t_}", mm(dn, kind, cn, "raw"), "{:.3f}")
            M(f"auc{t_}", mm(dn, kind, cn, "auc", absval=False))
            M(f"bsh{t_}", 100 * mm(dn, kind, cn, "bshare", absval=False), "{:.0f}")
            v_ = mm(dn, kind, cn, fn=cep_share)
            if np.isfinite(v_): M(f"cep{t_}", v_, "{:.0f}")

# label-induced share of topic co-activation for original LR probes: 1 - |CE|/|raw| per model
for kind in ("lr", "dom"):
    sh = []
    for m in models:
        rs = sel("presence", kind, "original", m)
        if rs: sh.append(1 - mabs(rs, "ce") / mabs(rs, "raw"))
    M(f"labShare{ktag[kind]}Lo", 100 * min(sh), "{:.0f}"); M(f"labShare{ktag[kind]}Hi", 100 * max(sh), "{:.0f}")

# label-induced share on stereotype labels (for honesty: labels explain far less there)
sh = [1 - mabs(strict("stereo", "lr", "original", m), "ce") / mabs(strict("stereo", "lr", "original", m), "raw") for m in models if strict("stereo", "lr", "original", m)]
M("labShareSteLRLo", 100 * min(sh), "{:.0f}"); M("labShareSteLRHi", 100 * max(sh), "{:.0f}")

# DoM headline: per-model counts and ranges
for dn in DSN:
    o, e, s = permodel(dn, "dom", "original"), permodel(dn, "dom", "euclid_orth"), permodel(dn, "dom", "sigma_orth")
    rnd = permodel(dn, "lr", "random_sigma")
    lr = permodel(dn, "lr", "original")
    M(f"nDomGtLR{tag[dn]}", sum(o[m] > lr[m] for m in o), "{:d}")
    M(f"nSigLtEucDom{tag[dn]}", sum(s[m] < e[m] for m in s), "{:d}")
    M(f"domOrigLo{tag[dn]}", min(o.values())); M(f"domOrigHi{tag[dn]}", max(o.values()))
    M(f"domSigLo{tag[dn]}", min(s.values())); M(f"domSigHi{tag[dn]}", max(s.values()))
    M(f"domEucLo{tag[dn]}", min(e.values())); M(f"domEucHi{tag[dn]}", max(e.values()))
    lo, hi = boot_models(dn, "dom", "original"); M(f"domOrigCILo{tag[dn]}", lo); M(f"domOrigCIHi{tag[dn]}", hi)
    # Euclidean removes what share of DoM conditional correlation, Sigma removes what share
    M(f"domEucRed{tag[dn]}", 100 * (1 - np.mean(list(e.values())) / np.mean(list(o.values()))), "{:.0f}")
    M(f"domSigRed{tag[dn]}", 100 * (1 - np.mean(list(s.values())) / np.mean(list(o.values()))), "{:.0f}")
eu_red = {dn: 100 * (1 - np.mean(list(permodel(dn, "dom", "euclid_orth").values())) / np.mean(list(permodel(dn, "dom", "original").values()))) for dn in DSN}
M("domEucRedLo", min(eu_red.values()), "{:.0f}"); M("domEucRedHi", max(eu_red.values()), "{:.0f}")
rnd_all = [mm(dn, "lr", "random_sigma") for dn in DSN]
cep_r = [mm(dn, "lr", "random_sigma", fn=cep_share) for dn in DSN]
M("cepRandLo", min(cep_r), "{:.0f}"); M("cepRandHi", max(cep_r), "{:.0f}")
M("rcRandSLo", min(rnd_all)); M("rcRandSHi", max(rnd_all))
dom_all = [mm(dn, "dom", "original") for dn in DSN]
M("rcDomOrigLo", min(dom_all)); M("rcDomOrigHi", max(dom_all))
# across all label sets x models
allpairs = [(dn, m) for dn in DSN for m in permodel(dn, "dom", "original")]
M("nDomCases", len(allpairs), "{:d}")
M("nDomSigBeatsEuc", sum(permodel(dn, "dom", "sigma_orth")[m] < permodel(dn, "dom", "euclid_orth")[m] for dn, m in allpairs), "{:d}")
M("nDomGtLR", sum(permodel(dn, "dom", "original")[m] > permodel(dn, "lr", "original")[m] for dn, m in allpairs), "{:d}")
# cosine between DoM and LR probe directions is not stored; AUC gap
M("aucGapDomLR", np.mean([mm(dn, "lr", "original", "auc", False) for dn in DSN]) - np.mean([mm(dn, "dom", "original", "auc", False) for dn in DSN]))

# factorial: unconditioned (labels only) vs template-conditioned, LR original CE p<.05 share
if T:
    Ru = rows_from(D_notpl, {"factorial"})  # the 4-model run without template cells
    for kind in ("lr", "dom"):
        pm_c, pm_r = [], []
        for m in NICE:
            rs = [r for r in Ru if r["kind"] == kind and r["cond"] == "original" and r["model"] == m]
            if rs: pm_c.append(cep_share(rs)); pm_r.append(mabs(rs, "rc"))
        M(f"cepFacNoTpl{ktag[kind]}", np.mean(pm_c), "{:.0f}")
        M(f"rcFacNoTpl{ktag[kind]}", np.mean(pm_r))
        # same models, with template cells
        ms4 = {r["model"] for r in Ru}
        pt_c = [cep_share(strict("factorial", kind, "original", m)) for m in NICE if m in ms4 and strict("factorial", kind, "original", m)]
        pt_r = [mabs(strict("factorial", kind, "original", m), "rc") for m in NICE if m in ms4 and strict("factorial", kind, "original", m)]
        M(f"cepFacTplSame{ktag[kind]}", np.mean(pt_c), "{:.0f}"); M(f"rcFacTplSame{ktag[kind]}", np.mean(pt_r))
        M("nFacTplCompare", len(pt_r), "{:d}")

# prior thresholds (each probe thresholded at its concept's label prior): |CE| for original probes
def prior_ce(src, dn, kind):
    v = []
    for key, val in src.items():
        an = (val.get("analysis") or {}).get(dn, {}).get(kind, {}).get("original")
        if not an: continue
        v += [abs(an[k]["mean"]) for k in an if k.startswith("prior_ce_") and not k.startswith("prior_ce_z") and not k.startswith("prior_ce_p") and not k.startswith("prior_ce_score")]
    return float(np.nanmean(v)) if v else float("nan")
for dn in ("presence", "stereo", "bios"):
    for kind in ("lr", "dom"):
        M(f"cePrior{ktag[kind]}{tag[dn]}", prior_ce(D, dn, kind), "{:.3f}")
if T:
    for kind in ("lr", "dom"): M(f"cePrior{ktag[kind]}Fac", prior_ce(T, "factorial", kind), "{:.3f}")

# table: mean |r_cond| (held-out AUC), model-level averages
COLS = [("lr", "original"), ("lr", "euclid_orth"), ("lr", "sigma_orth"), ("dom", "original"), ("dom", "euclid_orth"), ("dom", "sigma_orth"), ("lr", "random_sigma")]
L = [r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{4pt}",
     r"\caption{Conditional entanglement $|r_{\text{cond}}|$ and, in parentheses, held-out AUC.}",
     r"\label{tab:cond}", r"\begin{tabular}{lccccccc}", r"\toprule",
     r" & \multicolumn{3}{c}{Logistic-regression probes} & \multicolumn{3}{c}{Difference-of-means CAVs} & Random \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}",
     r"Label set & Original & Euclid. & $\Sigma$ & Original & Euclid. & $\Sigma$ & $\Sigma$-orthon. \\", r"\midrule"]
for dn in DSN:
    cells = []
    for kind, cn in COLS:
        v, au = mm(dn, kind, cn, "rc"), mm(dn, kind, cn, "auc", False)
        txt = f"{v:.2f}" if cn.startswith("random") else f"{v:.2f} ({au:.2f})"
        if kind == "dom" and cn == "original": txt = r"\textbf{" + f"{v:.2f}" + "}" + f" ({au:.2f})"
        cells.append(txt)
    L.append(f"{DSN[dn]} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
os.makedirs(f"{a.out}/tables", exist_ok=True)
open(f"{a.out}/tables/tab_cond.tex", "w").write("\n".join(L))

# per-model DoM table for appendix
L = [r"\begin{table*}[t]", r"\centering\small\setlength{\tabcolsep}{5pt}",
     r"\caption{DoM $|r_{\text{cond}}|$ per model: original\,/\,Euclidean\,/\,$\Sigma$.}",
     r"\label{tab:dommodel}", r"\begin{tabular}{l" + "c" * len(DSN) + "}", r"\toprule",
     "Model & " + " & ".join(DSN.values()) + r" \\", r"\midrule"]
for m in models:
    c = []
    for dn in DSN:
        o, e, s = permodel(dn, "dom", "original"), permodel(dn, "dom", "euclid_orth"), permodel(dn, "dom", "sigma_orth")
        c.append(f"{o[m]:.2f}/{e[m]:.2f}/{s[m]:.2f}" if m in o else "--")
    L.append(f"{NICE[m]} & " + " & ".join(c) + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
open(f"{a.out}/tables/tab_dommodel.tex", "w").write("\n".join(L))

# ---- uncertainty: bootstrap over models, seed SD, sign tests
from math import comb
def boot_diff(dn, kind, c1, c2, B=5000, seed=1):
    a1, a2 = permodel(dn, kind, c1), permodel(dn, kind, c2)
    ms_ = [m for m in a1 if m in a2]; d_ = np.array([a1[m] - a2[m] for m in ms_])
    r_ = np.random.default_rng(seed); bs_ = [r_.choice(d_, len(d_)).mean() for _ in range(B)]
    return float(d_.mean()), float(np.percentile(bs_, 2.5)), float(np.percentile(bs_, 97.5))
def boot_mean(dn, kind, cn, B=5000, seed=2):
    v_ = np.array(list(permodel(dn, kind, cn).values())); r_ = np.random.default_rng(seed)
    bs_ = [r_.choice(v_, len(v_)).mean() for _ in range(B)]
    return float(v_.mean()), float(np.percentile(bs_, 2.5)), float(np.percentile(bs_, 97.5))
L = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{3pt}",
     r"\caption{Means over models with bootstrap 95\% intervals (resampling models).}", r"\label{tab:ci}",
     r"\begin{adjustbox}{max width=\columnwidth}", r"\begin{tabular}{lccc}", r"\toprule",
     r"Label set & DoM $|r_{\text{cond}}|$ & DoM$-$LR & DoM: Eucl.$-\Sigma$ \\", r"\midrule"]
allexcl = True
for dn in DSN:
    m1 = boot_mean(dn, "dom", "original")
    d1 = boot_diff_kind = None
    a1, a2 = permodel(dn, "dom", "original"), permodel(dn, "lr", "original")
    ms_ = [m for m in a1 if m in a2]; dd = np.array([a1[m] - a2[m] for m in ms_]); r_ = np.random.default_rng(3)
    bs_ = [r_.choice(dd, len(dd)).mean() for _ in range(5000)]; d1 = (dd.mean(), np.percentile(bs_, 2.5), np.percentile(bs_, 97.5))
    d2 = boot_diff(dn, "dom", "euclid_orth", "sigma_orth")
    allexcl &= m1[1] > 0 and d1[1] > 0 and d2[1] > 0
    f = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    L.append(f"{DSN[dn]} & {f(m1)} & {f(d1)} & {f(d2)}" + r" \\")
L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
open(f"{a.out}/tables/tab_ci.tex", "w").write("\n".join(L))
M("ciAllExclude", "every" if allexcl else "not every")
n_ = len({m for _, m in allpairs})    # models as the unit: a model counts if Sigma beats Euclidean in all its label sets
mods_ = sorted({m for _, m in allpairs})
nwin_ = sum(all(permodel(dn, "dom", "sigma_orth")[m] < permodel(dn, "dom", "euclid_orth")[m] for dn, mm_ in allpairs if mm_ == m) for m in mods_)
M("nModelsSigWin", nwin_, "{:d}"); M("nModelsSign", n_, "{:d}")
from math import comb
pv_ = min(1.0, 2 * sum(comb(n_, k) for k in range(nwin_, n_ + 1)) / 2 ** n_)
M("signP", f"{pv_:.0e}".replace("e-0", "e-").replace("e-", r"\times10^{-") + "}")
sd_ = [x for k, v in D.items() if k.startswith(("bios|", "presence|", "stereo|", "factorial|")) for dn_, kd in (v.get("analysis") or {}).items()
       for x in [kd.get("dom", {}).get("original", {}).get(f"median_r_cond_{p}", {}).get("std") for p in ("G-P", "G-R", "R-P")] if x is not None]
M("seedSD", float(np.nanmean(sd_)) if sd_ else float("nan"))

# ---- case study: gender vs profession group on Bias in Bios (labels independent by design)
def bios_case(src):
    out = {}
    for m in NICE:
        ks = [k for k in src if k.startswith(f"bios|{m}|")]
        if not ks: continue
        for kind, cn in (("dom", "original"), ("dom", "euclid_orth"), ("dom", "sigma_orth"), ("dom", "filter_inv"), ("lr", "original")):
            if cn not in src[ks[0]]["analysis"]["bios"][kind]: continue
            g = lambda key: float(np.mean([src[k]["analysis"]["bios"][kind][cn][key]["mean"] for k in ks if key in src[k]["analysis"]["bios"][kind][cn]]))
            per = np.array([src[k]["analysis"]["bios"][kind][cn]["median_r_cond_G-P"]["mean"] for k in ks])
            pj = np.array([src[k]["analysis"]["bios"][kind][cn].get("proj_G-P", {"mean": np.nan})["mean"] for k in ks])
            out[(m, kind, cn)] = dict(proj=float(np.nanmean(np.abs(pj))), cos=g("cos_G-P"), rc=g("median_r_cond_G-P"), arc=float(np.mean(np.abs(per))),
                                      npos=int(np.sum(per > 0.1)), nneg=int(np.sum(per < -0.1)), nl=len(per),
                                      joint=g("median_joint_G-P"), auc=(g("auc_gender") + g("auc_profession")) / 2)
    return out
BIOS = {}
for f in a.bios: BIOS.update(json.load(open(f)))
BC = bios_case(BIOS if BIOS else D)
if BC:
    bm = [m for m in NICE if (m, "dom", "original") in BC]
    COLS_ = [("dom", "original", "DoM"), ("dom", "euclid_orth", "Eucl."), ("dom", "sigma_orth", r"$\Sigma$"), ("dom", "filter_inv", r"$\Sigma^{-1}$"), ("lr", "original", "LR")]
    COLS_ = [c for c in COLS_ if all((m, c[0], c[1]) in BC for m in bm)]
    L = [r"\begin{table}[t]", r"\centering\small\setlength{\tabcolsep}{2.5pt}",
         r"\caption{Bias in Bios: mean over layers of $|r_{\text{cond}}|$ (top) and of the projection-audit gap (bottom, SD units).}",
         r"\label{tab:case}", r"\begin{adjustbox}{max width=\columnwidth}", r"\begin{tabular}{l" + "c" * len(COLS_) + "}", r"\toprule",
         r" & \multicolumn{%d}{c}{Difference of means} & \\" % (len(COLS_) - 1), r"\cmidrule(lr){2-%d}" % len(COLS_),
         "Model & " + " & ".join(c[2] for c in COLS_) + r" \\", r"\midrule"]
    for m in bm:
        L.append(NICE[m] + " & " + " & ".join(f"{BC[(m, k, c)]['arc']:.2f}" for k, c, _ in COLS_) + r" \\")
    L += [r"\midrule", r"\multicolumn{%d}{l}{\emph{Projection-audit gap (mean over models)}} \\" % (len(COLS_) + 1),
          " & " + " & ".join(f"{np.mean([BC[(m, k, c)]['proj'] for m in bm]):.2f}" for k, c, _ in COLS_) + r" \\",
          "AUC & " + " & ".join(f"{np.mean([BC[(m, k, c)]['auc'] for m in bm]):.2f}" for k, c, _ in COLS_) + r" \\"]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    projm = {c: np.mean([BC[(m, "dom", c)]['proj'] for m in bm]) for c in ("original", "euclid_orth", "sigma_orth", "filter_inv") if all((m, "dom", c) in BC for m in bm)}
    projlr = np.mean([BC[(m, "lr", "original")]['proj'] for m in bm])
    for c_, nm_ in (("original", "Orig"), ("euclid_orth", "Eucl"), ("sigma_orth", "Sigma"), ("filter_inv", "Inv")):
        if c_ in projm: M(f"projDoM{nm_}", projm[c_])
    M("projLR", projlr); M("projRatio", projm["original"] / projlr, "{:.1f}")
    if "filter_inv" in projm:
        M("aucInv", np.mean([BC[(m, "dom", "filter_inv")]['auc'] for m in bm]))
        M("nInvBelowDoM", sum(BC[(m, "dom", "filter_inv")]['proj'] < BC[(m, "dom", "original")]['proj'] for m in bm), "{:d}")
    # random floor
    rf = [abs(BIOS[k]["analysis"]["bios"]["lr"]["random_sigma"].get("proj_G-P", {"mean": np.nan})["mean"]) for k in BIOS if k.startswith("bios|") and "random_sigma" in BIOS[k]["analysis"]["bios"]["lr"]]
    if rf: M("projFloor", float(np.nanmean(rf)))
    open(f"{a.out}/tables/tab_case.tex", "w").write("\n".join(L))
    json.dump({NICE[m]: {f"{kind}|{cn}": BC[(m, kind, cn)] for (mm_, kind, cn) in BC if mm_ == m} for m in bm},
              open(f"{a.out}/case_bios.json", "w"), indent=1)
    dom = {m: BC[(m, "dom", "original")] for m in bm}
    M("caseAbsLo", min(dom[m]["arc"] for m in bm)); M("caseAbsHi", max(dom[m]["arc"] for m in bm))
    M("caseNFlip", sum(dom[m]["npos"] > 0 and dom[m]["nneg"] > 0 for m in bm), "{:d}")
    M("caseLRAbsMax", max(BC[(m, "lr", "original")]["arc"] for m in bm))
    M("caseSigAbsMax", max(BC[(m, "dom", "sigma_orth")]["arc"] for m in bm))
    M("caseEuAbsMin", min(BC[(m, "dom", "euclid_orth")]["arc"] for m in bm))
    M("caseEuAbsGe", sum(BC[(m, "dom", "euclid_orth")]["arc"] >= 0.2 for m in bm), "{:d}")
    strong = [m for m in bm if abs(dom[m]["rc"]) >= 0.2]
    M("caseNModels", len(bm), "{:d}"); M("caseNStrong", len(strong), "{:d}")
    M("caseNStrongPos", sum(dom[m]["rc"] > 0 for m in strong), "{:d}"); M("caseNStrongNeg", sum(dom[m]["rc"] < 0 for m in strong), "{:d}")
    M("caseStrongLo", min(abs(dom[m]["rc"]) for m in strong)); M("caseStrongHi", max(abs(dom[m]["rc"]) for m in strong))
    weak = [m for m in bm if m not in strong]
    M("caseWeakModels", ", ".join(NICE[m] for m in weak) if weak else "none"); M("caseWeakMax", max([abs(dom[m]["rc"]) for m in weak] or [0]))
    M("caseEuKeep", sum(abs(BC[(m, "dom", "euclid_orth")]["rc"]) >= 0.2 for m in strong), "{:d}")
    pos = [m for m in bm if dom[m]["rc"] > 0]; neg = [m for m in bm if dom[m]["rc"] < 0]
    M("casePosModels", " and ".join(NICE[m] for m in pos)); M("caseNegModels", " and ".join(NICE[m] for m in neg))
    M("casePosLo", min(dom[m]["rc"] for m in pos), "{:+.2f}"); M("casePosHi", max(dom[m]["rc"] for m in pos), "{:+.2f}")
    M("caseNegLo", min(dom[m]["rc"] for m in neg), "{:+.2f}"); M("caseNegHi", max(dom[m]["rc"] for m in neg), "{:+.2f}")
    M("caseLRMaxAbs", max(abs(BC[(m, "lr", "original")]["rc"]) for m in bm))
    M("caseSigMaxAbs", max(abs(BC[(m, "dom", "sigma_orth")]["rc"]) for m in bm))
    M("caseEuMinAbs", min(abs(BC[(m, "dom", "euclid_orth")]["rc"]) for m in bm))
    M("caseAucLR", np.mean([BC[(m, "lr", "original")]["auc"] for m in bm])); M("caseAucDoM", np.mean([dom[m]["auc"] for m in bm]))
    jm = min(bm, key=lambda m: dom[m]["joint"]); jM = max(bm, key=lambda m: dom[m]["joint"])
    M("caseJointLoModel", NICE[jm]); M("caseJointLo", 100 * dom[jm]["joint"], "{:.0f}")
    M("caseJointHiModel", NICE[jM]); M("caseJointHi", 100 * dom[jM]["joint"], "{:.0f}")
    cs = max(bm, key=lambda m: BC[(m, "dom", "sigma_orth")]["cos"])
    M("caseCosSigModel", NICE[cs]); M("caseCosSig", BC[(cs, "dom", "sigma_orth")]["cos"], "{:+.2f}")
    M("caseCosSigRc", BC[(cs, "dom", "sigma_orth")]["rc"], "{:+.2f}")
    M("caseCosOrigLo", min(dom[m]["cos"] for m in bm), "{:+.2f}"); M("caseCosOrigHi", max(dom[m]["cos"] for m in bm), "{:+.2f}")

# ---- controls (present only in runs with the Sigma_W / Ledoit-Wolf conditions) and text macros
TXT = {}
has_w = any(r["cond"] == "sigmaW_orth" for r in R)
if has_w:
    def f2(x): return f"{x:.2f}"
    # controls are compared on the same models (those run with the control conditions)
    def msub(dn, cond, key="rc", absval=True):
        pw = set(permodel(dn, "dom", "sigmaW_orth"))
        vals = []
        for m in pw:
            a_ = np.array([x[key] for x in strict(dn, "dom", cond, m)], float)
            if len(a_): vals.append(np.nanmean(np.abs(a_)) if absval else np.nanmean(a_))
        return float(np.mean(vals)) if vals else float("nan")
    bs = {dn: 100 * msub(dn, "original", "bshare", absval=False) for dn in DSN}
    sw = {dn: msub(dn, "sigmaW_orth") for dn in DSN}
    sg = {dn: msub(dn, "sigma_orth") for dn in DSN}
    lw = {dn: msub(dn, "sigmaLW_orth") for dn in DSN}
    awd = {dn: msub(dn, "sigmaW_orth", "auc", False) for dn in DSN}
    pairs_lw = [(dn, m) for dn in DSN for m in permodel(dn, "dom", "sigmaLW_orth")]
    nlw = sum(abs(permodel(dn, "dom", "sigmaLW_orth")[m] - permodel(dn, "dom", "sigma_orth")[m]) < 0.02 for dn, m in pairs_lw)
    ncase = len(pairs_lw)
    M("nCtrlModels", len({m for _, m in pairs_lw}), "{:d}")
    aus = {dn: msub(dn, "sigma_orth", "auc", False) for dn in DSN}
    TXT["sigmaWText"] = (
        "Between label cells lies "
        + f"{bs['presence']:.0f}\\% of the held-out score variance of DoM vectors on topic labels, {bs['stereo']:.0f}\\% on stereotype, "
        + f"{bs['bios']:.0f}\\% on Bios and {bs['factorial']:.0f}\\% on factorial labels, where every template forms its own cells. "
        + "Where this share is small, zero total correlation leaves little room for within-cell correlation, so $\\Sigma$-orthogonalization "
        + "is close to a within-cell decorrelation by construction; on topic labels it is not, yet $\\Sigma$ still removes "
        + f"\\domSigRedTop\\% of the entanglement. The direct control orthogonalizes in the within-cell covariance $\\Sigma_W$, which zeroes "
        + "within-cell correlation on training data by construction. On held-out data it gives DoM $|\\rcond|$ of "
        + ", ".join(f2(sw[d]) for d in DSN) + " on topic, stereotype, factorial and Bios labels (\\nCtrlModels{} models; $\\Sigma$: "
        + ", ".join(f2(sg[d]) for d in DSN) + "), reaching the random baseline on the factorial set where $\\Sigma$ does not, "
        + "with AUC unchanged (" + ", ".join(f2(awd[d]) for d in DSN) + "). When concept labels are available, $\\Sigma_W$ is therefore the "
        + "better metric for conditional separation. Replacing the sample covariance by a Ledoit--Wolf shrinkage estimate changes DoM "
        + f"$|\\rcond|$ after $\\Sigma$-orthogonalization by less than 0.02 in {nlw} of {ncase} model--label-set cases, so the result does not "
        + "hinge on covariance estimation with a few thousand sentences in up to 5{,}120 dimensions.")
    M("nLWclose", nlw, "{:d}"); M("nLWcases", ncase, "{:d}")
    for dn in DSN:
        M(f"bshDoM{tag[dn]}", bs[dn], "{:.0f}"); M(f"rcDoMSigW{tag[dn]}", sw[dn]); M(f"rcDoMSigLW{tag[dn]}", lw[dn]); M(f"aucDoMSigW{tag[dn]}", awd[dn])
nm = len(models)
TXT["condModelsText"] = (f"The conditional analyses use {nm} models." if nm < 11 else "") + " The behavioral test uses instruction-tuned Qwen models."
TXT["coverageCondModels"] = ", ".join(NICE[m] for m in models) if nm <= 4 else f"all {nm}"
TXT["coverageCondLayers"] = "every 2nd--4th (7B, 14B: 6 spaced), 3" if nm > 4 else "every 2nd (Qwen: 4th), 3"

with open(f"{a.out}/numbers_cond.tex", "w") as f:
    for k, v in TXT.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    f.write("% AUTO-GENERATED by gen_cond.py\n")
    for k, v in macros.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in macros.items()))

# figure: |r_cond| by label set, LR vs DoM, original / Euclidean / Sigma, per-model dots, random baseline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "legend.frameon": False})
COL = {"original": "#0072B2", "euclid_orth": "#D55E00", "sigma_orth": "#009E73"}
NM = {"original": "Original", "euclid_orth": "Euclidean-orth.", "sigma_orth": r"$\Sigma$-orth."}
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.05), sharey=True)
dsl = list(DSN)
for ax, kind, title in zip(axes, ("lr", "dom"), ("Logistic-regression probes", "Difference-of-means CAVs")):
    for j, cn in enumerate(("original", "euclid_orth", "sigma_orth")):
        xs = np.arange(len(dsl)) + (j - 1) * 0.26
        pm = [permodel(dn, kind, cn) for dn in dsl]
        ax.bar(xs, [np.mean(list(p.values())) for p in pm], width=0.24, color=COL[cn], label=NM[cn], zorder=2)
        for x, p in zip(xs, pm):
            ax.scatter([x] * len(p), list(p.values()), s=7, color="#1f1f1e", zorder=3, linewidths=0)
    rnd = [np.mean(list(permodel(dn, "lr", "random_sigma").values())) for dn in dsl]
    for i, v in enumerate(rnd):
        ax.plot([i - 0.42, i + 0.42], [v, v], color="#1f1f1e", lw=0.9, ls=(0, (2, 1.5)), zorder=4,
                label=r"Random $\Sigma$-orthon." if i == 0 else None)
    ax.set_xticks(range(len(dsl))); ax.set_xticklabels([{"stereo": "Stereo."}.get(d, DSN[d]) for d in dsl], fontsize=7.5); ax.set_title(title, loc="left", fontsize=8)
axes[0].set_ylabel(r"$|r_{\mathrm{cond}}|$")
axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7)
fig.tight_layout(); fig.savefig(f"{a.out}/fig_cond.pdf"); fig.savefig(f"{a.out}/fig_cond.png", dpi=200)
