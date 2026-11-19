#!/usr/bin/env python
"""
CAR audit v2 (Tiers A+B): separates label-induced from representation-induced co-activation.

New relative to car_audit.py
  * Conditional measure entanglement (CE): co-activation measured WITHIN label cells,
        CE = P(A and B) - sum_c w_c P(A|c) P(B|c),
    i.e. raw co-activation minus what the label structure alone implies. CE = 0 when, given the labels,
    the two probes' memberships are independent. Also the partial correlation of scores given the cells.
  * Label-preserving permutation null: B's memberships are permuted within label cells (n_perm draws),
    giving the null SD of CE and a two-sided p-value per case.
  * Probe types: logistic regression ("lr") and difference of means ("dom", the common CAV recipe).
  * Conditions: original; Euclidean and Sigma Gram-Schmidt averaged over ALL orders; random directions
    orthonormal in the Euclidean metric and in the Sigma metric; shuffled-label probes (Euclidean GS).
  * Thresholds: median (50/50) and prior (each probe flags the fraction of the pool that is truly positive).
  * Datasets: StereoSet (cells = bias type x stereotype), factorial templates (cells = gender x race x
    profession), Bias in Bios rebalanced so gender and profession group are independent
    (cells = gender x profession group).

Usage:
  python car_audit2.py --models roberta-base gpt2 EleutherAI/pythia-160m --layers stride2 --seeds 3
  python car_audit2.py --synthetic        # pipeline self-test, no GPU, no downloads
"""
import argparse, itertools, json, math, time, urllib.request
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

TEMPLATE_CELLS = False   # set by --template-cells: condition factorial cells on template as well as labels
STEREOSET_URL = "https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json"
DECODER_TYPES = {"gpt2", "gpt_neox", "gpt_neo", "llama", "mistral", "qwen2", "gemma", "gemma2", "olmo", "olmo2", "phi", "phi3"}
FACT_GENDER = [["man", "gentleman", "father", "husband", "boy", "son"], ["woman", "lady", "mother", "wife", "girl", "daughter"]]
FACT_RACE = [["white", "European American", "Caucasian"], ["Black", "African American", "dark-skinned"]]
FACT_PROF = [["engineer", "mechanic", "carpenter", "pilot", "surgeon", "firefighter", "plumber", "electrician"],
             ["nurse", "secretary", "receptionist", "hairdresser", "librarian", "dancer", "babysitter", "florist"]]
FACT_TEMPLATES = [
    "The {r} {g} worked as a {p}.", "My neighbour is a {r} {g} who is a {p}.", "A {r} {g} applied for the {p} job yesterday.",
    "Everyone agreed that the {r} {g} was a good {p}.", "The {r} {g} has been a {p} for ten years.",
    "I met a {r} {g} at the conference, and the person was a {p}.",
    "The {p} we hired is a {r} {g}.", "Our new {p} is a {r} {g} from the city.", "The {r} {g} trained for years to become a {p}.",
    "Last week a {r} {g} started work as a {p} downtown.", "The {r} {g} next door retired after a long career as a {p}.",
    "According to the article, the {r} {g} is the best {p} in town.", "A {r} {g} answered the phone and said they were the {p}.",
    "The {r} {g} smiled; being a {p} was a dream come true.", "When the {p} arrived, it turned out to be a {r} {g}.",
    "The local paper interviewed a {r} {g} who works as a {p}.", "She introduced me to a {r} {g}, a {p} by profession.",
    "The {r} {g} spent the morning working as a {p}.", "Nobody was surprised that the {r} {g} became a {p}.",
    "The {r} {g} is proud of being a {p}.", "A {r} {g} was promoted to senior {p} this year.",
    "The {r} {g} explained what a {p} does every day.", "Yesterday the {r} {g} quit being a {p}.", "The {r} {g} wants to be a {p}.",
]


# ================================================================ datasets
# Each dataset returns rows (dicts with group, text, cell, and per-concept labels) and a list of
# "definitions": (name, concepts, label_fn) where label_fn(rows, idx, concept, rng) -> (sel_idx, y).
def stereoset_rows():
    raw = json.loads(urllib.request.urlopen(STEREOSET_URL, timeout=60).read())["data"]
    rows = []
    for split in ("intrasentence", "intersentence"):
        for ex in raw[split]:
            for s in ex["sentences"]:
                if s["gold_label"] in ("stereotype", "anti-stereotype"):
                    st = int(s["gold_label"] == "stereotype")
                    t = s["sentence"] if split == "intrasentence" else f'{ex["context"]} {s["sentence"]}'
                    rows.append(dict(group=f'{split}:{ex["id"]}', text=t, bias=ex["bias_type"], stereo=st,
                                     cell=f'{ex["bias_type"]}|{st}'))
    return rows


def lab_topic(rows, idx, c, rng):
    pos = [i for i in idx if rows[i]["bias"] == c]; neg = [i for i in idx if rows[i]["bias"] != c]
    neg = list(rng.choice(neg, min(len(neg), len(pos)), replace=False))
    return np.array(pos + neg), np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(int)


def lab_stereo(rows, idx, c, rng):
    sel = np.array([i for i in idx if rows[i]["bias"] == c]); return sel, np.array([rows[i]["stereo"] for i in sel])


def lab_field(rows, idx, c, rng):
    idx = np.asarray(idx); return idx, np.array([rows[i][c] for i in idx])


def truth_topic(rows, c): return np.array([int(r["bias"] == c) for r in rows])
def truth_field(rows, c): return np.array([r[c] for r in rows])
def truth_stereo(rows, c): return np.array([int(r["bias"] == c and r["stereo"] == 1) for r in rows])


def factorial_rows(seed=0, per_cell=16):
    rng = np.random.default_rng(seed); rows = []
    for t_id, t in enumerate(FACT_TEMPLATES):
        for g, r, p in itertools.product((0, 1), (0, 1), (0, 1)):
            for _ in range(per_cell):
                rows.append(dict(group=f"t{t_id}", gender=g, race=r, profession=p, cell=(f"t{t_id}|{g}{r}{p}" if TEMPLATE_CELLS else f"{g}{r}{p}"),
                                 text=t.format(g=rng.choice(FACT_GENDER[g]), r=rng.choice(FACT_RACE[r]), p=rng.choice(FACT_PROF[p]))))
    return rows


def bios_rows(per_cell=1000, seed=0):
    """Bias in Bios, rebalanced: equal counts in gender x profession-group cells -> independent labels."""
    from datasets import load_dataset
    ds = load_dataset("LabHC/bias_in_bios", split="test")
    prof, gen, text = np.array(ds["profession"]), np.array(ds["gender"]), ds["hard_text"]
    female_share = {p: gen[prof == p].mean() for p in np.unique(prof)}       # gender coding per dataset card
    group = np.array([int(female_share[p] > 0.5) for p in prof])              # 1 = female-majority profession
    rng = np.random.default_rng(seed); rows = []
    for g in (0, 1):
        for q in (0, 1):
            idx = np.where((gen == g) & (group == q))[0]
            for i in rng.choice(idx, min(per_cell, len(idx)), replace=False):
                rows.append(dict(group=f"b{i}", gender=int(g), profession=int(q), cell=f"{g}{q}", text=text[i][:1000]))
    return rows


def synthetic_rows(kind):
    if kind == "stereoset":
        b = ["gender", "race", "profession", "religion"]
        return [dict(group=f"g{i // 2}", text="", bias=b[(i // 2) % 4], stereo=i % 2, cell=f"{b[(i // 2) % 4]}|{i % 2}") for i in range(2400)]
    if kind == "factorial":
        return factorial_rows()
    rng = np.random.default_rng(1)
    return [dict(group=f"b{i}", gender=int(i % 2), profession=int((i // 2) % 2), cell=f"{i % 2}{(i // 2) % 2}", text="") for i in range(2000)]


DATASETS = {
    "stereoset": dict(load=stereoset_rows, concepts=["gender", "race", "profession"],
                      defs=[("presence", lab_topic, truth_topic), ("stereo", lab_stereo, truth_stereo)]),
    "factorial": dict(load=factorial_rows, concepts=["gender", "race", "profession"],
                      defs=[("factorial", lab_field, truth_field)]),
    "bios": dict(load=bios_rows, concepts=["gender", "profession"], defs=[("bios", lab_field, truth_field)]),
}


def split_rows(rows, seed=42, test_frac=0.3):
    groups = sorted({r["group"] for r in rows}); rng = np.random.default_rng(seed)
    test = set(rng.choice(groups, int(test_frac * len(groups)), replace=False))
    return np.array([r["group"] in test for r in rows])


# ================================================================ activations
def extract(model_name, texts, bs=32, max_len=128, half=False):
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(model_name, dtype=(torch.bfloat16 if half else torch.float32)).to(dev).eval()
    is_dec = model.config.model_type in DECODER_TYPES
    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(dev)
            hs = model(**b, output_hidden_states=True).hidden_states
            pos = (b["attention_mask"].sum(1) - 1) if is_dec else torch.zeros(len(b["input_ids"]), dtype=torch.long, device=dev)
            ar = torch.arange(len(pos), device=dev)
            chunks.append(torch.stack([h[ar, pos] for h in hs], 1).float().cpu().numpy())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(chunks), ("decoder" if is_dec else "encoder")


# ================================================================ probes and geometry
def fit_probe(X, y, seed, kind, sub=0.8):
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(y), max(10, int(sub * len(y))), replace=False)
    Xk, yk = X[keep], y[keep]
    if kind == "dom":
        return (Xk[yk == 1].mean(0) - Xk[yk == 0].mean(0)).astype(np.float64)
    return LogisticRegression(C=100, max_iter=1000, tol=1e-3).fit(Xk, yk).coef_[0].astype(np.float64)


def gram_schmidt(ws, M=None):
    ip = (lambda a, b: a @ b) if M is None else (lambda a, b: a @ M @ b)
    out = []
    for w in ws:
        v = w.copy()
        for u in out:
            uu = ip(u, u)
            if uu > 1e-12:
                v = v - ip(v, u) / uu * u
        out.append(v)
    return out


def membership(s, frac):
    """Top `frac` of the pool by rank (frac=0.5 -> exact median split)."""
    m = np.zeros(len(s), bool); k = int(round(frac * len(s)))
    if k > 0:
        m[np.argsort(s, kind="stable")[-k:]] = True
    return m


class Cells:
    def __init__(self, cell_labels):
        self.keys, self.inv = np.unique(cell_labels, return_inverse=True)
        self.idx = [np.where(self.inv == k)[0] for k in range(len(self.keys))]
        self.w = np.array([len(i) for i in self.idx]) / len(cell_labels)

    def expected_joint(self, A, B):
        return float(sum(w * A[i].mean() * B[i].mean() for w, i in zip(self.w, self.idx) if len(i)))

    def partial_corr(self, sa, sb):
        ra, rb = sa.astype(float).copy(), sb.astype(float).copy()
        for i in self.idx:
            ra[i] -= ra[i].mean(); rb[i] -= rb[i].mean()
        d = np.sqrt((ra @ ra) * (rb @ rb))
        return float(ra @ rb / d) if d > 0 else float("nan")

    def perm_null(self, A, B, rng, n_perm):
        """Joint rate when B is permuted within cells (label structure kept, conditional dependence destroyed)."""
        out = np.empty(n_perm)
        for t in range(n_perm):
            Bp = B.copy()
            for i in self.idx:
                Bp[i] = B[rng.permutation(i)]
            out[t] = np.mean(A & Bp)
        return out


def pair_metrics(sa, sb, frac_a, frac_b, cells, rng, n_perm):
    """Membership-level: raw excess, label-expected joint, CE (+ permutation null if n_perm>0).
    Score-level (primary conditional measure): partial correlation of scores given label cells, with its
    Fisher z statistic, and its orthant-law translation into a conditional co-activation excess."""
    A, B = membership(sa, frac_a), membership(sb, frac_b)
    joint = float(np.mean(A & B)); lab = cells.expected_joint(A, B); ce = joint - lab
    if n_perm > 0:
        null = cells.perm_null(A, B, rng, n_perm); sd = float(null.std())
        p = float((np.sum(np.abs(null - null.mean()) >= abs(joint - null.mean()) - 1e-12) + 1) / (n_perm + 1))
        z = ce / sd if sd > 0 else float("nan")
    else:
        sd = p = z = float("nan")
    r = float(np.corrcoef(sa, sb)[0, 1]) if np.std(sa) > 0 and np.std(sb) > 0 else float("nan")
    rc = cells.partial_corr(sa, sb)
    dof = max(len(sa) - len(cells.idx) - 1, 1)
    rcz = float(np.arctanh(np.clip(rc, -0.999999, 0.999999)) * math.sqrt(dof)) if np.isfinite(rc) else float("nan")
    return dict(joint=joint, indep=frac_a * frac_b, excess=joint - frac_a * frac_b, label_expected=lab, ce=ce,
                ce_z=z, ce_p=p, null_sd=sd, r=r, r_cond=rc, r_cond_z=rcz,
                ce_score=(math.asin(np.clip(rc, -1, 1)) / (2 * math.pi)) if np.isfinite(rc) else float("nan"),
                orthant=(0.25 + math.asin(np.clip(r, -1, 1)) / (2 * math.pi)) if np.isfinite(r) else float("nan"))


# ================================================================ per-layer analysis
def analyse_layer(X, rows, tr, te, concepts, defs, seeds, n_perm=200, n_random=5, probe_kinds=("lr", "dom")):
    tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
    mu, sd = X[tr_idx].mean(0), X[tr_idx].std() + 1e-8
    Xtr, Xe = (X[tr_idx] - mu) / sd, (X[te_idx] - mu) / sd
    loc = {g: k for k, g in enumerate(tr_idx)}
    Sigma = np.cov(Xtr, rowvar=False) + 1e-4 * np.eye(Xtr.shape[1])
    # Ledoit-Wolf shrinkage covariance (sensitivity to covariance estimation, n ~ d regime)
    from sklearn.covariance import LedoitWolf
    Sigma_lw = LedoitWolf(assume_centered=False).fit(Xtr).covariance_
    # within-cell (pooled) training covariance: Sigma_W-orthogonalization zeroes within-cell score correlation
    # on training data by construction, the control for "Sigma-orthogonalization works by construction"
    tr_cells = np.array([rows[i]["cell"] for i in tr_idx])
    Xw = Xtr.copy()
    for cval in np.unique(tr_cells):
        m_ = tr_cells == cval
        Xw[m_] -= Xw[m_].mean(0)
    Sigma_w = (Xw.T @ Xw) / max(len(Xw) - len(np.unique(tr_cells)), 1) + 1e-4 * np.eye(Xtr.shape[1])
    ev_, evec_ = np.linalg.eigh(Sigma)
    ev_ = np.clip(ev_, ev_.max() * 1e-8, None)
    S_half = (evec_ * np.sqrt(ev_)) @ evec_.T; Sinv_half = (evec_ / np.sqrt(ev_)) @ evec_.T; Sinv = (evec_ / ev_) @ evec_.T
    cells = Cells([rows[i]["cell"] for i in te_idx])
    pairs = list(itertools.combinations(concepts, 2))
    orders = list(itertools.permutations(concepts))
    res = {}
    for dname, labfn, truthfn in defs:
        truth = {c: truthfn([rows[i] for i in te_idx], c) for c in concepts}
        prior = {c: float(np.clip(truth[c].mean(), 0.02, 0.98)) for c in concepts}
        out = {}
        for kind in probe_kinds:
            per = {}
            for seed in seeds:
                rng = np.random.default_rng(seed)
                W, Wsh, Dm = {}, {}, {}
                for c in concepts:
                    gi, y = labfn(rows, tr_idx, c, rng)
                    Xi = Xtr[[loc[g] for g in gi]]
                    W[c] = fit_probe(Xi, y, seed, kind)
                    Wsh[c] = fit_probe(Xi, rng.permutation(y), seed, kind)
                    Dm[c] = Xi[y == 1].mean(0) - Xi[y == 0].mean(0)      # label cross-covariance direction (for LEACE)
                conds = {"original": [W]}
                # LEACE-style readout: read concept c after erasing the other concepts' label cross-covariance
                # (closed form, oblique projection in the whitened space): w' = (I - Sig^{-1/2} Q Sig^{1/2}) w
                lea = {}
                for c in concepts:
                    U_ = Sinv_half @ np.stack([Dm[o] for o in concepts if o != c], 1)
                    Q_ = U_ @ np.linalg.pinv(U_.T @ U_) @ U_.T
                    lea[c] = W[c] - Sinv_half @ (Q_ @ (S_half @ W[c]))
                conds["leace"] = [lea]
                # pattern -> filter mapping (the reading recipe): Sigma^{-1} w
                conds["filter_inv"] = [{c: Sinv @ W[c] for c in concepts}]
                conds["euclid_orth"] = [dict(zip(o, gram_schmidt([W[c] for c in o]))) for o in orders]
                conds["sigma_orth"] = [dict(zip(o, gram_schmidt([W[c] for c in o], Sigma))) for o in orders]
                conds["sigmaW_orth"] = [dict(zip(o, gram_schmidt([W[c] for c in o], Sigma_w))) for o in orders]
                conds["sigmaLW_orth"] = [dict(zip(o, gram_schmidt([W[c] for c in o], Sigma_lw))) for o in orders]
                conds["shuffled_orth"] = [dict(zip(concepts, gram_schmidt([Wsh[c] for c in concepts])))]
                if kind == probe_kinds[0]:
                    conds["random_euclid"] = [dict(zip(concepts, gram_schmidt([rng.standard_normal(Xtr.shape[1]) for _ in concepts]))) for _ in range(n_random)]
                    conds["random_sigma"] = [dict(zip(concepts, gram_schmidt([rng.standard_normal(Xtr.shape[1]) for _ in concepts], Sigma))) for _ in range(n_random)]
                for cn, variants in conds.items():
                    for vi, V in enumerate(variants):
                        npm = n_perm if (vi == 0 and cn in ("original", "euclid_orth", "sigma_orth", "random_sigma")) else 0
                        S = {c: Xe @ V[c] for c in concepts}
                        rec = {}
                        for c in concepts:
                            yv = truth[c]
                            rec[f"auc_{c}"] = float(roc_auc_score(yv, S[c])) if 0 < yv.sum() < len(yv) and np.all(np.isfinite(S[c])) else float("nan")
                            # share of held-out score variance that lies between label cells
                            sc_ = S[c]; tv_ = sc_.var()
                            if tv_ > 0:
                                wv_ = sum(((sc_[i] - sc_[i].mean()) ** 2).sum() for i in cells.idx) / len(sc_)
                                rec[f"bshare_{c}"] = float(1 - wv_ / tv_)
                            else:
                                rec[f"bshare_{c}"] = float("nan")
                        for a, b in pairs:
                            k = f"{a[0].upper()}-{b[0].upper()}"
                            rec[f"cos_{k}"] = float(V[a] @ V[b] / (np.linalg.norm(V[a]) * np.linalg.norm(V[b]) + 1e-12))
                            # standard projection audit: does concept a's score separate the levels of concept b among
                            # inputs with the same level of a? (mean difference within a-level, in within-cell SD units)
                            for (u_, w_, tag_) in ((a, b, "proj"), (b, a, "projr")):
                                su, yu, yw = S[u_], truth[u_], truth[w_]
                                gaps = []
                                for lv in (0, 1):
                                    m1 = (yu == lv) & (yw == 1); m0 = (yu == lv) & (yw == 0)
                                    if m1.sum() > 5 and m0.sum() > 5: gaps.append(su[m1].mean() - su[m0].mean())
                                wsd = np.sqrt(np.mean([su[i].var() for i in cells.idx if len(i) > 1])) + 1e-12
                                rec[f"{tag_}_{k}"] = float(np.mean(gaps) / wsd) if gaps else float("nan")
                            for th in ("median", "prior"):
                                fa, fb = (0.5, 0.5) if th == "median" else (prior[a], prior[b])
                                for mk, mv in pair_metrics(S[a], S[b], fa, fb, cells, rng, npm).items():
                                    rec[f"{th}_{mk}_{k}"] = mv
                        per.setdefault(cn, []).append(rec)
            out[kind] = {cn: agg(v) for cn, v in per.items()}
        res[dname] = out
    return res


def agg(recs):
    o = {}
    for k in recs[0]:
        v = np.array([r[k] for r in recs], float)
        o[k] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)), "n": int(np.sum(np.isfinite(v)))}
    return o


def layer_list(spec, nL):
    if spec == "all": return list(range(1, nL))
    if spec == "last": return [nL - 1]
    if spec.startswith("frac"):            # k layers evenly spaced from the first to the last hidden state
        k = int(spec[4:]); return sorted({int(round(1 + (nL - 2) * i / (k - 1))) for i in range(k)})
    if spec.startswith("stride"):
        k = int(spec[6:]); return sorted(set(range(1, nL, k)) | {nL - 1})
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["roberta-base", "gpt2", "EleutherAI/pythia-160m"])
    ap.add_argument("--bs", type=int, default=32, help="extraction batch size (lower for 3B models on long texts)")
    ap.add_argument("--datasets", nargs="+", default=["stereoset", "factorial", "bios"], choices=list(DATASETS))
    ap.add_argument("--layers", default="stride2", help="all | last | strideK | fracK | comma list (hidden state 0 always excluded)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--out", default="car_audit2_results.json")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--template-cells", action="store_true", help="factorial: condition on template x labels")
    a = ap.parse_args()
    global TEMPLATE_CELLS
    TEMPLATE_CELLS = a.template_cells
    seeds = list(range(a.seeds)); results = {}; t0 = time.time()
    for dsn in a.datasets:
        spec = DATASETS[dsn]
        try:
            rows = synthetic_rows(dsn) if a.synthetic else spec["load"]()
        except Exception as e:
            print(f"\n##### {dsn}: could not load ({e!r}); skipping this dataset", flush=True)
            continue
        te = split_rows(rows); tr = ~te
        print(f"\n##### {dsn}: {len(rows)} rows, {int(te.sum())} held out, cells={len(set(r['cell'] for r in rows))}", flush=True)
        for model in (["synthetic"] if a.synthetic else a.models):
            if a.synthetic:
                rng = np.random.default_rng(0); d, L = 48, 3
                scale = np.ones(d); scale[:2] = [15, 6]
                X = rng.standard_normal((len(rows), L + 1, d)) * scale
                sig = {c: rng.standard_normal(d) for c in ("gender", "race", "profession")}
                shared = rng.standard_normal(d)
                for i, r in enumerate(rows):
                    for c in spec["concepts"]:
                        lab = (r["bias"] == c) if dsn == "stereoset" else r[c]
                        X[i, 1:, :] += 1.2 * sig[c] * (2 * lab - 1)
                    if dsn != "stereoset" and r["gender"] == 1:      # planted conditional entanglement in layer 3 only
                        X[i, 3, :] += 0.0
                X[:, 3, :] += np.outer(rng.standard_normal(len(rows)), sig["gender"] + sig["profession"]) * 0.8  # shared noise -> CE>0
                arch = "synthetic"
            else:
                print(f"=== {model}", flush=True)
                X, arch = extract(model, [r["text"] for r in rows], bs=a.bs, half=a.half)
            nL = X.shape[1]
            for l in layer_list(a.layers, nL):
                rel = float(X[:, l, :].std(0).mean() / (np.abs(X[:, l, :]).mean() + 1e-12))
                key = f"{dsn}|{model}|{l}"
                if rel < 1e-3:
                    results[key] = {"arch": arch, "rel_spread": rel, "analysis": {}, "note": "constant layer"}; continue
                try:
                    results[key] = {"arch": arch, "rel_spread": rel,
                                    "analysis": analyse_layer(X[:, l, :].astype(np.float64), rows, tr, te, spec["concepts"], spec["defs"], seeds, a.n_perm)}
                except Exception as e:
                    import traceback; traceback.print_exc()
                    results[key] = {"arch": arch, "rel_spread": rel, "analysis": {}, "error": repr(e)}
                print(f"  layer {l}/{nL - 1} done ({time.time() - t0:.0f}s)", flush=True)
                json.dump(results, open(a.out, "w"))       # save after every layer (long runs can be interrupted)
            del X
            json.dump(results, open(a.out, "w"))
    json.dump(results, open(a.out, "w"))
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
