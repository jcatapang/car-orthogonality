#!/usr/bin/env python
"""
CAR audit: corrected E5 / E7 with the controls reviewers asked for.

What it tests (per model, per layer, per label definition, 5 seeds):
  original      - probes as trained (what the AAAI code actually measured for E5)
  euclid_orth   - Gram-Schmidt in the Euclidean metric, thresholds RE-calibrated (what the paper claims)
  sigma_orth    - Gram-Schmidt in the activation-covariance metric (the proposed fix)
  random_orth   - 3 random Euclidean-orthonormal directions, no concept at all (null control)
  shuffled_orth - probes trained on shuffled labels, then Euclidean-orthogonalized (null control)
  leace         - erase gender with LEACE, retrain gender+profession probes, pairwise overlap (E7)

Label definitions:
  stereo   - positive = stereotype sentence, negative = anti-stereotype sentence, within one bias type
  presence - positive = sentence of this bias type, negative = sentence of another bias type

All overlap is reported against the CORRECT independence baseline: 0.25 pairwise, 0.125 three-way.
Thresholds split the held-out pool exactly 50/50 by rank, so median ties cannot inflate overlap.

Colab:
  !pip -q install transformers scikit-learn concept-erasure
  !python car_audit.py --models roberta-base gpt2 EleutherAI/pythia-160m --layers all
Results: car_audit_results.json (send this back) + a printed summary.
"""
import argparse, json, math, sys, time, urllib.request
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CONCEPTS = ["gender", "race", "profession"]
ORTH_ORDER = ["profession", "gender", "race"]  # same conditioning order as the AAAI code
STEREOSET_URL = "https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json"
ALL_MODELS = ["roberta-base", "roberta-large", "bert-base-uncased", "answerdotai/ModernBERT-base",
              "gpt2", "gpt2-medium", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]
DECODER_TYPES = {"gpt2", "gpt_neox", "gpt_neo", "llama", "mistral", "qwen2", "gemma", "gemma2", "olmo", "olmo2", "phi", "phi3"}


# ---------------------------------------------------------------- data
def load_stereoset():
    raw = json.loads(urllib.request.urlopen(STEREOSET_URL, timeout=60).read())["data"]
    rows = []
    for split in ("intrasentence", "intersentence"):
        for ex in raw[split]:
            for s in ex["sentences"]:
                if s["gold_label"] not in ("stereotype", "anti-stereotype"):
                    continue
                text = s["sentence"] if split == "intrasentence" else f'{ex["context"]} {s["sentence"]}'
                rows.append(dict(group=f'{split}:{ex["id"]}', bias=ex["bias_type"], text=text,
                                 stereo=int(s["gold_label"] == "stereotype")))
    return rows


# Factorial templates: gender x race x profession fully crossed, so the three LABELS are
# statistically independent by construction. Any excess co-activation of probes on this data
# is a property of the representation, not of label co-occurrence in the corpus (Uet5 W1).
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


def build_factorial(seed=0, per_cell=400):
    rng = np.random.default_rng(seed)
    rows = []
    for t_id, t in enumerate(FACT_TEMPLATES):
        for g in (0, 1):
            for r in (0, 1):
                for pr in (0, 1):
                    for _ in range(max(1, per_cell // len(FACT_TEMPLATES))):
                        rows.append(dict(group=f"t{t_id}", bias="factorial", stereo=0,
                                         text=t.format(g=rng.choice(FACT_GENDER[g]), r=rng.choice(FACT_RACE[r]), p=rng.choice(FACT_PROF[pr])),
                                         gender=g, race=r, profession=pr))
    return rows


def split_rows(rows, seed=42, test_frac=0.3):
    groups = sorted({r["group"] for r in rows})
    rng = np.random.default_rng(seed)
    test = set(rng.choice(groups, int(test_frac * len(groups)), replace=False))
    return np.array([r["group"] in test for r in rows])


def labels_for(rows, mask, concept, definition, rng):
    """Return (indices, y) for training/validating a probe of `concept` among rows[mask]."""
    idx = np.where(mask)[0]
    if definition == "factorial":
        return idx, np.array([rows[i][concept] for i in idx])
    bias = np.array([rows[i]["bias"] for i in idx])
    if definition == "stereo":
        sel = idx[bias == concept]
        return sel, np.array([rows[i]["stereo"] for i in sel])
    pos = idx[bias == concept]
    neg = idx[bias != concept]
    neg = rng.choice(neg, min(len(neg), len(pos)), replace=False)
    sel = np.concatenate([pos, neg])
    return sel, np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(int)


# ---------------------------------------------------------------- activations
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
            hs = model(**b, output_hidden_states=True).hidden_states  # tuple of L+1 (B,T,D); [0] = embeddings
            pos = (b["attention_mask"].sum(1) - 1) if is_dec else torch.zeros(len(b["input_ids"]), dtype=torch.long, device=dev)
            ar = torch.arange(len(pos), device=dev)
            chunks.append(torch.stack([h[ar, pos] for h in hs], 1).float().cpu().numpy())  # (B, L+1, D)
    return np.concatenate(chunks), ("decoder" if is_dec else "encoder")


# ---------------------------------------------------------------- geometry
def fit_probe(X, y, seed, sub=0.8):
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(y), max(10, int(sub * len(y))), replace=False)
    clf = LogisticRegression(C=100, max_iter=1000, tol=1e-3).fit(X[keep], y[keep])
    return clf.coef_[0].astype(np.float64)


def gram_schmidt(ws, M=None):
    """Orthogonalize a list of vectors in order; metric M (None = Euclidean)."""
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


def rank_membership(s):
    """Exactly-50% membership by rank (immune to median ties)."""
    m = np.zeros(len(s), bool)
    m[np.argsort(s, kind="stable")[len(s) // 2:]] = True
    return m


def overlap_metrics(W, Xe, auc_sets):
    """W: dict concept->vector. Xe: held-out pool. auc_sets: concept->(row idx into Xe, y)."""
    S = {c: Xe @ W[c] for c in CONCEPTS}
    degenerate = any(np.std(S[c]) < 1e-9 * (np.abs(S[c]).mean() + 1e-12) for c in CONCEPTS)
    M = {c: rank_membership(S[c]) for c in CONCEPTS}
    pairs = [("gender", "profession"), ("gender", "race"), ("race", "profession")]
    out = {"degenerate_scores": bool(degenerate)}
    r = {}
    for a, b in pairs:
        k = f"{a[0].upper()}-{b[0].upper()}"
        wa, wb = W[a], W[b]
        out[f"cos_{k}"] = float(wa @ wb / (np.linalg.norm(wa) * np.linalg.norm(wb) + 1e-12))
        r[k] = float(np.corrcoef(S[a], S[b])[0, 1]) if not degenerate else float("nan")
        out[f"scorecorr_{k}"] = r[k]
        out[f"mass_{k}"] = float(np.mean(M[a] & M[b]))
        out[f"excess_{k}"] = out[f"mass_{k}"] - 0.25
        out[f"gausspred_{k}"] = 0.25 + math.asin(np.clip(r[k], -1, 1)) / (2 * math.pi) if not degenerate else float("nan")
    out["mass_3way"] = float(np.mean(M["gender"] & M["race"] & M["profession"]))
    out["excess_3way"] = out["mass_3way"] - 0.125
    out["gausspred_3way"] = (0.125 + sum(math.asin(np.clip(v, -1, 1)) for v in r.values()) / (4 * math.pi)) if not degenerate else float("nan")
    for c in CONCEPTS:
        ii, y = auc_sets[c]
        ok = len(set(y)) == 2 and np.all(np.isfinite(S[c][ii]))
        out[f"auc_{c}"] = float(roc_auc_score(y, S[c][ii])) if ok else float("nan")
    return out


def maybe_leace(Xtr, Xe, rows_tr_idx, rows, tr_mask, te_mask, definition, seed, auc_sets_fn):
    try:
        import torch
        from concept_erasure import LeaceEraser
    except Exception as e:
        return {"skipped": f"concept-erasure not available: {e}"}
    rng = np.random.default_rng(seed)
    gi, gy = labels_for(rows, tr_mask, "gender", definition, rng)
    pos = {g: k for k, g in enumerate(rows_tr_idx)}
    Xg = torch.tensor(Xtr[[pos[i] for i in gi]], dtype=torch.float32)
    eraser = LeaceEraser.fit(Xg, torch.tensor(gy, dtype=torch.float32).unsqueeze(1))
    er = lambda X: eraser(torch.tensor(X, dtype=torch.float32)).numpy().astype(np.float64)
    Xtr2, Xe2 = er(Xtr), er(Xe)
    W = {}
    for c in CONCEPTS:
        ci, cy = labels_for(rows, tr_mask, c, definition, rng)
        W[c] = fit_probe(Xtr2[[pos[i] for i in ci]], cy, seed)
    m = overlap_metrics(W, Xe2, auc_sets_fn())
    return {k: v for k, v in m.items() if "G-P" in k or k.startswith("auc_") or k == "degenerate_scores"}


# ---------------------------------------------------------------- driver
def analyse_layer(X, rows, tr_mask, te_mask, seeds, n_random=20, do_leace=True, definitions=("stereo", "presence")):
    tr_idx, te_idx = np.where(tr_mask)[0], np.where(te_mask)[0]
    mu, sd = X[tr_idx].mean(0), X[tr_idx].std() + 1e-8  # centre + one global scale: directions unchanged
    Xtr, Xe = (X[tr_idx] - mu) / sd, (X[te_idx] - mu) / sd
    pos_tr = {g: k for k, g in enumerate(tr_idx)}
    Sigma = np.cov(Xtr, rowvar=False) + 1e-4 * np.eye(Xtr.shape[1])
    res = {}
    for definition in definitions:
        def auc_sets(definition=definition):
            rng = np.random.default_rng(0)
            pos_te = {g: k for k, g in enumerate(te_idx)}
            out = {}
            for c in CONCEPTS:
                ii, y = labels_for(rows, te_mask, c, definition, rng)
                out[c] = (np.array([pos_te[i] for i in ii]), y)
            return out
        A = auc_sets()
        per_cond = {k: [] for k in ("original", "euclid_orth", "sigma_orth", "random_orth", "shuffled_orth", "leace")}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            W, Wsh = {}, {}
            for c in CONCEPTS:
                ci, cy = labels_for(rows, tr_mask, c, definition, rng)
                Xi = Xtr[[pos_tr[i] for i in ci]]
                W[c] = fit_probe(Xi, cy, seed)
                Wsh[c] = fit_probe(Xi, rng.permutation(cy), seed)
            per_cond["original"].append(overlap_metrics(W, Xe, A))
            for name, src, M in (("euclid_orth", W, None), ("sigma_orth", W, Sigma), ("shuffled_orth", Wsh, None)):
                o = gram_schmidt([src[c] for c in ORTH_ORDER], M)
                per_cond[name].append(overlap_metrics(dict(zip(ORTH_ORDER, o)), Xe, A))
            for _ in range(n_random if seed == seeds[0] else 0):
                o = gram_schmidt([rng.standard_normal(Xtr.shape[1]) for _ in CONCEPTS])
                per_cond["random_orth"].append(overlap_metrics(dict(zip(ORTH_ORDER, o)), Xe, A))
            if do_leace:
                per_cond["leace"].append(maybe_leace(Xtr, Xe, tr_idx, rows, tr_mask, te_mask, definition, seed, auc_sets))
        res[definition] = {k: aggregate(v) for k, v in per_cond.items() if v}
    return res


def aggregate(dicts):
    if not dicts or "skipped" in dicts[0]:
        return dicts[0] if dicts else {}
    out = {}
    for k in dicts[0]:
        vals = np.array([d[k] for d in dicts], dtype=float)
        out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)), "n": int(len(vals))}
    return out


def summarize(all_res):
    hdr = f'{"model":24s} {"layer":>5s} {"def":8s} {"condition":13s} {"maxcos":>7s} {"rG-P":>6s} {"exG-P":>7s} {"3way":>6s} {"ex3":>7s} {"pred3":>6s} {"AUCg":>5s} {"AUCr":>5s} {"AUCp":>5s} {"degen":>5s}'
    print(hdr); print("-" * len(hdr))
    for (ds, model, layer), res in all_res.items():
        for d, conds in res["analysis"].items():
            for cond, m in conds.items():
                if "skipped" in m or not m:
                    continue
                g = lambda k: m.get(k, {}).get("mean", float("nan"))
                maxcos = max(abs(g(f"cos_{p}")) for p in ("G-P", "G-R", "R-P")) if "cos_G-R" in m else abs(g("cos_G-P"))
                print(f'{(ds[:4]+":"+model)[-24:]:24s} {layer:>5} {d:8s} {cond:13s} {maxcos:7.3f} {g("scorecorr_G-P"):6.3f} {g("excess_G-P"):+7.3f} '
                      f'{g("mass_3way"):6.3f} {g("excess_3way"):+7.3f} {g("gausspred_3way"):6.3f} {g("auc_gender"):5.2f} {g("auc_race"):5.2f} '
                      f'{g("auc_profession"):5.2f} {g("degenerate_scores"):5.2f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["roberta-base", "gpt2", "EleutherAI/pythia-160m"])
    ap.add_argument("--all-models", action="store_true", help="the paper's 8 models")
    ap.add_argument("--layers", default="last", help="'last', 'all', or comma list of hidden_states indices (0 = embeddings)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no-leace", action="store_true")
    ap.add_argument("--out", default="car_audit_results.json")
    ap.add_argument("--half", action="store_true", help="load model in bfloat16 (activations analysed in float64)")
    ap.add_argument("--dataset", default="both", choices=["stereoset", "factorial", "both"])
    ap.add_argument("--synthetic", action="store_true", help="pipeline self-test on fake activations")
    a = ap.parse_args()
    models = ALL_MODELS if a.all_models else a.models
    seeds = list(range(a.seeds))

    datasets = ["stereoset", "factorial"] if a.dataset == "both" else [a.dataset]
    all_res, t0 = {}, time.time()
    for ds in datasets:
      if a.synthetic:
        rng = np.random.default_rng(0)
        if ds == "factorial":
            rows = build_factorial()
        else:
            biases = ["gender", "race", "profession", "religion"]
            rows = [dict(group=f"g{i // 2}", bias=biases[(i // 2) % 4], text="", stereo=i % 2) for i in range(2400)]
        models = ["synthetic"]
      else:
        rows = load_stereoset() if ds == "stereoset" else build_factorial()
      defs = ("factorial",) if ds == "factorial" else ("stereo", "presence")
      print(f"\n##### dataset={ds}: {len(rows)} sentences; bias counts:", {b: sum(r['bias'] == b for r in rows) for b in sorted({r['bias'] for r in rows})})
      te = split_rows(rows); tr = ~te
      for model in models:
          if a.synthetic:
              d, L = 64, 3
              scale = np.ones(d); scale[:2] = [20, 8]
              X = rng.standard_normal((len(rows), L + 1, d)) * scale
              X[:, 0, :] = X[0, 0, :] * 10 + 1e-6 * rng.standard_normal((len(rows), d))  # near-constant layer, like RoBERTa <s> at the embeddings
              sig = {c: rng.standard_normal(d) for c in CONCEPTS}
              for i, r in enumerate(rows):
                  if ds == "factorial":
                      for c in CONCEPTS:
                          X[i, 1:, :] += 1.5 * sig[c] * (2 * r[c] - 1)
                  elif r["bias"] in sig:
                      X[i, 1:, :] += 1.5 * sig[r["bias"]] * (1 + r["stereo"])
              arch = "encoder"
          else:
              print(f"\n=== {model}: extracting", flush=True)
              X, arch = extract(model, [r["text"] for r in rows], half=a.half)
          nL = X.shape[1]
          # relative spread: input-to-input std divided by typical activation magnitude
          layer_std = [float(X[:, l, :].std(0).mean() / (np.abs(X[:, l, :]).mean() + 1e-12)) for l in range(nL)]
          if a.layers == "all":
              layers = range(nL)
          elif a.layers == "last":
              layers = [nL - 1]
          elif a.layers.startswith("stride"):  # e.g. stride3: every 3rd layer, always including first and last
              k = int(a.layers[6:]); layers = sorted(set(range(0, nL, k)) | {nL - 1})
          else:
              layers = [int(x) for x in a.layers.split(",")]
          for l in layers:
              if layer_std[l] < 1e-3:
                  all_res[(ds, model, l)] = {"arch": arch, "layer_input_std": layer_std[l], "analysis": {},
                                         "note": "activations constant across inputs at this layer; probes undefined"}
                  print(f"  layer {l}: CONSTANT across inputs (std={layer_std[l]:.2e}) -> skipped", flush=True)
                  continue
              try:
                  all_res[(ds, model, l)] = {"arch": arch, "layer_input_std": layer_std[l],
                                         "analysis": analyse_layer(X[:, l, :].astype(np.float64), rows, tr, te, seeds, do_leace=not a.no_leace, definitions=defs)}
              except Exception as e:
                  import traceback; traceback.print_exc()
                  all_res[(ds, model, l)] = {"arch": arch, "layer_input_std": layer_std[l], "analysis": {}, "error": repr(e)}
                  print(f"  layer {l}: ERROR {e!r} -> recorded, continuing", flush=True)
                  continue
              print(f"  layer {l}/{nL - 1} done ({time.time() - t0:.0f}s)", flush=True)
          del X
          json.dump({f"{d}|{m}|{l}": v for (d, m, l), v in all_res.items()}, open(a.out, "w"), indent=1)
    json.dump({f"{d}|{m}|{l}": v for (d, m, l), v in all_res.items()}, open(a.out, "w"), indent=1)
    print(); summarize(all_res); print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
