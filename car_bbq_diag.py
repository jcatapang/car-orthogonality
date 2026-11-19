"""BBQ diagnostic. Why was biased-answer steering not category-selective?
(1) Are the per-category biased-answer directions one shared direction? (cosine and score correlation matrices)
(2) Continuous metric: change in the biased-vs-other answer logit margin on ambiguous items, per category,
    with bootstrap CIs (argmax flips are too coarse).
(3) Layer: steering at a third, half and two thirds of depth.
(4) Does orthogonalizing the categories' filters against each other (Euclidean, Sigma, within-cell Sigma_W)
    and steering along the pattern make steering selective?
"""
import argparse, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from car_bbq2 import LM, load_bbq, prompt_text, gs, PILOT


def run(lm, texts, collect=None, steer=None):
    torch = lm.torch; handle = None
    if steer is not None:
        b, vec = steer
        v = torch.tensor(vec, dtype=next(lm.model.parameters()).dtype, device=lm.dev)
        handle = lm.blocks[b].register_forward_hook(lambda m, i, o: (o[0] + v,) + tuple(o[1:]) if isinstance(o, tuple) else o + v)
    L, H = [], []
    try:
        with torch.no_grad():
            for i in range(0, len(texts), lm.bs):
                enc = lm.tok(texts[i:i + lm.bs], return_tensors="pt", padding=True).to(lm.dev)
                o = lm.model(**enc, output_hidden_states=collect is not None)
                L.append(o.logits[:, -1, lm.letter_ids].float().cpu().numpy())
                if collect is not None:
                    H.append(np.stack([o.hidden_states[h][:, -1].float().cpu().numpy() for h in collect], 1))
    finally:
        if handle is not None:
            handle.remove()
    return np.concatenate(L), (np.concatenate(H).astype(np.float64) if H else None)


def margin(items, logits):
    """biased answer logit minus the other non-unknown answer logit (ambiguous items)"""
    m = []
    for it, lg in zip(items, logits):
        other = [j for j in range(3) if j != it["unk"] and j != it["target_loc"]][0]
        m.append(lg[it["target_loc"]] - lg[other])
    return np.array(m)


def boot(x, rng, B=1000):
    bs = [rng.choice(x, len(x)).mean() for _ in range(B)]
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-amb", type=int, default=300)
    ap.add_argument("--k", type=float, default=16.0)
    ap.add_argument("--out", default="car_bbq_diag.json")
    a = ap.parse_args()
    cats = PILOT
    train, test = load_bbq(cats, a.n_train, a.n_amb, 20)
    amb = [it for it in test if it["context_condition"] == "ambig"]
    lm = LM(a.model, a.half); lm.bs = a.bs
    nb = len(lm.blocks); blocks = sorted({nb // 3, nb // 2, 2 * nb // 3})
    tr_txt = [prompt_text(it, lm.tok) for it in train]; te_txt = [prompt_text(it, lm.tok) for it in amb]
    t0 = time.time(); rng = np.random.default_rng(0)
    tr_lg, Htr = run(lm, tr_txt, collect=[b + 1 for b in blocks])
    te_lg0, Hte = run(lm, te_txt, collect=[b + 1 for b in blocks])
    tr_pred, te_pred = tr_lg.argmax(1), te_lg0.argmax(1)
    m0 = margin(amb, te_lg0)
    cat_tr = np.array([it["category"] for it in train]); cat_te = np.array([it["category"] for it in amb])
    print(f"baseline done ({time.time() - t0:.0f}s)", flush=True)
    out = {"model": a.model, "blocks": blocks, "k": a.k, "layers": {}}
    for j, b in enumerate(blocks):
        mu = Htr[:, j].mean(0); X = Htr[:, j] - mu; Xte = Hte[:, j] - mu; sc = X.std() + 1e-8
        S = np.cov(X, rowvar=False); S = S + 1e-4 * np.trace(S) / len(S) * np.eye(len(S))
        W, D, auc, cells = {}, {}, {}, np.full(len(train), -1)
        lab = np.array([-1 if tr_pred[i] == train[i]["unk"] else int(tr_pred[i] == train[i]["target_loc"]) for i in range(len(train))])
        for ci, c in enumerate(cats):
            idx = np.where((cat_tr == c) & (lab >= 0))[0]
            W[c] = LogisticRegression(C=1.0, max_iter=2000).fit(X[idx] / sc, lab[idx]).coef_[0] / sc
            D[c] = X[idx][lab[idx] == 1].mean(0) - X[idx][lab[idx] == 0].mean(0)
            cells[idx] = 2 * ci + lab[idx]
            ti = [i for i in range(len(amb)) if cat_te[i] == c and te_pred[i] != amb[i]["unk"]]
            yt = np.array([int(te_pred[i] == amb[i]["target_loc"]) for i in ti])
            auc[c] = float(roc_auc_score(yt, Xte[ti] @ W[c])) if 0 < yt.sum() < len(yt) else float("nan")
        Xw = X.copy()
        for cv in np.unique(cells[cells >= 0]):
            Xw[cells == cv] -= Xw[cells == cv].mean(0)
        keep = cells >= 0
        SW = (Xw[keep].T @ Xw[keep]) / max(keep.sum() - len(np.unique(cells[keep])), 1)
        SW = SW + 1e-4 * np.trace(SW) / len(SW) * np.eye(len(SW))
        cosm = lambda V: [[float(V[p] @ V[q] / np.linalg.norm(V[p]) / np.linalg.norm(V[q])) for q in cats] for p in cats]
        rsm = lambda V: [[float(V[p] @ S @ V[q] / np.sqrt((V[p] @ S @ V[p]) * (V[q] @ S @ V[q]))) for q in cats] for p in cats]
        L = {"auc": auc, "cos_W": cosm(W), "rsig_W": rsm(W), "cos_D": cosm(D), "rsig_D": rsm(D), "steer": {}}
        for T in cats:
            others = [c for c in cats if c != T]
            wE = gs([W[c] for c in others] + [W[T]])[-1]
            wS = gs([W[c] for c in others] + [W[T]], S)[-1]
            wW = gs([W[c] for c in others] + [W[T]], SW)[-1]
            sdT = (X @ W[T]).std()
            vecs = {"dom": D[T], "lr": W[T], "euclid": wE, "sigvec": S @ wS, "sigWvec": SW @ wW}
            sv = {vn: -a.k * sdT * v / (W[T] @ v) for vn, v in vecs.items() if abs(W[T] @ v) > 1e-12}
            r = rng.standard_normal(len(wE)); sv["random"] = r / np.linalg.norm(r) * np.linalg.norm(sv["sigvec"])
            for vn, v in sv.items():
                lg, _ = run(lm, te_txt, steer=(b, v))
                d = margin(amb, lg) - m0
                per = {c: boot(d[cat_te == c], rng) for c in cats}
                on = abs(per[T][0]); off = np.mean([abs(per[c][0]) for c in others])
                L["steer"][f"{T}|{vn}"] = {"per_cat": per, "sel": float(on / max(off, 1e-9)), "norm": float(np.linalg.norm(v)),
                                           "frac_changed": float(np.mean(lg.argmax(1) != te_pred))}
            print(f"  block {b} target {T} done ({time.time() - t0:.0f}s)", flush=True)
        out["layers"][str(b)] = L
        json.dump(out, open(a.out, "w"), indent=1)
    print(f"Saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
