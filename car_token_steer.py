"""Token-level steering spillover. Steer along the gender topic direction (vectors as in car_steer.py) and
measure what the model outputs next: change in log-probability mass on gendered words (target) and on
profession nouns (spillover), on held-out StereoSet sentences.
"""
import argparse, json, time
import numpy as np
import torch
from car_steer import load_rows, topic_labels, fit

GENDER_WORDS = [" He", " She", " His", " Her", " he", " she", " his", " her", " man", " woman"]
PROF_WORDS = [" doctor", " nurse", " teacher", " engineer", " lawyer", " manager", " chef", " pilot", " scientist",
              " carpenter", " secretary", " mechanic", " plumber", " surgeon", " professor", " clerk"]


class LM:
    def __init__(self, name, half, bs=16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=(torch.bfloat16 if half else torch.float32)).to(self.dev).eval()
        base = self.model.base_model
        self.blocks = getattr(base, "layers", None) or getattr(base, "h")
        self.bs = bs
        ids = lambda ws: sorted({self.tok.encode(w, add_special_tokens=False)[0] for w in ws})
        self.gid, self.pid = ids(GENDER_WORDS), ids(PROF_WORDS)

    @torch.no_grad()
    def run(self, texts, collect=None, steer=None):
        handle = None
        if steer is not None:
            b, vec = steer
            v = torch.tensor(vec, dtype=next(self.model.parameters()).dtype, device=self.dev)
            handle = self.blocks[b].register_forward_hook(lambda m, i, o: (o[0] + v,) + tuple(o[1:]) if isinstance(o, tuple) else o + v)
        G, P, H = [], [], []
        try:
            for i in range(0, len(texts), self.bs):
                enc = self.tok(texts[i:i + self.bs], return_tensors="pt", padding=True).to(self.dev)
                o = self.model(**enc, output_hidden_states=collect is not None)
                lp = torch.log_softmax(o.logits[:, -1].float(), -1)
                G.append(torch.logsumexp(lp[:, self.gid], -1).cpu().numpy())
                P.append(torch.logsumexp(lp[:, self.pid], -1).cpu().numpy())
                if collect is not None:
                    H.append(np.stack([o.hidden_states[h][:, -1].float().cpu().numpy() for h in collect], 1))
        finally:
            if handle is not None:
                handle.remove()
        return np.concatenate(G), np.concatenate(P), (np.concatenate(H).astype(np.float64) if H else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--ks", nargs="+", type=float, default=[4.0])
    ap.add_argument("--out", default="car_token_steer.json")
    a = ap.parse_args()
    rows = [r for r in load_rows() if r["bias"] in ("gender", "profession", "race")]
    groups = sorted({r["group"] for r in rows}); rng = np.random.default_rng(42)
    test_g = set(rng.choice(groups, int(0.3 * len(groups)), replace=False))
    tr = list(rng.choice([i for i, r in enumerate(rows) if r["group"] not in test_g], a.n_train, replace=False))
    te = list(rng.choice([i for i, r in enumerate(rows) if r["group"] in test_g], a.n_test, replace=False))
    te_txt = [rows[i]["text"] + " The" for i in te]          # next token: a noun position (profession or gendered word)
    out, t0 = {}, time.time()
    for name in a.models:
        print(f"\n=== {name}", flush=True)
        lm = LM(name, a.half)
        nb = len(lm.blocks); blocks = sorted({max(1, nb // 3), nb // 2, max(1, 2 * nb // 3)})
        _, _, Htr = lm.run([rows[i]["text"] for i in tr], collect=[b + 1 for b in blocks])
        g0, p0, _ = lm.run(te_txt)
        res = {"blocks": blocks, "base_g": float(g0.mean()), "base_p": float(p0.mean()), "steer": {}}
        srng = np.random.default_rng(0)
        for j, b in enumerate(blocks):
            X = Htr[:, j] - Htr[:, j].mean(0); sc = X.std() + 1e-8; loc = {g: k for k, g in enumerate(tr)}
            W = {}
            for c in ("gender", "profession"):
                gi, y = topic_labels(rows, tr, c, srng)
                W[c] = fit(X[[loc[g] for g in gi]] / sc, y) / sc
            S = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1]); sd = (X @ W["gender"]).std()
            wg, wp = W["gender"], W["profession"]
            ws = wg - (wg @ S @ wp) / (wp @ S @ wp) * wp
            vecs = {"original": wg, "euclid_orth": wg - (wg @ wp) / (wp @ wp) * wp, "sigma_orth": ws, "sigma_vec": S @ ws}
            for k in a.ks:
                sv = {vn: k * sd * v / (v @ wg) for vn, v in vecs.items()}
                r = srng.standard_normal(len(wg)); sv["random"] = r / np.linalg.norm(r) * np.linalg.norm(sv["euclid_orth"])
                for vn, v in sv.items():
                    g, p, _ = lm.run(te_txt, steer=(b, v))
                    dg, dp = g - g0, p - p0
                    res["steer"][f"{b}|{vn}|{k:g}"] = {"dG": float(dg.mean()), "dP": float(dp.mean()),
                                                      "absdG": float(np.abs(dg).mean()), "absdP": float(np.abs(dp).mean()),
                                                      "norm": float(np.linalg.norm(v))}
            print(f"  block {b} done ({time.time() - t0:.0f}s)", flush=True)
        out[name] = res
        json.dump(out, open(a.out, "w"), indent=1); print(f"Saved {a.out}", flush=True)
        del lm; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
