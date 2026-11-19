#!/usr/bin/env python
"""
Phase 4: does the choice of orthogonalization metric matter for INTERVENTION?

For a decoder model and a steering layer l:
  * fit gender and profession probes (topic labels, StereoSet train split) at every hidden state
  * build four steering vectors at layer l: gender probe as trained, Euclidean-orthogonalized to the
    profession probe, Sigma-orthogonalized to it, and a random direction
  * scale each so it raises the layer-l gender score by K standard deviations
  * add it to the residual stream (all positions) at block l on held-out sentences
  * read out gender and profession probe scores at every later hidden state (last token)
Spillover = |shift in profession score| / |shift in gender score|, both in units of the unsteered
score standard deviation at that layer.

Theory: at layer l the profession shift is alpha * w_p . v, so Euclidean orthogonality gives zero
first-order spillover at the steering layer; what happens downstream is the empirical question.
"""
import argparse, json, time, urllib.request
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

STEREOSET_URL = "https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json"


def load_rows():
    raw = json.loads(urllib.request.urlopen(STEREOSET_URL, timeout=60).read())["data"]
    rows = []
    for split in ("intrasentence", "intersentence"):
        for ex in raw[split]:
            for s in ex["sentences"]:
                if s["gold_label"] in ("stereotype", "anti-stereotype"):
                    t = s["sentence"] if split == "intrasentence" else f'{ex["context"]} {s["sentence"]}'
                    rows.append(dict(group=f'{split}:{ex["id"]}', bias=ex["bias_type"], text=t))
    return rows


def blocks_of(model):
    for attr in ("h", "layers"):
        if hasattr(model, attr):
            return getattr(model, attr)
    raise ValueError("unknown block container")


class Runner:
    def __init__(self, name, half=False, bs=16, max_len=96):
        from transformers import AutoModel, AutoTokenizer
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModel.from_pretrained(name, dtype=(torch.bfloat16 if half else torch.float32)).to(self.dev).eval()
        self.blocks = blocks_of(self.model)
        self.bs, self.max_len = bs, max_len

    @torch.no_grad()
    def hidden(self, texts, steer=None):
        """Last-token hidden states (n, L+1, d). steer=(block_index, vector) adds vector at that block's output."""
        handle = None
        if steer is not None:
            bi, vec = steer
            v = torch.tensor(vec, dtype=next(self.model.parameters()).dtype, device=self.dev)

            def hook(mod, inp, out):
                if isinstance(out, tuple):
                    return (out[0] + v,) + tuple(out[1:])
                return out + v
            handle = self.blocks[bi].register_forward_hook(hook)
        chunks = []
        try:
            for i in range(0, len(texts), self.bs):
                b = self.tok(texts[i:i + self.bs], return_tensors="pt", padding=True, truncation=True, max_length=self.max_len).to(self.dev)
                hs = self.model(**b, output_hidden_states=True).hidden_states
                pos = b["attention_mask"].sum(1) - 1
                ar = torch.arange(len(pos), device=self.dev)
                chunks.append(torch.stack([h[ar, pos] for h in hs], 1).float().cpu().numpy())
        finally:
            if handle is not None:
                handle.remove()
        return np.concatenate(chunks).astype(np.float64)


def topic_labels(rows, idx, concept, rng):
    pos = [i for i in idx if rows[i]["bias"] == concept]
    neg = [i for i in idx if rows[i]["bias"] != concept]
    neg = list(rng.choice(neg, min(len(neg), len(pos)), replace=False))
    return np.array(pos + neg), np.r_[np.ones(len(pos)), np.zeros(len(neg))]


def fit(X, y):
    return LogisticRegression(C=100, max_iter=1000, tol=1e-3).fit(X, y).coef_[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gpt2", "EleutherAI/pythia-160m", "Qwen/Qwen2.5-1.5B"])
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--k", type=float, default=2.0, help="target gender shift at the steering layer, in score SDs")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="car_steer_results.json")
    ap.add_argument("--tiny-test", action="store_true", help="random-weight tiny GPT-2 + synthetic rows (pipeline test)")
    a = ap.parse_args()

    if a.tiny_test:
        biases = ["gender", "profession", "race", "religion"]
        words = {"gender": "he she man woman", "profession": "doctor nurse pilot clerk", "race": "black white asian", "religion": "church mosque"}
        rng0 = np.random.default_rng(0)
        rows = [dict(group=f"g{i}", bias=biases[i % 4], text=" ".join(rng0.choice(words[biases[i % 4]].split(), 5))) for i in range(400)]
    else:
        rows = load_rows()
    rows = [r for r in rows if r["bias"] in ("gender", "profession", "race")]
    groups = sorted({r["group"] for r in rows})
    rng = np.random.default_rng(42)
    test_g = set(rng.choice(groups, int(0.3 * len(groups)), replace=False))
    tr = [i for i, r in enumerate(rows) if r["group"] not in test_g]
    te = [i for i, r in enumerate(rows) if r["group"] in test_g]
    tr = list(rng.choice(tr, min(a.n_train, len(tr)), replace=False))
    te = list(rng.choice(te, min(a.n_test, len(te)), replace=False))
    results, t0 = {}, time.time()

    for name in (["tiny"] if a.tiny_test else a.models):
        if a.tiny_test:
            from transformers import GPT2Config, GPT2Model, AutoTokenizer
            R = Runner.__new__(Runner)
            R.dev = "cpu"; R.bs, R.max_len = 16, 32
            cfg = GPT2Config(n_layer=4, n_embd=64, n_head=4, vocab_size=50257)
            R.model = GPT2Model(cfg).eval(); R.blocks = R.model.h
            from transformers import PreTrainedTokenizerFast
            from tokenizers import Tokenizer, models, pre_tokenizers
            tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(sorted({w for r in rows for w in r["text"].split()} | {"[PAD]", "[UNK]"}))}, unk_token="[UNK]"))
            tk.pre_tokenizer = pre_tokenizers.Whitespace()
            R.tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]"); R.tok.padding_side = "right"
        else:
            print(f"\n=== {name}", flush=True)
            R = Runner(name, half=a.half)
        Htr = R.hidden([rows[i]["text"] for i in tr])
        Hte = R.hidden([rows[i]["text"] for i in te])
        nH = Htr.shape[1]                     # hidden states: 0 = embeddings, b+1 = output of block b
        steer_blocks = sorted({max(1, (nH - 1) // 3), (nH - 1) // 2, max(1, 2 * (nH - 1) // 3)})
        mres = {}
        for seed in range(a.seeds):
            srng = np.random.default_rng(seed)
            P = {}
            for h in range(1, nH):
                mu, X = Htr[:, h].mean(0), Htr[:, h] - Htr[:, h].mean(0)
                sc = X.std() + 1e-8
                W = {}
                for c in ("gender", "profession"):
                    loc = {g: k for k, g in enumerate(tr)}
                    gi, y = topic_labels(rows, tr, c, srng)
                    W[c] = fit(X[[loc[g] for g in gi]] / sc, y) / sc   # weights in raw activation units
                Xe = Hte[:, h] - mu
                P[h] = dict(mu=mu, W=W, Sigma=np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1]),
                            sd={c: (Xe @ W[c]).std() + 1e-12 for c in W}, base={c: Xe @ W[c] for c in W})
            for b in steer_blocks:
                h = b + 1
                wg, wp, S = P[h]["W"]["gender"], P[h]["W"]["profession"], P[h]["Sigma"]
                vecs = {"original": wg,
                        "euclid_orth": wg - (wg @ wp) / (wp @ wp) * wp,
                        "sigma_orth": wg - (wg @ S @ wp) / (wp @ S @ wp) * wp,
                        # covariance-consistent steering VECTOR for the Sigma-orthogonal probe (covector):
                        # v = Sigma w_g^Sigma, which satisfies w_p . v = 0 exactly (Corollary 1)
                        "sigma_vec": S @ (wg - (wg @ S @ wp) / (wp @ S @ wp) * wp),
                        "random": srng.standard_normal(len(wg))}
                scaled = {}
                for vn, v in vecs.items():
                    gain = (v @ wg) / P[h]["sd"]["gender"]          # gender-SD shift per unit of v at layer h
                    if vn == "random" or abs(gain) < 1e-9:
                        v = v / np.linalg.norm(v) * np.linalg.norm(vecs["original"] / np.linalg.norm(vecs["original"]) * a.k * P[h]["sd"]["gender"] / ((vecs["original"] / np.linalg.norm(vecs["original"])) @ wg))
                    else:
                        v = v * (a.k / gain)
                    scaled[vn] = v
                # norm-matched controls: everything at the norm of the scaled pattern vector
                nb_ = np.linalg.norm(scaled["sigma_vec"])
                rb_ = srng.standard_normal(len(wg))
                scaled["random_big"] = rb_ / np.linalg.norm(rb_) * nb_
                scaled["euclid_big"] = scaled["euclid_orth"] / np.linalg.norm(scaled["euclid_orth"]) * nb_
                for vn, v in scaled.items():
                    Hs = R.hidden([rows[i]["text"] for i in te], steer=(b, v))
                    # transformers records hidden_states with its own hook, which may capture the block output
                    # before ours adds v; if the steering layer came back unchanged, apply v there explicitly
                    if np.allclose(Hs[:, h], Hte[:, h], atol=1e-4):
                        Hs[:, h] = Hs[:, h] + v
                    for hh in range(h, nH):
                        d = {c: float(np.mean((Hs[:, hh] - P[hh]["mu"]) @ P[hh]["W"][c] - P[hh]["base"][c]) / P[hh]["sd"][c]) for c in ("gender", "profession")}
                        key = f"block{b}|{vn}|h{hh}"
                        mres.setdefault(key, []).append(dict(d_gender=d["gender"], d_prof=d["profession"],
                                                             spill=abs(d["profession"]) / max(abs(d["gender"]), 1e-9),
                                                             steer_norm=float(np.linalg.norm(v)), rel_depth=(hh - h) / max(nH - 1 - h, 1)))
            print(f"  seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
        results[name] = {k: {m: float(np.mean([x[m] for x in v])) for m in v[0]} | {"n": len(v)} for k, v in mres.items()}
        results[name]["_meta"] = {"n_hidden": nH, "steer_blocks": steer_blocks}
        json.dump(results, open(a.out, "w"), indent=1)
        del R
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # summary
    for name, res in results.items():
        meta = res["_meta"]; print(f"\n{name}  (hidden states {meta['n_hidden']})")
        for b in meta["steer_blocks"]:
            for vn in ("original", "euclid_orth", "sigma_orth", "sigma_vec", "random", "random_big", "euclid_big"):
                ks = sorted([k for k in res if k.startswith(f"block{b}|{vn}|")], key=lambda k: int(k.split("|h")[1]))
                if not ks: continue
                first, last = res[ks[0]], res[ks[-1]]
                print(f"  block {b:>2} {vn:12s} at steer layer: dG={first['d_gender']:+.2f} dP={first['d_prof']:+.2f} | final: dG={last['d_gender']:+.2f} dP={last['d_prof']:+.2f} spill={last['spill']:.2f}")
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
