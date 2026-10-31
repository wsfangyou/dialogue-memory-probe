# -*- coding: utf-8 -*-
"""P0-2：可学习投影能把记忆检索的信噪比压到多少？

背景
----
P0 的结论是"原始表示的 SNR_pair 0.48–0.59，刚好压在判据线上"，且中层塌缩（1–4°）。
但 MSP 的 key/value 是**可学的**，训练能放大判别方向。所以 P0 的未训练结果是下界。
本实验补这个下界：在缓存表示上拟合投影，看 SNR 能压到多少。

三种投影
--------
none   原样（基线），只做 L2 归一化
whiten 无监督：PCA 白化保留 k 维，缓解各向异性
lda    有监督：正则化 Fisher 判别取前 k 个方向（线性情形下的最优）

纪律
----
- **必须留出**：按"记忆组"分组做 5 折，训练集拟合投影，测试集评估。
  不划分的结果一律不看——投影参数比样本多的场景下，训练集准确率可以是 1.000 而毫无意义。
- 超参只报曲线，不挑单点吹。选点规则写死：测试集 SNR 最低者，如并列取更小的 k。

用法
----
python train_proj.py --reps results/reps --out results/proj.json
"""
import argparse
import itertools
import json
import os

import numpy as np

from run_probe import analyze_group, l2norm


def group_key(iid):
    """G4-01a / G4-01b 归为同一组，避免同项目的两个维度被拆到训练/测试两侧"""
    return iid[:-1] if iid[-1].isalpha() else iid


def load_group(reps_dir, gname, layer, pool="mean"):
    z = np.load(os.path.join(reps_dir, f"{gname}.npz"))
    return z[pool].astype(np.float32)[layer]          # [n_texts, d]


def fit_whiten(X, k):
    """PCA 白化：返回 (mu, V, scale)，投影后各维方差为 1"""
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    # 用 SVD 求主成分，样本数远小于 d
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vt.shape[0])
    V = Vt[:k].T                                        # [d, k]
    scale = (S[:k] / np.sqrt(max(len(X) - 1, 1))).clip(min=1e-6)
    return mu, V, scale


def apply_whiten(P, X):
    mu, V, scale = P
    return ((X - mu) @ V) / scale


def fit_lda(X, y, k, lam=0.1, n_pre=128):
    """正则化 Fisher 判别（两步：先 PCA 降维，再求判别方向）

    样本数只有几十，必须在低维子空间里做，且 Sw 要加 shrinkage。
    """
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    npre = min(n_pre, Vt.shape[0], len(X) - 1)
    Vpre = Vt[:npre].T                                  # [d, npre]
    Z = Xc @ Vpre                                       # [N, npre]

    classes = np.unique(y)
    d = Z.shape[1]
    mu_all = Z.mean(axis=0, keepdims=True)
    Sw = np.zeros((d, d), dtype=np.float64)
    Sb = np.zeros((d, d), dtype=np.float64)
    for c in classes:
        Zc = Z[y == c]
        mc = Zc.mean(axis=0, keepdims=True)
        Dc = Zc - mc
        Sw += Dc.T @ Dc
        diff = mc - mu_all
        Sb += len(Zc) * (diff.T @ diff)
    tr = np.trace(Sw) / max(d, 1)
    Sw_reg = Sw + lam * max(tr, 1e-9) * np.eye(d)

    # 用 Sw^{-1/2} Sb Sw^{-1/2} 的对称化形式，比 cholesky 稳（Sw 可能近奇异）
    ev, EV = np.linalg.eigh(Sw_reg)
    ev = np.clip(ev, 1e-9, None)
    Sw_isqrt = (EV * (1.0 / np.sqrt(ev))) @ EV.T
    M = Sw_isqrt @ Sb @ Sw_isqrt
    M = (M + M.T) / 2
    vals, vecs = np.linalg.eigh(M)
    order = np.argsort(vals)[::-1]
    k = min(k, len(order))
    W = Sw_isqrt @ vecs[:, order[:k]]                    # [npre, k]
    return mu, Vpre @ W                                  # [d, k]


def evaluate(reps_test, items_test):
    return analyze_group(items_test, reps_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", default="results/reps")
    ap.add_argument("--corpus", default="data/p0_corpus.json")
    ap.add_argument("--out", default="results/proj.json")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    corpus = json.load(open(args.corpus, encoding="utf-8"))
    out = {"folds": args.folds, "groups": {}}

    layers = [0, 18, 36]          # L0 embedding / L18 塌缩区代表 / L36 最后一层
    for gname, items in corpus["groups"].items():
        n_items = len(items)
        v = len(items[0]["texts"])
        keys = np.array([group_key(it["id"]) for it in items])
        uniq = np.array(sorted(set(keys)))
        rng = np.random.RandomState(0)
        rng.shuffle(uniq)
        folds = np.array_split(uniq, args.folds)

        print(f"\n=== {gname}  items={n_items} variants={v} ===")
        gout = {}
        for layer in layers:
            flat = l2norm(load_group(args.reps, gname, layer))   # [n*v, d]
            reps = flat.reshape(n_items, v, -1)

            configs = [("none", None), ]
            for k in (8, 16, 32, 64, 128):
                configs.append(("whiten", k))
            for k, lam in itertools.product((4, 8, 16, 32), (0.01, 0.1, 1.0)):
                configs.append(("lda", (k, lam)))

            rows = []
            for name, cfg in configs:
                accs, snrs, margins = [], [], []
                ok = True
                for f in folds:
                    te_mask = np.isin(keys, f)
                    tr_idx = np.where(~te_mask)[0]
                    te_idx = np.where(te_mask)[0]
                    if len(te_idx) < 2:
                        ok = False
                        break
                    Xtr = reps[tr_idx].reshape(-1, reps.shape[-1])
                    ytr = np.repeat(tr_idx, v)

                    if name == "none":
                        def tf(X):
                            return X
                    elif name == "whiten":
                        P = fit_whiten(Xtr, cfg)
                        tf = lambda X, P=P: apply_whiten(P, X)
                    else:
                        k, lam = cfg
                        P = fit_lda(Xtr, ytr, k, lam)
                        mu, W = P
                        tf = lambda X, mu=mu, W=W: (X - mu) @ W

                    r_te = l2norm(tf(reps[te_idx].reshape(-1, reps.shape[-1])))
                    r_te = r_te.reshape(len(te_idx), v, -1)
                    m = evaluate(r_te, [items[i] for i in te_idx])
                    accs.append(m["acc_top1"])
                    snrs.append(m["snr_paired_median"])
                    margins.append(m["margin_mean"])
                if not ok:
                    continue
                rows.append({
                    "method": name, "cfg": cfg,
                    "acc": round(float(np.mean(accs)), 4),
                    "acc_std": round(float(np.std(accs)), 4),
                    "snr": round(float(np.mean(snrs)), 4),
                    "margin": round(float(np.mean(margins)), 4),
                })

            base = rows[0]
            # 选点规则写死：测试集 SNR 最低者；并列时取行序靠前者（配置表从简到繁）
            best = min(rows, key=lambda r: r["snr"])
            gout[layer] = {"baseline": base, "all": rows, "best": best}
            print(f"  L{layer:<2} base  acc={base['acc']:.3f} SNR={base['snr']:.3f} margin={base['margin']:+.4f}")
            print(f"  L{layer:<2} best  [{best['method']} {best['cfg']}] "
                  f"acc={best['acc']:.3f} SNR={best['snr']:.3f} margin={best['margin']:+.4f}")

        out["groups"][gname] = gout

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
