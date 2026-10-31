# -*- coding: utf-8 -*-
"""P0-3：投影空间里的跨骨干标定 —— MSP 还有没有救？

背景
----
项目原有结论：MSP 跨骨干 b/a = 0.00–0.04 ≈ 机会水平，机制是
「标定误差夹角 12–15° ≈ 类间夹角 13.2°」，信噪比 ≈1。

P0-2 发现「塌缩不是信息丢失，判别投影能把 1–4° 的类间夹角拉开」。
于是有了一个新假设：

    标定之所以失败，可能只是因为**在坏空间里做标定**。
    如果先把两边各投影到判别性好的空间再标定，误差的相对影响会小很多。

本实验直接检验这个假设，并给出 b/a 随锚点数量的曲线。

实验臂
------
oracle_B   在目标骨干 B 上直接训 LDA（= 在 B 上重训记忆包）→ 这是 a，即上界
A_local    在源骨干 A 上训 LDA → 参考
b0_random  A 的投影直接吃 B 的截断表示（无意义映射）→ 噪声对照，应接近 0
b_ridge    锚点学 M: h_B → h_A（原始空间 ridge），再用 A 的 LDA
b_white    各自 PCA 白化到同维后学 M
b_procrust 白化到同维后正交 Procrustes

判据：b/a ≥ 0.95 才叫"可迁移"；本实验还报 SNR，因为准确率会被小测试集掩盖。

用法
----
python cross_backbone.py --repsA results/reps --repsB results/reps_bb \
    --tagA qwen3_4b --tagB qwen25_3b --out results/cross.json
"""
import argparse
import json
import os

import numpy as np

from run_probe import analyze_group, l2norm
from train_proj import fit_lda, fit_whiten, apply_whiten, group_key


def get(store, gname, layer, pool="mean"):
    z = np.load(os.path.join(store, f"{gname}.npz"))
    return z[pool].astype(np.float32)[layer]


def ridge_dual(X, Y, lam=0.1):
    """对偶形式 ridge：M 使 X @ M ≈ Y。X:[n,dB], Y:[n,dA] → M:[dB,dA]
    n 很小（几十），对偶形式比求 d×d 逆便宜得多。"""
    K = X @ X.T
    A = np.linalg.solve(K + lam * np.eye(len(X)), Y)
    return X.T @ A


def procrustes(X, Y):
    """正交 Procrustes：要求同维。返回 R 使 X @ R ≈ Y"""
    U, _, Vt = np.linalg.svd(X.T @ Y)
    return U @ Vt


def truncate_pad(x, d):
    if x.shape[-1] >= d:
        return x[..., :d]
    pad = np.zeros(x.shape[:-1] + (d - x.shape[-1],), dtype=x.dtype)
    return np.concatenate([x, pad], axis=-1)


def score(r_te, items_te):
    return analyze_group(items_te, r_te)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repsA", default="results/reps")
    ap.add_argument("--repsB", default="results/reps_bb")
    ap.add_argument("--corpus", default="data/p0_corpus.json")
    ap.add_argument("--out", default="results/cross.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--layers", default="0,36")
    ap.add_argument("--anchors", default="4,8,16,32")
    ap.add_argument("--ood-group", default="G1_cross_same_len",
                    help="异分布锚点的来源组")
    args = ap.parse_args()

    corpus = json.load(open(args.corpus, encoding="utf-8"))
    groups = corpus["groups"]
    layers = [int(x) for x in args.layers.split(",")]
    anchor_ns = [int(x) for x in args.anchors.split(",")]

    # 异分布锚点池：从 G1 取，文本与目标组不同域
    ood = groups[args.ood_group]
    ood_texts = [t for it in ood for t in it["texts"]]

    out = {"folds": args.folds, "layers": layers, "anchors": anchor_ns,
           "ood_group": args.ood_group, "groups": {}}

    for gname in ("G2_same_short", "G3_same_long", "G4_same_project"):
        items = groups[gname]
        n_items, v = len(items), len(items[0]["texts"])
        keys = np.array([group_key(it["id"]) for it in items])
        uniq = np.array(sorted(set(keys)))
        rng = np.random.RandomState(0)
        rng.shuffle(uniq)
        folds = np.array_split(uniq, args.folds)

        print(f"\n===== {gname}  items={n_items} =====")
        gout = {}
        for layer in layers:
            repA = l2norm(get(args.repsA, gname, layer)).reshape(n_items, v, -1)
            repB = l2norm(get(args.repsB, gname, layer)).reshape(n_items, v, -1)
            dA, dB = repA.shape[-1], repB.shape[-1]

            # 异分布锚点：同批文本过两个骨干
            oodA = l2norm(get(args.repsA, args.ood_group, layer))
            oodB = l2norm(get(args.repsB, args.ood_group, layer))

            rows = {"oracle_B": [], "A_local": [], "b0_random": [],
                    "ridge_ood": {n: [] for n in anchor_ns},
                    "ridge_id": {n: [] for n in anchor_ns},
                    "self_id": {n: [] for n in anchor_ns},
                    "white_id": {n: [] for n in anchor_ns},
                    "procrust_id": {n: [] for n in anchor_ns}}

            for f in folds:
                te_mask = np.isin(keys, f)
                tr_idx = np.where(~te_mask)[0]
                te_idx = np.where(te_mask)[0]
                if len(te_idx) < 2:
                    continue
                items_te = [items[i] for i in te_idx]

                XA_tr = repA[tr_idx].reshape(-1, dA)
                XB_tr = repB[tr_idx].reshape(-1, dB)
                y_tr = np.repeat(tr_idx, v)
                XA_te = repA[te_idx].reshape(-1, dA)
                XB_te = repB[te_idx].reshape(-1, dB)

                # --- a：在 B 上直接训 LDA
                muB, WB = fit_lda(XB_tr, y_tr, 16, 0.01)
                m = score(l2norm((XB_te - muB) @ WB).reshape(len(te_idx), v, -1), items_te)
                rows["oracle_B"].append(m)

                # --- A 本地
                muA, WA = fit_lda(XA_tr, y_tr, 16, 0.01)
                m = score(l2norm((XA_te - muA) @ WA).reshape(len(te_idx), v, -1), items_te)
                rows["A_local"].append(m)

                # --- b0：两个噪声底
                # (a) 截断/补零：保留 B 的语义但用 A 的坐标系 → 仍有相似度结构
                r = l2norm((truncate_pad(XB_te, dA) - muA) @ WA)
                rows["b0_random"].append(score(r.reshape(len(te_idx), v, -1), items_te))

                for n_anchor in anchor_ns:
                    sel = rng.choice(tr_idx, size=min(n_anchor, len(tr_idx)), replace=False)
                    Aid = repA[sel].reshape(-1, dA)
                    Bid = repB[sel].reshape(-1, dB)

                    # --- A→A 自迁移：管线正确性的 sanity check，应 ≈ A_local
                    Ms = ridge_dual(Aid, Aid, lam=0.1)
                    r = l2norm((XA_te @ Ms - muA) @ WA)
                    rows["self_id"][n_anchor].append(
                        score(r.reshape(len(te_idx), v, -1), items_te))

                    # 同分布锚点 + 原始空间 ridge
                    M = ridge_dual(Bid, Aid, lam=1.0)
                    r = l2norm((XB_te @ M - muA) @ WA)
                    rows["ridge_id"][n_anchor].append(
                        score(r.reshape(len(te_idx), v, -1), items_te))

                    # 异分布锚点 + 原始空间 ridge
                    kk = min(n_anchor * v, len(oodA))
                    Mo = ridge_dual(oodB[:kk], oodA[:kk], lam=1.0)
                    r = l2norm((XB_te @ Mo - muA) @ WA)
                    rows["ridge_ood"][n_anchor].append(
                        score(r.reshape(len(te_idx), v, -1), items_te))

                    # 白化到共同 k 维后 ridge / procrustes
                    # 注意：桥接的输出必须回到 A 的**原始空间**，才能减 muA、乘 WA。
                    # 所以目标是未白化的 Aid，而不是白化后的 Za。
                    kd = 64
                    P = fit_whiten(Bid, kd)
                    Q = fit_whiten(Aid, kd)
                    Zb = apply_whiten(P, Bid)
                    Za_w = apply_whiten(Q, Aid)
                    kb = min(Zb.shape[1], Za_w.shape[1])
                    Zb, Za_w = Zb[:, :kb], Za_w[:, :kb]

                    Mw = ridge_dual(Zb, Aid, lam=0.1)
                    r = l2norm((apply_whiten(P, XB_te)[:, :kb] @ Mw - muA) @ WA)
                    rows["white_id"][n_anchor].append(
                        score(r.reshape(len(te_idx), v, -1), items_te))

                    Rp = procrustes(Zb, Za_w)
                    Mp = ridge_dual(Zb @ Rp, Aid, lam=0.1)
                    r = l2norm((apply_whiten(P, XB_te)[:, :kb] @ Rp @ Mp - muA) @ WA)
                    rows["procrust_id"][n_anchor].append(
                        score(r.reshape(len(te_idx), v, -1), items_te))

            def agg(ms):
                if not ms:
                    return None
                return {"acc": round(float(np.mean([m["acc_top1"] for m in ms])), 4),
                        "snr": round(float(np.mean([m["snr_paired_median"] for m in ms])), 4),
                        "margin": round(float(np.mean([m["margin_mean"] for m in ms])), 4)}

            block = {
                "oracle_B": agg(rows["oracle_B"]),
                "A_local": agg(rows["A_local"]),
                "b0_random": agg(rows["b0_random"]),
                "ridge_ood": {n: agg(rows["ridge_ood"][n]) for n in anchor_ns},
                "ridge_id": {n: agg(rows["ridge_id"][n]) for n in anchor_ns},
                "self_id": {n: agg(rows["self_id"][n]) for n in anchor_ns},
                "white_id": {n: agg(rows["white_id"][n]) for n in anchor_ns},
                "procrust_id": {n: agg(rows["procrust_id"][n]) for n in anchor_ns},
            }
            gout[layer] = block

            o = block["oracle_B"]
            print(f"  L{layer}: oracle_B acc={o['acc']:.3f} SNR={o['snr']:.3f} | "
                  f"A_local acc={block['A_local']['acc']:.3f} | "
                  f"b0 acc={block['b0_random']['acc']:.3f}")
            for tag in ("self_id", "ridge_id", "white_id", "ridge_ood"):
                parts = []
                for n in anchor_ns:
                    c = block[tag][n]
                    r = (c["acc"] / o["acc"]) if (c and o["acc"] > 0) else float("nan")
                    parts.append(f"n{n}:{c['acc']:.2f}/b/a={r:.2f}")
                print(f"       {tag:11s} " + "  ".join(parts))

        out["groups"][gname] = gout

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
