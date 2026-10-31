# -*- coding: utf-8 -*-
"""P0-4：锚点规模曲线 —— 跨骨干迁移能到 0.95 吗？

P0-3 的曲线在 n=32（训练折全部）就断了，而那个点还在单调上升。
本实验把锚点池扩大，看能不能逼近 oracle。

关键：**锚点必须来自不同项目**。
G2/G3/G4 是同一批 30 个项目的不同视角（短/长/双维度），如果把同项目的另一条
当锚点，等于把答案泄漏给映射——结果会虚高。所以按项目分组留出，
锚点池只取"测试项目之外"的项目，每个项目贡献 4 条 item（G2+G3+G4a+G4b）。

测试集：G4（同项目双维度，最难的一组）
锚点数量：8 / 16 / 32 / 48 / 64 / 96 条 item（每条 3 变体 = 3n 个配对向量）

用法：python anchor_scaling.py --out results/anchor_scale.json
"""
import argparse
import json
import os

import numpy as np

from run_probe import analyze_group, l2norm
from train_proj import fit_lda
from cross_backbone import ridge_dual, truncate_pad


def get(store, gname, layer, pool="mean"):
    z = np.load(os.path.join(store, f"{gname}.npz"))
    return z[pool].astype(np.float32)[layer]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repsA", default="results/reps")
    ap.add_argument("--repsB", default="results/reps_bb")
    ap.add_argument("--corpus", default="data/p0_corpus.json")
    ap.add_argument("--out", default="results/anchor_scale.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--anchors", default="8,16,32,48,64,96")
    ap.add_argument("--repeats", type=int, default=3, help="每个 n 重复抽样次数")
    args = ap.parse_args()

    corpus = json.load(open(args.corpus, encoding="utf-8"))
    g2, g3, g4 = (corpus["groups"][k] for k in
                  ("G2_same_short", "G3_same_long", "G4_same_project"))
    v = 3
    n_proj = len(g2)
    anchor_ns = [int(x) for x in args.anchors.split(",")]
    L = args.layer

    keys = ("G2_same_short", "G3_same_long", "G4_same_project")
    # 注意：缓存里存的是「文本行」[n_items*v, d]，要先摊回 [n_items, v, d]
    rawA = {k: l2norm(get(args.repsA, k, L)) for k in keys}
    rawB = {k: l2norm(get(args.repsB, k, L)) for k in keys}
    repA = {k: rawA[k].reshape(-1, v, rawA[k].shape[-1]) for k in keys}
    repB = {k: rawB[k].reshape(-1, v, rawB[k].shape[-1]) for k in keys}
    dA = rawA["G4_same_project"].shape[-1]
    dB = rawB["G4_same_project"].shape[-1]

    # 项目级分组，保证锚点与测试项目不重叠
    proj_ids = np.arange(n_proj)
    rng = np.random.RandomState(0)
    rng.shuffle(proj_ids)
    folds = np.array_split(proj_ids, args.folds)

    curve = {n: [] for n in anchor_ns}
    oracle_ms, b0_ms = [], []

    for f in folds:
        te_flat = sorted([2 * i + s for i in f for s in (0, 1)])   # G4 的 item 下标
        tr_proj = [i for i in range(n_proj) if i not in set(f.tolist())]
        if not te_flat:
            continue

        # 测试集
        XA_te = repA["G4_same_project"][te_flat]
        XB_te = repB["G4_same_project"][te_flat]
        n_te = len(te_flat)
        items_te = [g4[i] for i in te_flat]

        # 训练折（G4 的训练项目）用于训 LDA
        tr_g4 = sorted([2 * i + s for i in tr_proj for s in (0, 1)])
        XA_tr = repA["G4_same_project"][tr_g4].reshape(-1, dA)
        XB_tr = repB["G4_same_project"][tr_g4].reshape(-1, dB)
        y_tr = np.repeat(np.arange(len(tr_g4)), v)

        muA, WA = fit_lda(XA_tr, y_tr, 16, 0.01)
        muB, WB = fit_lda(XB_tr, y_tr, 16, 0.01)

        # a：在 B 上直接训（= 在目标骨干上重训记忆包）
        rB = l2norm((XB_te.reshape(-1, dB) - muB) @ WB)
        oracle_ms.append(analyze_group(items_te, rB.reshape(n_te, v, -1))["margin_mean"])

        r = l2norm((truncate_pad(XB_te.reshape(-1, dB), dA) - muA) @ WA)
        b0_ms.append(analyze_group(items_te, r.reshape(n_te, v, -1))["margin_mean"])

        # 锚点池：训练折的每个项目贡献 G2 / G3 / G4a / G4b 四条
        poolA, poolB = [], []
        for i in tr_proj:
            poolA.append(repA["G2_same_short"][i])
            poolB.append(repB["G2_same_short"][i])
            poolA.append(repA["G3_same_long"][i])
            poolB.append(repB["G3_same_long"][i])
            for s in (0, 1):
                poolA.append(repA["G4_same_project"][2 * i + s])
                poolB.append(repB["G4_same_project"][2 * i + s])
        poolA = np.stack(poolA)          # [n_pool_items, v, d]
        poolB = np.stack(poolB)
        n_pool = len(poolA)

        for n in anchor_ns:
            k = min(n, n_pool)
            accs = []
            for _ in range(args.repeats):
                sel = rng.choice(n_pool, size=k, replace=False)
                Aid = poolA[sel].reshape(-1, dA)
                Bid = poolB[sel].reshape(-1, dB)
                M = ridge_dual(Bid, Aid, lam=0.1)
                r = l2norm((XB_te.reshape(-1, dB) @ M - muA) @ WA)
                accs.append(analyze_group(items_te, r.reshape(n_te, v, -1))["margin_mean"])
            curve[n].append(float(np.mean(accs)))

    o = float(np.mean(oracle_ms))
    b0 = float(np.mean(b0_ms))
    print(f"oracle_B margin = {o:+.4f}   b0 = {b0:+.4f}")
    print(f"{'n_anchor':>8s} {'margin':>9s} {'keep/oracle':>12s}")
    out = {"layer": L, "oracle_margin": round(o, 4), "b0_margin": round(b0, 4), "curve": {}}
    for n in anchor_ns:
        mm = float(np.mean(curve[n]))
        out["curve"][n] = {"margin": round(mm, 4), "ratio": round(mm / o, 4) if o else None}
        print(f"{n:>8d} {mm:>+9.4f} {mm/o:>12.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
