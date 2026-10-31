# -*- coding: utf-8 -*-
"""P0 探针：同骨干、自然语言、段落级记忆的可区分性

要回答的问题
------------
把一段对话压成一个向量、再从一堆记忆里把它找回来，前提是：
  **同一件事的不同说法（类内）之间的距离，明显小于不同事之间的距离（类间）。**

MSP 跨骨干失败的真实机制就是这个比值约等于 1（标定误差夹角 12–15° ≈ 类间夹角 13.2°）。
本脚本把同一个判据搬到「同骨干 + 自然语言」场景下重测，此时误差源不是跨骨干标定，
而是**语义等价的不同说法**。

四个量（逐层）
--------------
θ_inter  不同记忆条目之间的平均夹角（类间可区分性，越大越好）
θ_intra  同一记忆条目不同说法之间的平均夹角（类内可变性，越小越好）
SNR      θ_inter / θ_intra，判据 ≥ 2 才可能可靠检索（误差须 ≲ 类间的一半）
acc      top-1 检索准确率：s1 建库，s2/s3 查询；随机基线 = 1/N
d_prime  标准可分性 (μ_inter − μ_intra) / sqrt((σ²_inter + σ²_intra)/2)，>2 为好

用法
----
python run_probe.py --corpus data/p0_corpus.json --out results/probe_qwen3_4b.json
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def l2norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def cos_to_deg(c):
    return math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))


def pairwise_angles(mat):
    """mat: [n, d] 已归一化 → 返回所有两两夹角（度）的数组（上三角）"""
    sim = mat @ mat.T
    iu = np.triu_indices(len(mat), k=1)
    sims = sim[iu]
    return np.array([cos_to_deg(c) for c in sims])


def encode_texts(model, tok, texts, device, bs=8, max_len=512):
    """返回 (n_layers, n, d) 的 mean-pooled 与 last-token 两套表示"""
    mean_reps, last_reps = None, None
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(device)
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states  # tuple of [B, T, D]
            mask = enc["attention_mask"].unsqueeze(-1).float()          # [B,T,1]
            denom = mask.sum(dim=1).clamp(min=1e-6)                     # [B,1]
            m = torch.stack([(h * mask).sum(dim=1) / denom for h in hs]).float().cpu().numpy()
            idx = (enc["attention_mask"].sum(dim=1) - 1).clamp(min=0)
            l = torch.stack([h[torch.arange(h.size(0)), idx] for h in hs]).float().cpu().numpy()
            mean_reps = m if mean_reps is None else np.concatenate([mean_reps, m], axis=1)
            last_reps = l if last_reps is None else np.concatenate([last_reps, l], axis=1)
            del out, hs
            torch.cuda.empty_cache()
    return mean_reps, last_reps


def analyze_group(items, reps):
    """reps: [n_items, n_variants, d]，已归一化。

    关键：类间距离必须用**最近邻**，不能用类中心。
    用中心会做平均、把个体差异抹平，系统性低估类间夹角——这会让 SNR 与
    top-1 准确率互相矛盾。检索真正关心的是"最容易混淆的那一个"。
    """
    n, v, _ = reps.shape
    flat = reps.reshape(n * v, -1)
    N = n * v

    sim = np.clip(flat @ flat.T, -1.0, 1.0)
    ang = np.degrees(np.arccos(sim))

    item_id = np.repeat(np.arange(n), v)
    same_item = item_id[:, None] == item_id[None, :]
    intra_mask = same_item & ~np.eye(N, dtype=bool)
    inter_mask = ~same_item

    intra = ang[intra_mask]

    tmp = np.where(inter_mask, sim, -np.inf)
    nearest_inter = np.degrees(np.arccos(np.clip(tmp.max(axis=1), -1.0, 1.0)))

    # top-1 检索 + margin（按变体分别统计，便于看"信息量递减"）
    bank = reps[:, 0, :]
    hits, total, margins = 0, 0, []
    acc_by_variant = []
    ratios = []          # 配对信噪比：正确夹角 / 最近错误夹角（<1 才可分）
    for q in range(1, v):
        s = reps[:, q, :] @ bank.T                          # [n, n]
        diag = np.diag(s).copy()
        s2 = np.where(np.eye(n, dtype=bool), -np.inf, s)
        wrong = s2.max(axis=1)
        margins.extend((diag - wrong).tolist())
        # 角度版配对量：与"类间夹角 ÷ 误差夹角 ≥ 2"的判据同构
        ang_c = np.degrees(np.arccos(np.clip(diag, -1, 1)))
        ang_w = np.degrees(np.arccos(np.clip(wrong, -1, 1)))
        ratios.extend((ang_c / np.maximum(ang_w, 1e-6)).tolist())
        qhit = int((s.argmax(axis=1) == np.arange(n)).sum())
        hits += qhit
        total += n
        acc_by_variant.append(round(qhit / n, 4))
    margins = np.array(margins)
    ratios = np.array(ratios)

    mu_i, mu_e = float(intra.mean()), float(nearest_inter.mean())
    sd_i, sd_e = float(intra.std()), float(nearest_inter.std())
    dprime = (mu_e - mu_i) / math.sqrt(max((sd_i ** 2 + sd_e ** 2) / 2, 1e-9))

    return {
        "n_items": n,
        "theta_intra_mean": round(mu_i, 3),
        "theta_intra_p90": round(float(np.percentile(intra, 90)), 3),
        "theta_nearest_inter_mean": round(mu_e, 3),
        "theta_nearest_inter_p10": round(float(np.percentile(nearest_inter, 10)), 3),
        # snr_mean：均值口径的粗比值（不可靠，见 README 的坑）
        "snr_mean": round(mu_e / mu_i, 3) if mu_i > 1e-6 else None,
        # snr_paired：逐查询的「正确夹角 / 最近错误夹角」中位数，<1 才可分。
        # 判据：中位数 ≲ 0.5（等价于"误差夹角 ≲ 类间夹角的一半"）
        "snr_paired_median": round(float(np.median(ratios)), 4),
        "snr_paired_p90": round(float(np.percentile(ratios, 90)), 4),
        "sep_rate": round(float((ratios < 1).mean()), 4),
        "d_prime": round(float(dprime), 3),
        "acc_top1": round(hits / total, 4),
        "acc_by_variant": acc_by_variant,
        "acc_random": round(1.0 / n, 4),
        "margin_mean": round(float(margins.mean()), 4),
        "margin_min": round(float(margins.min()), 4),
        "margin_pos_rate": round(float((margins > 0).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/p0_corpus.json")
    ap.add_argument("--model", default="D:/Doubao-works/训练前置/02_基座模型/Qwen3-4B")
    ap.add_argument("--out", default="results/probe.json")
    ap.add_argument("--pool", default="both", choices=["mean", "last", "both"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    data = json.load(open(args.corpus, encoding="utf-8"))
    groups = data["groups"]

    t0 = time.time()
    print(f"loading {args.model} ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(args.model, dtype=torch.float16).to(args.device).eval()
    n_layers = model.config.num_hidden_layers
    print(f"loaded in {time.time()-t0:.1f}s, layers={n_layers}, "
          f"vram={torch.cuda.memory_allocated()/2**30:.2f} GiB")

    result = {"model": args.model, "n_layers": n_layers, "groups": {}}

    for gname, items in groups.items():
        texts = [t for it in items for t in it["texts"]]
        v = len(items[0]["texts"])
        print(f"\n[{gname}] {len(items)} items x {v} variants = {len(texts)} texts")
        mean_reps, last_reps = encode_texts(model, tok, texts, args.device, bs=args.bs)

        pools = {}
        if args.pool in ("mean", "both"):
            pools["mean"] = mean_reps
        if args.pool in ("last", "both"):
            pools["last"] = last_reps

        # 缓存表示，后续分析不必重跑模型
        os.makedirs("results/reps", exist_ok=True)
        np.savez_compressed(f"results/reps/{gname}.npz",
                            mean=mean_reps.astype(np.float16),
                            last=last_reps.astype(np.float16))

        gres = {}
        for pname, reps in pools.items():
            layer_res = []
            for L in range(reps.shape[0]):
                r = l2norm(reps[L]).reshape(len(items), v, -1)
                layer_res.append({"layer": L, **analyze_group(items, r)})
            gres[pname] = layer_res
            # 选"最优层"用 margin（鲁棒裕度），不用 acc —— 中层 acc 也能到 1.000，
            # 但 margin 只有 0.001，属于"数值可分、一扰就翻"，不能当可用层。
            best = max(layer_res, key=lambda x: x["margin_mean"])
            print(f"  {pname}: best layer={best['layer']:2d} "
                  f"acc={best['acc_top1']:.3f} per-variant={best['acc_by_variant']} "
                  f"SNR_pair={best['snr_paired_median']:.3f} "
                  f"margin={best['margin_mean']:+.4f} "
                  f"θ_intra={best['theta_intra_mean']:.1f}° "
                  f"θ_nearest={best['theta_nearest_inter_mean']:.1f}°")
        result["groups"][gname] = gres

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwritten {args.out}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
