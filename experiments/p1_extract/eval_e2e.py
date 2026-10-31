# -*- coding: utf-8 -*-
"""P1：端到端评估 —— 对话 → 抽取 → 入库 → 检索

三层指标
--------
1. 抽取层：召回（gold 被抽到多少）、精确（抽出的有多少有用）
2. 规范层：**锚词完整率** —— entity / dimension / value 三者都具体有效的比例
   （P0 已证：缺锚词的条目在任何模型上都检索不回来，所以这一层是硬门槛）
3. 检索层：把抽出的条目建成记忆库，用后续追问查 top-1 命中率

检索表示：Qwen3-4B 的 hidden，逐层配置分别评估（L0 / L18 / L36）。
不要拼接多层——不同层的向量范数量级不同，拼完做 L2 归一化会让其中一层淹没另一层，
第一版就是这么崩的（top-1 0.05）。

用法：
  python eval_e2e.py --extracted results/extracted_rule.json --out results/eval_rule.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "p0_memory"))
from run_probe import encode_texts, l2norm  # noqa: E402

GENERIC = {"", "这个", "那个", "它", "就是", "上面", "之前", "上次", "那边", "这块",
           "那个事", "方案", "东西", "事情", "按之前说的"}


def is_specific(s):
    s = (s or "").strip()
    if not s or s in GENERIC or len(s) < 2:
        return False
    return not any(g in s for g in ("之前说的", "上次定", "那个方案"))


def norm(s):
    return (s or "").replace(" ", "").replace("　", "").lower()


def gold_hit(gold, item):
    """gold 是否被该抽出条目命中：value 主体互相包含即可（value 是核心信息）"""
    gv, iv = norm(gold["value"]), norm(item["value"])
    if not gv or not iv:
        return False
    core = gv[:12]
    return core in iv or iv[:12] in gv


def item_text(it):
    return f"{it['entity']} | {it['dimension']} | {it['value']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", default="data/p1_dialogues.json")
    ap.add_argument("--extracted", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="D:/Doubao-works/训练前置/02_基座模型/Qwen3-4B")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dlg = json.load(open(args.dialogues, encoding="utf-8"))["samples"]
    ext = {o["id"]: o["items"] for o in json.load(open(args.extracted, encoding="utf-8"))}

    # ---------- 1. 抽取质量 ----------
    n_gold = n_gold_hit = n_item = n_item_useful = n_specific = 0
    n_primary = n_primary_hit = 0
    per_sample = []
    for s in dlg:
        items = ext.get(s["id"], [])
        hit_gold = 0
        for gi, g in enumerate(s["gold"]):
            n_gold += 1
            ok = any(gold_hit(g, it) for it in items)
            if ok:
                n_gold_hit += 1
                hit_gold += 1
            if gi == 0:                      # primary = 决策条目，是一定要抽到的
                n_primary += 1
                if ok:
                    n_primary_hit += 1
        useful = 0
        for it in items:
            n_item += 1
            if any(gold_hit(g, it) for g in s["gold"]):
                n_item_useful += 1
                useful += 1
            if is_specific(it["entity"]) and is_specific(it["dimension"]) and is_specific(it["value"]):
                n_specific += 1
        per_sample.append({"id": s["id"], "n_gold": len(s["gold"]), "gold_hit": hit_gold,
                           "n_item": len(items), "n_useful": useful, "items": items})

    recall = n_gold_hit / max(n_gold, 1)
    recall_primary = n_primary_hit / max(n_primary, 1)
    precision = n_item_useful / max(n_item, 1)
    anchor_rate = n_specific / max(n_item, 1)

    print(f"[抽取] 全部 gold {n_gold_hit}/{n_gold} → 召回 {recall:.3f}")
    print(f"[抽取] 决策条目 {n_primary_hit}/{n_primary} → 召回 {recall_primary:.3f}  ← 关键口径")
    print(f"[抽取] 抽出 {n_item} 条，有用 {n_item_useful} → 精确 {precision:.3f}")
    print(f"[规范] 锚词完整 {n_specific}/{n_item} → {anchor_rate:.3f}")

    # ---------- 2. 端到端检索 ----------
    import torch
    from transformers import AutoModel, AutoTokenizer
    print(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(args.model, dtype=torch.float16).to(args.device).eval()

    mem_texts, mem_ids = [], []
    for s in dlg:
        for it in ext.get(s["id"], []):
            mem_texts.append(item_text(it))
            mem_ids.append(s["id"])
    queries = [s["query"] for s in dlg]

    results_by_layer = {}
    if mem_texts:
        reps, _ = encode_texts(model, tok, mem_texts + queries, args.device)
        for tag, L in (("L0", 0), ("L18", 18), ("L36", 36)):
            R = l2norm(reps[L])
            M, Q = R[:len(mem_texts)], R[len(mem_texts):]
            sim = Q @ M.T
            pred = sim.argmax(axis=1)
            hits = sum(1 for i, p in enumerate(pred) if mem_ids[p] == dlg[i]["id"])
            margins = []
            for i in range(len(pred)):
                row = sim[i]
                cmask = np.array([m == dlg[i]["id"] for m in mem_ids])
                if cmask.any() and (~cmask).any():
                    margins.append(float(row[cmask].max() - row[~cmask].max()))
            acc = hits / len(dlg)
            mg = float(np.mean(margins)) if margins else 0.0
            miss = [dlg[i]["id"] for i, p in enumerate(pred) if mem_ids[p] != dlg[i]["id"]]
            results_by_layer[tag] = {"acc_top1": round(acc, 4), "margin": round(mg, 4),
                                     "miss": miss}
            print(f"[检索-{tag}] top-1 = {acc:.3f} ({hits}/{len(dlg)})  margin = {mg:+.4f}"
                  + (f"  未命中 {len(miss)}" if miss else ""))
    else:
        print("没有抽出任何条目，检索跳过")

    best = max(results_by_layer, key=lambda k: results_by_layer[k]["margin"]) if results_by_layer else None
    res = {"recall": round(recall, 4), "recall_primary": round(recall_primary, 4),
           "precision": round(precision, 4), "anchor_rate": round(anchor_rate, 4),
           "n_items": n_item, "n_gold": n_gold, "best_layer": best,
           "by_layer": results_by_layer, "per_sample": per_sample}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
