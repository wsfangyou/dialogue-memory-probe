# -*- coding: utf-8 -*-
"""把 P0 语料编码到第二个骨干上，供跨骨干实验使用。

与 run_probe.py 的区别：只编码、不评估，输出格式与 results/reps/ 一致，
方便 cross_backbone.py 直接对读两个骨干的表示。

用法：
python encode_bb.py --model D:/.../Qwen2.5-3B-Instruct --tag qwen25_3b
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def encode_texts(model, tok, texts, device, bs=8, max_len=512):
    mean_reps, last_reps = None, None
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to(device)
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states
            mask = enc["attention_mask"].unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1e-6)
            m = torch.stack([(h * mask).sum(dim=1) / denom for h in hs]).float().cpu().numpy()
            idx = (enc["attention_mask"].sum(dim=1) - 1).clamp(min=0)
            l = torch.stack([h[torch.arange(h.size(0)), idx] for h in hs]).float().cpu().numpy()
            mean_reps = m if mean_reps is None else np.concatenate([mean_reps, m], axis=1)
            last_reps = l if last_reps is None else np.concatenate([last_reps, l], axis=1)
            del out, hs
            torch.cuda.empty_cache()
    return mean_reps, last_reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="输出文件名后缀")
    ap.add_argument("--corpus", default="data/p0_corpus.json")
    ap.add_argument("--outdir", default="results/reps_bb")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    data = json.load(open(args.corpus, encoding="utf-8"))
    t0 = time.time()
    print(f"loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(args.model, dtype=torch.float16).to(args.device).eval()
    print(f"loaded {time.time()-t0:.1f}s  vram={torch.cuda.memory_allocated()/2**30:.2f} GiB  "
          f"layers={model.config.num_hidden_layers} hidden={model.config.hidden_size}")

    os.makedirs(args.outdir, exist_ok=True)
    for gname, items in data["groups"].items():
        texts = [t for it in items for t in it["texts"]]
        m, l = encode_texts(model, tok, texts, args.device, bs=args.bs)
        np.savez_compressed(os.path.join(args.outdir, f"{gname}.npz"),
                            mean=m.astype(np.float16), last=l.astype(np.float16))
        print(f"  {gname}: {m.shape} -> {args.outdir}/{gname}.npz")

    meta = {"model": args.model, "tag": args.tag,
            "n_layers": model.config.num_hidden_layers,
            "hidden": model.config.hidden_size,
            "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(args.outdir, f"_meta_{args.tag}.json"), "w"), indent=1)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
