# -*- coding: utf-8 -*-
"""P1：记忆抽取器 —— 从对话里产出结构化记忆条目

两种实现，用来对比"规则够不够"：

  rule  纯正则。故意不借助实体表，看模板语料上规则能到哪。
  llm   本地 Qwen3-4B-Instruct-2507，few-shot，贪心解码。
        prompt 里把 P0 得出的「锚词规范」写成了硬规则。

输出格式（两种实现一致）：
  {"id": "D01", "items": [{"entity","dimension","value","kind","raw"}...]}

用法：
  python extract_memory.py --mode rule --out results/extracted_rule.json
  python extract_memory.py --mode llm  --out results/extracted_llm.json
"""
import argparse
import json
import os
import re

# ---------------------------------------------------------------- 规则版

VALUE_PATTERNS = [
    re.compile(r"定的是([^。；\n]{2,60})"),
    re.compile(r"是([^。；\n]{2,60})。"),
    re.compile(r"用的是([^。；\n]{2,60})"),
    re.compile(r"存在([^。；\n]{2,60})"),
]
DIM_PATTERN = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,10})(?:这块|这个|的选型)")
ENT_PATTERN = re.compile(r"那个([\u4e00-\u9fa5A-Za-z0-9]{2,8})的")


def extract_rule(sample):
    turns = sample["turns"]
    user_text = " ".join(t["content"] for t in turns if t["role"] == "user")
    asst_text = " ".join(t["content"] for t in turns if t["role"] == "assistant")

    ent_m = ENT_PATTERN.search(user_text)
    dim_m = DIM_PATTERN.search(user_text)
    entity = ent_m.group(1) if ent_m else ""
    dimension = dim_m.group(1) if dim_m else ""

    value = ""
    for p in VALUE_PATTERNS:
        m = p.search(asst_text)
        if m:
            value = m.group(1).strip()
            break

    if not value:
        return []
    return [{"entity": entity, "dimension": dimension, "value": value,
             "kind": "decision", "raw": ""}]


# ---------------------------------------------------------------- LLM 版

SYSTEM = """你是记忆抽取器。从对话中抽取所有值得长期记住的决策、事实和偏好。

硬规则：
1. entity 必须是对话里出现的具体名称（项目名、产品名、人名），不能写"这个""那个"。
2. dimension 必须说明是哪一方面，例如"数据库选型""部署方式""上报策略"。
3. value 必须是具体取值，绝对不能写"按之前说的""那个方案""上次定的"。
4. 闲聊、临时事务（会议时间、天气、午饭）一律不抽取。
5. kind 只能取 decision / fact / preference / plan 之一。

示例输入：
用户：那个天枢的项目，缓存这块怎么定的？
助手：定的是 Redis Cluster 三主三从。Caffeine 本地缓存考虑过，但要多实例共享，所以没用。
输出：
{"entity": "天枢", "dimension": "缓存层", "value": "Redis Cluster 三主三从", "kind": "decision"}

只输出 JSON 行，一行一条，不要解释，不要代码块。"""


def transcript(sample):
    lines = []
    for t in sample["turns"]:
        who = "用户" if t["role"] == "user" else "助手"
        lines.append(f"{who}：{t['content']}")
    return "\n".join(lines)


def build_prompt(sample):
    return f"{SYSTEM}\n\n现在抽取下面这段对话：\n{transcript(sample)}\n输出："


def parse_jsonl(text):
    items = []
    for line in text.splitlines():
        line = line.strip().strip(",")
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:
            # 容错：截到最后一个 }
            try:
                o = json.loads(line[: line.rfind("}") + 1])
            except Exception:
                continue
        if not isinstance(o, dict):
            continue
        items.append({
            "entity": str(o.get("entity", "")).strip(),
            "dimension": str(o.get("dimension", "")).strip(),
            "value": str(o.get("value", "")).strip(),
            "kind": str(o.get("kind", "")).strip() or "unknown",
            "raw": line,
        })
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rule", "llm"], default="rule")
    ap.add_argument("--dialogues", default="data/p1_dialogues.json")
    ap.add_argument("--out", default="results/extracted.json")
    ap.add_argument("--model", default="D:/Doubao-works/训练前置/02_基座模型/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    samples = json.load(open(args.dialogues, encoding="utf-8"))["samples"]
    out = []

    if args.mode == "rule":
        for s in samples:
            out.append({"id": s["id"], "items": extract_rule(s)})
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"loading {args.model}")
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16).to(args.device).eval()
        for s in samples:
            msgs = [{"role": "user", "content": build_prompt(s)}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
            enc = tok(text, return_tensors="pt").to(args.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new,
                                     do_sample=False, temperature=None, top_p=None, top_k=None,
                                     pad_token_id=tok.eos_token_id)
            resp = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            items = parse_jsonl(resp)
            out.append({"id": s["id"], "items": items, "raw_resp": resp[:600]})
            print(f"  {s['id']}: {len(items)} 条")
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n = sum(len(o["items"]) for o in out)
    print(f"written {args.out}: {len(out)} 段, 共抽出 {n} 条")


if __name__ == "__main__":
    main()
