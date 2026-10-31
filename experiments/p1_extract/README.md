# P1：记忆抽取器 —— 从对话到可检索条目

## 为什么这一步最重要

P0 / P0-2 / P0-3 / P0-4 验证了"记忆怎么存、怎么查、怎么跨基座搬"，
但那些实验里的记忆条目**全是手写的**。

真实系统必须从对话里自动产出条目。这一环失败，后面再好的检索也没有输入。
本目录就是给这一环出考题。

## 任务定义

输入：一段 4–8 轮的多轮对话（含与主题无关的闲聊噪声）
输出：若干结构化记忆条目

```json
{"entity": "星轨", "dimension": "数据库选型",
 "value": "PostgreSQL 14 单主加流复制", "kind": "decision"}
```

字段约束直接来自 P0 的实测结论：

| 字段 | 要求 | 依据 |
|---|---|---|
| `entity` | 具体名称（项目名/产品名/人名），不能是"这个""那个" | P0：剥掉锚词后召回从 1.000 掉到 **0.283** |
| `dimension` | 说明是哪一方面 | 同上，同项目多条目必须靠维度区分 |
| `value` | 具体取值，不能写"按之前说的" | 同上 |
| `kind` | decision / fact / preference / plan | 用于下游分层（决策进检索、偏好进 LoRA） |

## 两种抽取器

| 模式 | 做法 | 目的 |
|---|---|---|
| `rule` | 纯正则，**故意不用实体表** | 看规则的天花板在哪 |
| `llm` | Qwen3-4B-Instruct-2507，few-shot，贪心解码 | 看本地 4B 够不够 |

LLM 版的 prompt 把上面三条锚词规范写成了硬规则，并要求"闲聊与临时事务（会议时间、天气、午饭）一律不抽取"。

## 三层评估

| 层 | 指标 | 含义 |
|---|---|---|
| 抽取 | 召回 / 精确 | gold 被抽到多少、抽出的有多少有用 |
| **规范** | **锚词完整率** | entity + dimension + value 三者都具体的比例。**这是硬门槛** |
| 检索 | top-1 命中率 / margin | 条目入库后，用后续追问能不能查回来 |

检索表示用 Qwen3-4B 的 **L0 + L36 拼接**（两处都是 P0 认定的可用层），L2 归一化后余弦。

## 数据

- 20 段对话，来自 20 个不同项目，每段 121–184 字、4–8 轮
- 每段 2 条 gold（一条决策 + 一条"排除项"事实）
- 每段配 1 条后续追问，如"星轨那个数据平台的数据库最后到底用的哪个？"
- 噪声按固定池抽样且不重复，插入位置随机
- 种子 `20261101`，完全可复现

## 运行

```bash
PY="D:/Doubao-works/训练前置/train_env/Scripts/python.exe"
$PY gen_dialogues.py --out data/p1_dialogues.json
$PY extract_memory.py --mode rule --out results/extracted_rule.json
$PY extract_memory.py --mode llm  --out results/extracted_llm.json
$PY eval_e2e.py --extracted results/extracted_rule.json --out results/eval_rule.json
$PY eval_e2e.py --extracted results/extracted_llm.json  --out results/eval_llm.json
```

## 已知局限

1. **对话是模板生成的**，比真实对话规整得多：主题集中、没有指代回跳、没有多人插话。真实场景召回会更低。
2. **gold 只有 2 条/段**，粒度粗，无法评估"过度抽取"的细粒度错误。
3. **检索库只有 20–40 条**，量级远小于真实系统（成百上千条），所以检索命中率会偏乐观。
4. 评估用"value 前 12 字互相包含"做匹配，对改写的容忍度有限。
