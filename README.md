# Dialogue Memory Probe

**[English](#english)** · [中文](#chinese)

---

<a id="english"></a>

## English

**Turning conversations into retrievable long-term memory — a complete empirical report**

No training, no fine-tuning, no touching the weights. Structured memory entries + embedding-layer representations + cosine retrieval turn dialogue into memory the model can look up. End-to-end retrieval over 20 conversations: **top-1 = 1.000**.

Swapping the backbone doesn't require retraining either: **96 in-distribution anchors** and a single linear calibration carry the entire memory layer across, retaining **96%** of performance.

---

### What this project answers

The starting point is a very natural idea:

> Can we treat every conversation as one LoRA training step, so the model understands me better the more we talk?

Five experiments and roughly 300 evaluations later, this repository takes that question apart, quantifies it, and arrives at an answer **different from where it started**:

**Don't train conversations into the weights.** Write them as structured entries carrying anchor terms, and put them in a pluggable memory layer.

The "memory layer" route had previously been declared dead in a related project (cross-backbone transfer b/a ≈ 0). This project found **that verdict was a measurement error** — accuracy had long since saturated on that corpus, and under a margin-based criterion the same system reaches 96% once it has enough anchors.

---

### Core findings

| Goal | Verdict | Key numbers |
|---|---|---|
| Remember facts and agreements from dialogue | ✅ Works, verified end to end | Extraction recall 1.000, retrieval top-1 **1.000** |
| Learn speaking style / working habits | ✅ Works, but backbone must be ≥4B | General capability **0.861×** on 4B (net gain); **1.338×** on 0.83B (34% damage) |
| Automatic scheduling (what stays resident) | 🟡 Mechanism holds, not wired to a real model | Toy load tax p95 145→30 ms |
| Make it smarter | ❌ No — and adding skills makes it worse | 4B reasoning probe 0.533 → **0.400** after mounting a skill |
| **One LoRA per conversation** | ❌ Clearly infeasible | See below |

#### Why "train one LoRA per conversation" doesn't work

Three independent measurements pointing at the same conclusion:

1. **Skills suppress reasoning.** On the same Qwen3-4B, mounting a single skill adapter drops the general reasoning probe from 0.533 to 0.400; in chat format it falls to 0.250 — chance level.
2. **Zero knowledge transfer between skills.** In a three-skill cross test, mounting any skill never makes another worse — but it never makes it better either. The apparent cross-skill gain is only a formatting habit ("end with a short answer"; transfer rate 0.39–0.50), not knowledge transfer.
3. **General-capability damage accumulates and is irreversible.** On 0.83B a single skill costs 22–34% of general capability, and the harder you train it in, the more it takes.

A hundred conversations producing a hundred adapters does not accumulate into "understanding me better". It accumulates into a hundred narrow, mutually unrelated mappings while grinding down general capability.

---

### Recommended architecture

```
Conversation happens
   │
   ├─→ Context window (full detail of the current task; discarded at session end)
   │
   └─→ Extraction: an LLM emits structured memory entries
           {"entity": "Project StarTrack", "dimension": "database selection",
            "value": "PostgreSQL 14 primary with streaming replication", "kind": "decision"}
           │
           ├─→ Representation: layer L0 of Qwen3-4B (embedding layer, mean-pooled)
           ├─→ Retrieval: cosine top-k + margin threshold for abstention
           └─→ Storage: entries added/removed freely; weights untouched

Style / working habits (weekly or monthly — not per conversation)
   └─→ LoRA: 4B+ backbone, r=16, all-linear, isolated mounting, rolling full retrain

Backbone upgrade
   └─→ 96 in-distribution anchors → ridge maps h_new → h_old → memory layer moves over
```

**Cost**: extraction ≈ 20 s per conversation (unoptimized 4B), retrieval is one matrix multiply, backbone swap is one calibration.
**No model training is required anywhere in this pipeline.**

---

### The five experiments

| Experiment | Question | Finding |
|---|---|---|
| **P0** Separability | Can paragraph-level natural-language memories be retrieved? | Yes (top-1 1.000), but SNR is only 0.48–0.59, right on the criterion line. Discovered **mid-layer representation collapse** (inter-class angles of just 1–4° in L8–L24); lengthening the text doesn't help; **anchor terms decide everything** |
| **P0-2** Projection | How much does a learned projection help? | SNR pushed to **0.07–0.26**, margin up 2–289×; **collapse is repairable by projection** (L18: acc 0.683→1.000); but projection cannot invent missing information |
| **P0-3** Cross-backbone calibration | Does MSP transfer between backbones work at all? | The old verdict "b/a 0.00–0.04" was **metric distortion** (a meaningless mapping still scored acc 0.983 with margin +0.002). Under margin it is 0.50–0.80 and rises monotonically; **out-of-distribution anchors are worthless** |
| **P0-4** Anchor scaling | How far does it go with enough anchors? | **96 in-distribution anchors → retention 0.96 (L0) / 1.02 (L36)**, past the 0.95 criterion |
| **P1** Extractor | Where does memory come from automatically? | LLM extraction: decision recall **1.000**, anchor completeness **1.000**, end-to-end retrieval **1.000**; rule-based extraction only manages 0.550 / 0.300 / 0.550 |

Full reports live in [`docs/`](docs/):

- [P0 Separability](docs/P0-记忆可区分性实验报告-2026-10-31.md)
- [P0-2 Projection](docs/P0-2-投影实验报告-2026-10-31.md)
- [P0-3 Cross-backbone calibration](docs/P0-3-跨骨干标定实验报告-2026-11-01.md)
- [P0-4 Anchor scaling](docs/P0-4-锚点规模曲线-2026-11-01.md)
- [P1 Extractor](docs/P1-记忆抽取器实验报告-2026-11-01.md)
- [Summary: is dialogue memory feasible](docs/P0系列总结-对话记忆可行性-2026-11-01.md)
- [Design document](DESIGN.md)

*(All reports are written in Chinese.)*

---

### Key numbers

| Quantity | Value | Source |
|---|---|---|
| Retrieval SNR (untrained, same domain) | 0.48–0.59 | P0 |
| Retrieval SNR (after learned projection) | **0.07–0.26** | P0-2 |
| Zero-label working setup | L0 + PCA whitening (k=128) → SNR **0.068** | P0-2 |
| Mid-layer inter-class angle (collapse zone) | 1.2–4.8°, margin ±0.001 | P0 |
| L18 after projection | acc 0.683±0.18 → **1.000±0.00**, margin +0.0001 → **+0.555** | P0-2 |
| Cross-backbone retention (margin, 96 anchors) | **0.96 (L0) / 1.02 (L36)** | P0-4 |
| Out-of-distribution anchors | margin +0.004 ≈ noise floor +0.002 | P0-3 |
| Meaningless mapping: acc vs margin | **0.983 vs +0.002** | P0-3 |
| Recall when anchor terms are missing | **0.283** (worse than the 0.5 chance level) | P0 |
| End-to-end retrieval (dialogue → extraction → lookup) | **top-1 1.000, margin +0.142** | P1 |

---

### Three methodological lessons you can take away

These matter more than the specific numbers — anyone evaluating a memory system or a representation will run into them.

1. **Judge with margin, not accuracy.** On a corpus where every memory carries a distinctive proper noun, even a completely meaningless mapping scores acc 0.983 with a margin of +0.002. Using accuracy as the criterion makes "totally broken" and "works fine" look identical.

2. **Measure inter-class distance to the nearest neighbour, not to class centroids.** Averaging into centroids washes out individual variation and systematically understates inter-class angles, producing a metric that contradicts the accuracy you observe.

3. **Every "learn a mapping" experiment needs a self-transfer sanity check.** Here A→A self-transfer scores 0.74–1.15 (centred on 1.0), which is what proves the pipeline has no bug — only then do the A→B numbers mean anything.

---

### Quick start

```bash
PY="<your python, needs torch + transformers>"

# P0: separability (requires local Qwen3-4B weights)
cd experiments/p0_memory
$PY gen_data.py --out data/p0_corpus.json
$PY run_probe.py --device cuda:0 --out results/probe.json

# P0-2: projection
$PY train_proj.py --out results/proj.json

# P0-3 / P0-4: cross-backbone (requires a second backbone)
$PY encode_bb.py --model <another backbone> --tag bb
$PY cross_backbone.py --out results/cross.json
$PY anchor_scaling.py --layer 0 --out results/anchor_scale.json

# P1: extractor + end-to-end
cd ../p1_extract
$PY gen_dialogues.py --out data/p1_dialogues.json
$PY extract_memory.py --mode llm --out results/extracted_llm.json
$PY eval_e2e.py --extracted results/extracted_llm.json --out results/eval_llm.json
```

All corpora and dialogues are generated from fixed seeds (`20261031` / `20261101`), so results reproduce bit for bit.

**Note**: the repo does not ship the cached representation tensors (`results/reps*/**.npz`, ~360 MB). The first run re-encodes them; Qwen3-4B takes about a minute on a 2080 Ti.

---

### Limitations (important)

1. **Everything is built on templated synthetic corpora**, with a higher density of proper nouns than real dialogue. Accuracy will not saturate in the real world (which is good), but margins will be lower.
2. **Only one backbone pair was tested** (Qwen3-4B → Qwen2.5-3B, same family, adjacent generations). Cross-family and same-generation cross-size were not tested.
3. **The anchor curve shows no saturation point**; 96 is the corpus ceiling, not the optimum.
4. **Each fold has only 6–12 test items**, so margin confidence intervals are wide.
5. **The P1 dialogues have no anaphoric back-references** ("what about that one?") and no multi-speaker turns, and the memory store holds only 20–22 entries. At realistic scale retrieval is an order of magnitude harder.
6. **No conflict resolution**: when the same entity and dimension gets a new value, the system currently stores two contradictory memories.

Items 5 and 6 are the actual remaining distance between "it runs" and "it's usable".

---

### Environment

- GPU: 2× RTX 2080 Ti 22 GB (every experiment fits on a single card)
- Python 3.13 + torch 2.11 + transformers 5.8
- Backbones: Qwen3-4B (primary), Qwen2.5-3B-Instruct (cross-backbone control), Qwen3-4B-Instruct-2507 (extractor)

### License

MIT

---

<a id="chinese"></a>

## 中文

**把对话变成可检索的长期记忆 —— 一份完整的实证报告**

不训练、不微调、不碰权重。用「结构化记忆条目 + embedding 层表示 + 余弦检索」把对话变成模型能查回来的记忆，20 段对话的端到端检索 **top-1 = 1.000**。

换基座时也不需要重训：**96 条同分布锚点**做一次线性标定，记忆层整体带走，性能保持 **96%**。

[↑ Back to English](#english)

---

### 这个项目在回答什么

起点是一个非常自然的想法：

> 能不能把每一次对话都当作一次 LoRA 训练，让模型越用越懂我？

这个仓库用五组实验、约 300 次评估把这个问题拆开、量化、然后给出了一个**和出发点不同**的答案：

**不该把对话训进权重。** 该做的是把对话写成带锚词的结构化条目，存进一个可插拔的记忆层。

而"记忆层"这条路，最初在相关项目里被判过死刑（跨骨干迁移 b/a ≈ 0），本项目发现**那个判决是测量方法错了**——用准确率做判据时它早已饱和，换成 margin 之后，同一个系统在锚点足够时达到 96%。

---

### 核心结论

| 诉求 | 结论 | 关键数字 |
|---|---|---|
| 记住对话里的事实与约定 | ✅ 可行，已端到端跑通 | 抽取召回 1.000，检索 top-1 **1.000** |
| 学会说话风格 / 工作习惯 | ✅ 可行，但基座需 ≥4B | 4B 上通用能力 **0.861×**（净收益）；0.83B 上 **1.338×**（伤 34%） |
| 自动调度（什么该常驻） | 🟡 机制成立，未接真模型 | 玩具加载税 p95 145→30 ms |
| 让它变聪明 | ❌ 不能，且装技能会变笨 | 4B 推理探针 0.533 → 挂技能后 **0.400** |
| **每次对话训一次 LoRA** | ❌ 明确不可行 | 见下 |

#### 为什么"每次对话训一次"不行

三条独立实测，指向同一个结论：

1. **技能压制推理**：同一个 Qwen3-4B，挂上一个技能 adapter 后，通用推理探针从 0.533 掉到 0.400，chat 口径直接掉到随机 0.250。
2. **技能间零知识迁移**：三个技能交叉测试，挂任何技能都不会让别的变差，但也**不会让别的变好**——交叉准确率的提升只是"用短答案收尾"这个格式习惯（迁移率 0.39–0.50），没有知识迁移。
3. **通用伤害累积且不可逆**：0.83B 上单个技能就吃掉 22–34% 通用能力，而且"训得越熟，挤占越狠"。

一百场对话训一百个 adapter，不会累积成"更懂我"，只会累积成一百个互不相干的窄映射，同时把通用能力磨掉。

---

### 推荐架构

```
对话发生
   │
   ├─→ 上下文窗口（当前任务的全部细节，会话结束即弃）
   │
   └─→ 抽取：LLM 产出结构化记忆条目
           {"entity": "星轨", "dimension": "数据库选型",
            "value": "PostgreSQL 14 单主加流复制", "kind": "decision"}
           │
           ├─→ 表示：Qwen3-4B 的 L0（embedding 层 mean-pool）
           ├─→ 检索：余弦 top-k + margin 阈值拒答
           └─→ 入库：可增可删，不碰模型权重

风格 / 工作习惯（周或月级，不是每次）
   └─→ LoRA：4B+ 基座、r=16、全线性层、隔离挂载、滚动全量重训

换基座
   └─→ 96 条同分布锚点 → ridge 学 h_new → h_old → 记忆层整体带走
```

**成本**：抽取约 20 秒/段（4B 未优化），检索一次矩阵乘，换基座一次标定。
**全程不需要训练任何模型。**

---

### 五组实验

| 实验 | 问题 | 结论 |
|---|---|---|
| **P0** 可区分性 | 自然语言段落级记忆能否检索回来？ | 能（top-1 1.000），但 SNR 只有 0.48–0.59，刚好卡在判据线；发现**中层表示塌缩**（L8–L24 类间夹角仅 1–4°）；拉长文本无效；**锚词决定成败** |
| **P0-2** 投影 | 训练个投影能改善多少？ | SNR 压到 **0.07–0.26**，margin 提升 2–289 倍；**塌缩可被投影修复**（L18：acc 0.683→1.000）；但投影补不了信息缺失 |
| **P0-3** 跨骨干标定 | MSP 跨骨干到底行不行？ | 原判"b/a 0.00–0.04"是**指标失真**（无意义映射的 acc 也有 0.983，margin 只有 +0.002）；换 margin 口径是 0.50–0.80 且单调上升；**异分布锚点等于没做** |
| **P0-4** 锚点规模 | 锚点给够能到多少？ | **96 条同分布锚点 → 保持率 0.96（L0）/ 1.02（L36）**，越过 0.95 判据线 |
| **P1** 抽取器 | 记忆怎么自动从对话里来？ | LLM 抽取：决策召回 **1.000**、锚词完整率 **1.000**、端到端检索 **1.000**；规则抽取只有 0.550 / 0.300 / 0.550 |

详细报告见 [`docs/`](docs/)：

- [P0 可区分性](docs/P0-记忆可区分性实验报告-2026-10-31.md)
- [P0-2 投影](docs/P0-2-投影实验报告-2026-10-31.md)
- [P0-3 跨骨干标定](docs/P0-3-跨骨干标定实验报告-2026-11-01.md)
- [P0-4 锚点规模](docs/P0-4-锚点规模曲线-2026-11-01.md)
- [P1 抽取器](docs/P1-记忆抽取器实验报告-2026-11-01.md)
- [总结：对话记忆可行性](docs/P0系列总结-对话记忆可行性-2026-11-01.md)
- [设计文档](DESIGN.md)

---

### 关键数字

| 量 | 值 | 来源 |
|---|---|---|
| 检索 SNR（未训练，同域） | 0.48–0.59 | P0 |
| 检索 SNR（训练投影后） | **0.07–0.26** | P0-2 |
| 零标注可用方案 | L0 + PCA 白化(k=128) → SNR **0.068** | P0-2 |
| 中层类间夹角（塌缩区） | 1.2–4.8°，margin ±0.001 | P0 |
| L18 加投影后 | acc 0.683±0.18 → **1.000±0.00**，margin +0.0001 → **+0.555** | P0-2 |
| 跨骨干保持率（margin，96 锚点） | **0.96（L0）/ 1.02（L36）** | P0-4 |
| 异分布锚点 | margin +0.004 ≈ 噪声底 +0.002 | P0-3 |
| 无意义映射的 acc vs margin | **0.983 vs +0.002** | P0-3 |
| 锚词缺失时召回 | **0.283**（比随机 0.5 还低） | P0 |
| 端到端检索（对话→抽取→查） | **top-1 1.000，margin +0.142** | P1 |

---

### 三条可以直接搬走的方法学教训

这三条比具体数字更有普适价值，任何做记忆系统或表示评估的人都用得上。

1. **判据必须用 margin，不能用准确率。** 在"每条记忆都带独特专有名词"的语料上，连完全无意义的映射都能拿到 acc 0.983，而 margin 只有 +0.002。用 acc 做判据，会把"全废"和"能用"判成同一个数。

2. **检索用的类间距离必须取最近邻，不能用类中心。** 求中心会把个体差异平均掉，系统性低估类间夹角，导致指标与准确率互相矛盾。

3. **任何"学映射"的实验都要加自迁移 sanity check。** 本项目的 A→A 自迁移得到 0.74–1.15（围绕 1.0），这才证明管线没有 bug，A→B 的数字才有意义。

---

### 快速开始

```bash
PY="<你的 python，需 torch + transformers>"

# P0：可区分性（需要 Qwen3-4B 本地权重）
cd experiments/p0_memory
$PY gen_data.py --out data/p0_corpus.json
$PY run_probe.py --device cuda:0 --out results/probe.json

# P0-2：投影
$PY train_proj.py --out results/proj.json

# P0-3/P0-4：跨骨干（需要第二个骨干）
$PY encode_bb.py --model <另一个基座> --tag bb
$PY cross_backbone.py --out results/cross.json
$PY anchor_scaling.py --layer 0 --out results/anchor_scale.json

# P1：抽取器 + 端到端
cd ../p1_extract
$PY gen_dialogues.py --out data/p1_dialogues.json
$PY extract_memory.py --mode llm --out results/extracted_llm.json
$PY eval_e2e.py --extracted results/extracted_llm.json --out results/eval_llm.json
```

语料与对话全部用固定种子生成（`20261031` / `20261101`），结果可逐位复现。

**注意**：仓库不包含缓存的表示张量（`results/reps*/**.npz`，约 360 MB），首次运行会重新编码，Qwen3-4B 在 2080 Ti 上约 1 分钟。

---

### 局限（重要）

1. **全部基于模板合成的语料**，专有名词密度高于真实对话。真实场景 acc 不会饱和（这是好事），但 margin 会更低。
2. **只测了一对骨干**（Qwen3-4B → Qwen2.5-3B，跨代同族）。跨族、同代跨尺寸未测。
3. **锚点曲线未见饱和点**，96 条是语料上限，不是最优值。
4. **每折测试集只有 6–12 item**，margin 置信区间偏宽。
5. **P1 的对话没有指代回跳**（"那它呢？"）和多人插话，记忆库只有 20–22 条。真实规模下检索难度高一个量级。
6. **未做冲突消解**：同一实体同一维度出了新取值时，目前会存成两条互相矛盾的记忆。

第 5、6 条是"能跑"到"能用"之间实际剩下的距离。

---

### 环境

- GPU：2× RTX 2080 Ti 22 GB（单卡即可跑完所有实验）
- Python 3.13 + torch 2.11 + transformers 5.8
- 骨干：Qwen3-4B（主）、Qwen2.5-3B-Instruct（跨骨干对照）、Qwen3-4B-Instruct-2507（抽取器）

### License

MIT

[↑ 回到英文版](#english)
