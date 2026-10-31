# -*- coding: utf-8 -*-
"""P1：对话测试集 —— 20 段带噪声的多轮对话 + gold 记忆条目 + 后续追问

为什么需要它
------------
P0 系列所有记忆条目都是手写的。真实系统里，记忆必须**从对话里抽出来**。
抽取环节失败的话，后面再好的检索也白搭。本文件造的就是这个环节的考题。

每条样本包含：
  dialogue   4-8 轮对话，混入与主题无关的闲聊（真实对话的常态）
  gold       应该被抽出来的记忆条目 [{entity, dimension, value, kind}]
  query      若干天后的一条追问，用来测"能不能检索回来"
  answer     该追问的正确答案（= 对应 gold 的 value）

gold 的书写严格遵守 P0 得出的锚词规范：实体 + 维度 + 具体取值。
任何一条 gold 都不允许出现"按之前说的"这类无锚表述——那样的条目本来就检索不回来。

用法：python gen_dialogues.py --out data/p1_dialogues.json
"""
import argparse
import json
import os
import random

SEED = 20261101

# (项目, 类型, 维度, 取值, 备选, 原因, 注意事项, 追问, 答案关键词)
CASES = [
    ("星轨", "内部数据平台", "数据库选型", "PostgreSQL 14 单主加流复制", "MongoDB",
     "事务支持太弱，报表聚合写不动", "明年二季度要上分库分表，得提前留好 shard key",
     "星轨那个数据平台的数据库最后到底用的哪个？", "PostgreSQL"),
    ("潮汐", "客户工单系统", "消息队列", "RocketMQ", "Kafka",
     "需要事务消息，工单状态不能丢", "消费端必须幂等，重复投递要能扛",
     "潮汐的工单系统消息队列选的是哪个？", "RocketMQ"),
    ("长风", "日志分析平台", "存储引擎", "ClickHouse", "Elasticsearch",
     "聚合查询快十倍，磁盘还省一半", "不支持频繁更新，得走批量导入",
     "长风那个日志分析用的什么存储？", "ClickHouse"),
    ("青梧", "内容推荐服务", "缓存层", "Redis Cluster 三主三从", "本地 Caffeine",
     "推荐结果要在多个实例间共享", "热点 key 要做本地二级缓存",
     "青梧的推荐服务缓存是怎么搭的？", "Redis"),
    ("落霞", "电商结算系统", "金额精度", "全链路用 BIGINT 存分", "DECIMAL",
     "Java 侧 BigDecimal 序列化坑太多", "所有除法必须显式指定舍入模式",
     "落霞结算系统的金额是怎么存的？", "BIGINT"),
    ("白鹭", "移动端埋点 SDK", "上报策略", "批量攒够 20 条或 30 秒强制上报", "每条实时上报",
     "实时上报太耗电，用户会卸载", "App 退到后台必须立刻 flush",
     "白鹭埋点 SDK 的上报策略是什么？", "20 条"),
    ("寒山", "合同管理系统", "文件存储", "MinIO 私有化部署", "阿里云 OSS",
     "合同不能出内网，合规要求", "MinIO 要做四副本纠删码",
     "寒山的合同文件存在哪里？", "MinIO"),
    ("南乔", "客服机器人", "模型部署", "本地 vLLM 起 Qwen 系列", "调用公有云 API",
     "客服话术涉及用户隐私，不能出境", "显存不够时要降级到 4bit 量化",
     "南乔的客服机器人模型是怎么部署的？", "vLLM"),
    ("北辰", "设备管理后台", "前端框架", "Vue 3 加 Vite", "React",
     "团队三个人都是 Vue 出身", "老项目 jQuery 部分先做微前端隔离",
     "北辰的设备管理后台前端用什么框架？", "Vue"),
    ("望舒", "定时任务调度", "调度器", "XXL-JOB", "自研 Quartz 集群",
     "现成的有失败重试和告警", "任务必须支持分片广播",
     "望舒的定时任务用的什么调度器？", "XXL-JOB"),
    ("云梦", "在线教育直播", "推流协议", "RTMP 推流加 HTTP-FLV 播放", "WebRTC",
     "WebRTC 要自建 SFU，成本翻倍", "FLV 延迟在三秒左右",
     "云梦直播的推流协议用的什么？", "RTMP"),
    ("子衿", "供应链协同", "接口对接", "EDI 标准报文加 SFTP", "RESTful API",
     "对方是老牌制造企业，只有 EDI 通道", "SFTP 目录要按日期分，文件名带批次号",
     "子衿和供应链那边是怎么对接的？", "EDI"),
    ("清和", "医疗影像归档", "影像格式", "DICOM 原格式加 JPEG2000 压缩", "直接转 JPEG",
     "JPEG 是有损的，不能用于诊断", "归档必须保留原始 DICOM",
     "清和的影像归档用什么格式？", "DICOM"),
    ("松风", "财务报销系统", "审批流引擎", "Flowable", "硬编码状态机",
     "报销规则每个季度都变", "流程定义要能热部署",
     "松风的报销审批流用的哪个引擎？", "Flowable"),
    ("既明", "IoT 设备网关", "通信协议", "MQTT 3.1.1", "HTTP 轮询",
     "设备三万台，HTTP 轮询把服务打挂过", "QoS 要用 1，QoS 2 会阻塞队列",
     "既明的设备网关用什么协议？", "MQTT"),
    ("疏影", "会员积分系统", "一致性方案", "本地消息表加定时补偿", "分布式事务框架",
     "积分允许最终一致，上 Seata 太重", "补偿任务要能人工干预",
     "疏影的积分系统用什么保证一致性？", "本地消息表"),
    ("沧溟", "海运订舱平台", "汇率处理", "每小时拉取一次中间价并落库", "实时查询第三方接口",
     "实时接口有 QPS 限制", "汇率变动超过千分之五要发告警",
     "沧溟的汇率数据是怎么更新的？", "每小时"),
    ("怀瑾", "人事档案系统", "数据脱敏", "身份证和手机号在入库前加密", "查询时脱敏",
     "入库前加密可以避免 DBA 看到明文", "加密后不能做模糊查询",
     "怀瑾的人事档案敏感字段怎么处理？", "入库前加密"),
    ("映月", "视频会议录制", "转码方案", "GPU 硬编 H.264", "CPU 软编 x264",
     "一路 1080p 软编要吃掉八个核", "硬编在同码率下画质略差",
     "映月的会议录制用什么转码？", "GPU 硬编"),
    ("振鹭", "仓储拣货系统", "路径规划", "S 形蛇形拣货", "遗传算法优化",
     "货位是固定分区，蛇形已经接近最优", "临时插单要能立刻重排路径",
     "振鹭的拣货路径是怎么规划的？", "蛇形"),
]

NOISE = [
    "顺便问一下，今天下午那个会改到几点了？",
    "对了，你昨天说的那个文档我看了，还行。",
    "等一下我先接个电话。",
    "嗯，还有别的事吗？",
    "先这样吧，我这边还有个会。",
    "刚才说到哪了，哦对。",
    "这个不着急，下周再说也行。",
    "午饭吃了没？",
]

OPENERS = [
    "那个{n}的项目，{d}这块最后是怎么定的来着？",
    "{n}那边的{d}，我们当时是选了哪个？",
    "我想确认一下，{n}的{d}定下来了吗？",
    "帮我回忆下，{n}这个项目{d}用的什么方案？",
]

REPLIES = [
    "最终定的是{v}。当时考虑过{a}，但{r}，所以没走那条。另外{n2}。",
    "是{v}。一开始的备选是{a}，评估下来{r}，就换成{v}了。要留意的点是{n2}。",
    "{v}。没选{a}的原因是{r}。还有一条：{n2}。",
]

CONFIRMS = [
    "哦对，想起来了。",
    "行，那就这样。",
    "嗯，跟我记的一样。",
    "好，我知道了。",
    "对，就是这个。",
]

NOISE_REPLIES = [
    "我看一下日程，应该是改到四点了。",
    "那个文档我这边没收到，回头再发你一次。",
    "不着急，你先忙。",
    "是的，就这些。",
    "好，那你先开会，有事随时说。",
    "刚说到方案那块，你继续。",
    "行，下周再对一次。",
    "吃了，你呢。",
]


def build_one(idx, case, rng):
    name, kind, dim, val, alt, reason, note, query, answer = case
    opener = rng.choice(OPENERS).format(n=name, d=dim)
    reply = rng.choice(REPLIES).format(v=val, a=alt, r=reason, n2=note)
    confirm = rng.choice(CONFIRMS)

    turns = [
        {"role": "user", "content": opener},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": confirm},
    ]
    # 插 2-3 句噪声，位置随机（噪声也要有实质回复，且不重复，否则对话一眼假）
    noises = rng.sample(NOISE, rng.randint(2, 3))
    replies = rng.sample(NOISE_REPLIES, len(noises))
    for noise, nrep in zip(noises, replies):
        pos = rng.choice([2, 4])
        turns.insert(min(pos, len(turns)), {"role": "user", "content": noise})
        turns.insert(min(pos + 1, len(turns)), {"role": "assistant", "content": nrep})

    gold = [
        {"entity": name, "dimension": dim, "value": val, "kind": "decision"},
        {"entity": name, "dimension": "技术选型排除项", "value": f"未采用 {alt}，因为{reason}", "kind": "fact"},
    ]
    return {
        "id": f"D{idx:02d}",
        "turns": turns,
        "gold": gold,
        "query": query,
        "answer": answer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/p1_dialogues.json")
    args = ap.parse_args()
    rng = random.Random(SEED)

    samples = [build_one(i, c, rng) for i, c in enumerate(CASES, 1)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"seed": SEED, "samples": samples},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    ln = [sum(len(t["content"]) for t in s["turns"]) for s in samples]
    print(f"written {args.out}: {len(samples)} 段对话, 字数 min/mean/max = "
          f"{min(ln)}/{sum(ln)//len(ln)}/{max(ln)}, gold 条目 {sum(len(s['gold']) for s in samples)} 条")


if __name__ == "__main__":
    main()
