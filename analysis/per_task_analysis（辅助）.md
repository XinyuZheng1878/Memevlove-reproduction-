# xBench DeepSearch 20题逐题分析 —— 6个Memory System对比

> 更新时间: 2026-06-21
> 模型: DeepSeek-V4-Flash | Agent: Flash-Searcher | 数据集: xBench DeepSearch (前20题)

---

## 一、各系统实际运行状态

分析之前，首先通过 `.log` 文件确认每个 memory system 是否真的在存储和提供记忆：

| 系统 | 实际状态 | log 证据 |
|------|---------|---------|
| cerebra修复 | ✅ 正常工作 | 节点 7→184，边 0→177，全程提供 guidance |
| cerebra原版 | ❌ 空壳噪声 | 22 个空壳节点 0 条边，TF-IDF 随机匹配注入噪声 |
| voyager | ❌ 零记忆 | 全部 task `Loaded 0 memories`（`append` 缺失） |
| lightweight | ❌ 仅短时 | 20/20 task `Long-term memory provision disabled` |
| agent_kb | ⚠️ 部分 | 14/20 成功存储（LLM JSON 截断 3 次 + 任务错误跳过 3 次） |
| evolved | ⚠️ 部分 | 13/20 成功存储（LLM 编码失败 4 次 + 任务错误跳过 3 次） |

**关键事实**：6 个系统中，只有 cerebra修复真正完整地工作了。voyager 虽然名字叫 episodic memory，实际是零记忆系统——其 90% 的准确率反映的是 agent 自身能力，而非 episodic memory 的效果。cerebra原版的 22 个空壳节点不是"没提供记忆"，而是持续通过 TF-IDF 注入不相关内容，属于主动干扰。

---

## 二、总体结果

| 系统 | 准确率 | 实际 memory 状态 |
|------|--------|:---------------:|
| voyager | 18/20 (90%) | 零记忆 |
| cerebra 修复 | 18/20 (90%) | ✅ 正常 |
| lightweight | 17/20 (85%) | 仅短时 |
| agent_kb | 17/20 (85%) | ⚠️ 部分 |
| evolved | 17/20 (85%) | ⚠️ 部分 |
| cerebra 原版 | 15/20 (75%) | ❌ 噪声 |

---

## 三、逐题对照表

| # | 题目 | 正确答案 | voyager | CB原版 | light | agKB | evol | CB修复 |
|---|------|---------|:------:|:------:|:-----:|:----:|:----:|:-----:|
| 1 | 上海金交所Au(T+D)差价 | 161.27元 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 2 | Intel Gen9 GPU算力 | 384GFLOPs | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 3 | 聊斋·绿衣女，于璟讲话句数 | 4句 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 4 | B站HOPICO对方大同专访期数 | 292期 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 5 | 肖申克台词→GB/T→动物数 | 12只 | ✅ | ❌8 | ❌8只 | ✅ | ❌8 | ✅ |
| 6 | 两面包夹芝士干员职业 | 重装 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 7 | 围棋战鹰2024战绩 | 0胜2负 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 8 | 2012伦敦奥运中国奖牌数 | 金39银31**铜22** | ✅ | ✅ | ✅ | ✅ | ✅ | ❌铜23 |
| 9 | 北京地铁站距第二近 | 16号线玉渊潭东门-木樨地 | ✅ | ❌11号 | ✅ | ❌14号 | ✅ | ✅ |
| 10 | 三点等距几何（文天祥/于谦/袁崇焕祠） | **6～7km** | ✅ | ❌4km | ❌64.6 | ❌2.6 | ❌4.81 | ❌2.8 |
| 11 | 漠河最高温×10→最近熔点金属 | 锌 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 12 | 1949-2009城镇人口年均增长% | **4.04%** | ❌4.11 | ✅ | ✅ | ✅ | ❌4.11 | ✅ |
| 13 | 黑吉辽与外国接壤地级行政区数 | 12个 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 14 | B站道可道Science视频人物+诺奖数 | 2人 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 15 | 阿里18创始人马/蔡/张姓平均年龄 | 34.3岁 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 16 | 喜人奇妙夜S1最佳小队最后出场演员 | 王建华 | ✅ | ❌刘旸 | ❌刘旸 | ✅ | ✅ | ✅ |
| 17 | 中财大历任校长最多姓氏 | **王** | ❌并列 | ✅ | ✅ | ❌并列 | ✅ | ✅ |
| 18 | 蔚来轿车款数（时间窗口内） | 2款 | ✅ | ❌3 | ✅ | ✅ | ✅ | ✅ |
| 19 | 港中深韩晓光CVPR2025论文数 | 5篇 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 20 | 港中文2025 QS排名比2020高多少 | 10名 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | **正确数** | | **18** | **15** | **17** | **17** | **17** | **18** |

---

## 四、重点 case 逐题分析

以下分析全部基于各系统运行时的 `.log` 文件和 trajectory JSON。

### 4.1 Task 5 — 肖申克→羽毛→动物数（三系统对比）

正确答案 12 只。cerebra修复 ✅，lightweight ❌（8 只），cerebra原版 ❌（8 只）。

**两个失败系统的共同错误**：lightweight 和 cerebra原版都搜到了人民网的"平均制作一个羽毛球耗鹅 0.8 只"数据，直接用 10 × 0.8 = 8 只鹅。差别在于 lightweight 无任何 guidance，cerebra原版在 6 次空壳节点的 `[Past Experience]` 噪声中执行。

**cerebra修复为什么对了**：memory guidance 给出了策略提示——"交叉验证多个来源的羽毛产量数据有助于提高准确性"。这条 guidance 没有直接提供答案数字，但引导 agent 不满足于第一个搜到的"0.8 只/球"数据，而是继续搜索到了更准确的"6-7 根/侧"羽毛数据，最终得出 12 只。

**三种 memory 状态的对照**：

| 系统 | memory 状态 | 搜索行为 | 结果 |
|------|-----------|---------|:--:|
| lightweight | 无累积记忆 | 搜到"0.8只/球"即停止 | ❌ |
| cerebra原版 | 空壳噪声 | 同 lightweight | ❌ |
| cerebra修复 | 正常 guidance | 交叉验证，不满足于单一数据源 | ✅ |

memory 在这里的作用是策略层面的引导，而非知识层面的直接提供。

---

### 4.2 Task 8 — 伦敦奥运奖牌（正常 memory 的意外代价）

正确答案铜牌 22。cerebra修复答了铜牌 23——其余 5 个系统全对。

**log 对比**：

| 系统 | HTTP 请求数 | 结果 |
|------|:--------:|:--:|
| voyager（零记忆） | 11 | ✅ |
| cerebra修复 | **59** | ❌ |

cerebra修复的过程：
1. 第一步搜到正确数字 `China with 39 gold, 31 silver, 22 bronze`
2. memory guidance 提示交叉验证和查 doping 调整 → agent 搜索 `"IOC medal reallocation London 2012 China doping"`
3. 爬到 IOC 关于 1500 米项目 disqualification 的通告
4. agent 错误地将其他项目的奖牌调整关联到中国 → 输出"铜牌 23"

**要点**：memory 给出的 guidance（"交叉验证、注意奖牌重新分配"）本身是合理的搜索策略——奥运奖牌数确实可能因后续 doping 处罚而调整。问题不在于 memory 给了错误信息，而在于 agent 处理不了 guidance 带来的额外搜索路径，最终用错误推断覆盖了正确数字。voyager 无 memory，走最短路径（搜 IOC → 爬 Wikipedia），11 次请求直接答对。

---

### 4.3 Task 10 — 三点等距几何（工具能力天花板）

5/6 系统全错，仅 voyager 答对。需要计算三个祠堂的外接圆圆心和半径，但 Flash-Searcher 没有坐标计算工具。

voyager 的 55 次 HTTP 请求中很可能直接搜到了别人已算好的答案，属于搜索运气。cerebra修复在 memory guidance 引导下偏"直接搜答案"路线（"北京 三祠 文天祥 于谦 袁崇焕 等距 中心点"），未找到可靠来源。这题的瓶颈是工具缺失，memory 无法弥补。

---

### 4.4 Task 17 — 中财大校长姓氏（memory 无用）

正确答案"王"。voyager（零记忆）和 agent_kb（详细 guidance）都答了"王和陈并列"。

agent_kb 的 trajectory 中 guidance 明确包含"注意区分正式校长与临时代理"——但 agent 仍然多统计了陈姓（可能是党委书记或代理校长）。voyager 零 memory 也一样多算。有/无 memory 结果相同，说明这题的瓶颈是 agent 自身的精确实体消歧能力，memory 的 guidance 文本传递方式不足以改变 agent 的执行精度。

---

### 4.5 cerebra原版的独特错误模式：空壳节点 = 噪声源

cerebra原版错 5 题，是错误最多的系统。其 22 个节点全部是编码 bug 产生的空壳——只有 task 名，无实质内容。但 TF-IDF 检索仍然工作，依据节点名中的词频做匹配，而中文 task 名之间共享大量常用词，导致匹配近乎随机。以 Task 10 为例，trajectory 中 `[Past Experience] similar tasks have execution for: 围棋棋手战鹰...` 反复出现——一个围棋选手战绩查询被 TF-IDF 匹配到了几何题中。

cerebra原版 vs voyager 的对比最清晰地说明了 memory 的风险：voyager 的 90% 是没有 memory 干扰的真实水平，cerebra原版的 75% 是被 22 个噪声源持续误导后的退化水平。没有 memory 比有损坏的 memory 好。

---

## 五、错题分布模式

```
voyager     90%   错: Task 12(数值精度), Task 17(实体消歧)
cerebra修复  90%   错: Task 8(memory认知过载), Task 10(工具天花板)
lightweight  85%   错: Task 5(数据源陷阱), Task 10, Task 16(娱乐精确检索)
agent_kb     85%   错: Task 10, Task 17(实体消歧), Task 9(全局比较)
evolved      85%   错: Task 5(数据源陷阱), Task 10, Task 12(数值精度)
cerebra原版   75%   错: Task 5,9,10,16,18(最多，空壳节点噪声)
```

**voyager（零记忆）和 cerebra修复（正常 memory）的错误类型不同**：voyager 在需要精确数值精度的题上犯错（Task 12），cerebra修复在 memory 引入额外信息后 agent 处理不过来的题上犯错（Task 8）。这不是 episodic vs graph 的对比（voyager 的 memory 根本没工作），而是"无 memory"和"有 working memory"的对比。

---

## 六、从日志得出的核心结论

1. **Memory 有害的两种情况**：(a) 损坏的 memory 持续注入噪声——cerebra原版 22 个空壳节点，比无 memory 低了 15%；(b) 正常 memory 给出合理策略但 agent 处理不了额外信息——cerebra修复 Task 8，59 次请求反而答错。

2. **Memory 的正面效果微弱且仅在策略层面**：Task 5 cerebra修复的"交叉验证"guidance 帮助避开了单一数据源陷阱，但作用的不是提供知识而是引导搜索行为。

3. **编码可靠性是所有 memory system 的共同瓶颈**：cerebra原版 22/22 次编码失败（100%），evolved 7/20（35%），agent_kb 6/20（30%）。LLM 将长轨迹压缩为结构化知识这件事在当前模型下不可靠。

4. **简单任务上 memory 不产生区分度**：12/20 题所有系统全对，这些题 agent 自身搜索能力足够覆盖。

5. **正确的对比基准**：本次实验真正有效的对比是"无 memory（voyager）vs 噪声 memory（cerebra原版）vs 正常 memory（cerebra修复）"，而非"episodic vs semantic vs graph"。因为只有 cerebra修复真正工作了。
