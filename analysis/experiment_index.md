# MemEvolve 实验索引 —— 文件位置、命令、结果

> 更新时间: 2026-06-20
> 环境: conda env `memevolve`, Python 3.10, macOS, DeepSeek-V4-Flash
> 项目根目录: `/Users/zhengxinyu/projects/MemEvolve/`

---

## 一、项目结构

```
MemEvolve/
├── 1-evolution/          ← 第一步：MemEvolve 进化
│   ├── code/
│   │   ├── agent/        FlashOAgents（agent引擎）
│   │   ├── memory/       EvolveLab（25个memory system仓库）
│   │   ├── evolution/    MemEvolve（进化引擎）
│   │   ├── judge/        xbench-evals-main（评测框架）
│   │   ├── scripts/      入口脚本+工具函数
│   │   └── config/       .env + requirements
│   ├── data/             WebWalkerQA数据集
│   ├── storage/          运行时存储（模板空目录）
│   └── results/
│       └── memevolve_work/  3轮进化完整产物
│
├── 2-benchmark/          ← 第二步：xBench 评测
│   ├── code/
│   │   ├── agent/        FlashOAgents（agent引擎）
│   │   ├── memory/       EvolveLab（含进化产物）
│   │   ├── judge/        xbench-evals-main（评测框架）
│   │   ├── scripts/      入口脚本+工具函数
│   │   └── config/       .env + requirements
│   ├── data/             xBench DeepSearch.csv
│   ├── storage/          运行时记忆存储
│   └── results/
│       └── xbench_output/  6系统×20题完整结果
│
└── analysis/             ← 分析文档
    ├── experiment_index.md    本文档
    └── per_task_analysis.md   逐题case分析
```

---

## 二、代码修改

### 修改 1：cerebra 编码器 fix

**文件**: `1-evolution/code/memory/EvolveLab/providers/cerebra_fusion_memory_provider.py` 第 57-76 行
（`2-benchmark/code/memory/` 下有相同副本）

**原问题**：`_safe_get_model_response()` 调用顺序错误——先尝试 `smolagents.MessageRole`（未安装→异常），再 `model(prompt)` 纯字符串（OpenAIServerModel 不支持→异常），最终 return None → 触发 fallback 空节点。

**修复**：重新排序，先尝试标准 OpenAI 格式：
```python
# 第一尝试：标准 OpenAI message 格式（兼容 OpenAIServerModel）
messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
resp = model(messages)
# fallback: smolagents.MessageRole → model(prompt) 纯字符串
```

**效果**: 修复前 cerebra 产出 22 个空壳 success 节点 0 条边；修复后单任务即产出 10 个有效节点 + 9 条语义边。准确率从 75% → 90%。

### 修改 2：voyager 存储 bug fix

**文件**: `1-evolution/code/memory/EvolveLab/providers/voyager_memory_provider.py` 第 227 行
（`2-benchmark/code/memory/` 下有相同副本）

**原问题**：`take_in_memory()` 中创建了 `memory_doc` 并生成了 embedding，但缺少 `self.memories.append(memory_doc)`，导致记忆从未被加入列表。log 全程 `Loaded 0 memories`。

**修复**：在第 227 行（`new_embedding = self.embedding_model.encode(...)` 之前）添加：
```python
self.memories.append(memory_doc)
```

### 修改 3：data/ 和 storage/ 提至 code/ 同级

将 `data/` 和 `storage/` 从 `code/` 内部移至与 `code/` 平行。涉及改动：
- `config.py`：`STORAGE_BASE_DIR` 从 `"./storage"` → `"../storage"`
- 所有 provider 默认路径：`./storage/` → `../storage/`
- `run_xbench.sh`：`storage/` → `../storage/`，`./data/` → `../data/`
- `run_flash_searcher_webwalkerqa.py`：`./data/` → `../data/`

---

## 三、配置文件

### `.env`（`2-benchmark/code/config/.env` 和 `1-evolution/code/config/.env`）
```
DEFAULT_MODEL="deepseek-v4-flash"
OPENAI_API_KEY="sk-xxx"
OPENAI_API_BASE="https://api.deepseek.com"       # 注意：代码读的是 OPENAI_API_BASE
SERPER_API_KEY="xxx"
JINA_API_KEY="xxx"
WEB_ACCESS_PROVIDER="jina"
```

**注意事项**：
- 变量名是 `OPENAI_API_BASE`，不是 `OPENAI_BASE_URL`
- Serper 容易超额，充值地址: https://serper.dev
- Jina 也有额度上限，可用 `crawl4ai` 替代

---

## 四、可复现命令

### 第一步：进化（1-evolution）

```bash
cd /Users/zhengxinyu/projects/MemEvolve/1-evolution/code
bash run_evolution.sh

# 或手动执行：
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate memevolve
python -m MemEvolve.evolver \
    --dataset webwalkerqa \
    --seed_provider agent_kb \
    --analysis_model deepseek-v4-flash \
    --generation_model deepseek-v4-flash \
    --num_rounds 3 \
    --tasks_per_round 10 \
    --pareto true
```

### 第二步：xBench 评测（2-benchmark）

```bash
cd /Users/zhengxinyu/projects/MemEvolve/2-benchmark/code

# 依次运行 6 个 memory system（每个 20 题）：
bash run_xbench.sh voyager
bash run_xbench.sh cerebra_fusion_memory
bash run_xbench.sh temporal_experience_graph
bash run_xbench.sh lightweight_memory
bash run_xbench.sh agent_kb
bash run_xbench.sh dilu_memory

# 或手动执行（以 voyager 为例）：
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate memevolve
python scripts/run_flash_searcher_mm_xbench.py \
    --infile ../data/xbench/DeepSearch.csv \
    --outfile ../results/xbench_output/voyager_results.jsonl \
    --memory_provider voyager \
    --judge_model deepseek-v4-flash \
    --sample_num 20 --max_steps 40 --concurrency 1
```

---

## 五、最终结果

### 6 系统 × 20 题 xBench 对比

| 系统 | 准确率 | Memory类型 | 结果目录 |
|------|--------|-----------|---------|
| **voyager** | **18/20 (90%)** | episodic | `2-benchmark/results/xbench_output/voyager_results_runs/voyager_20260620_131425/` |
| **cerebra修复** | **18/20 (90%)** | graph | `2-benchmark/results/xbench_output/cerebra_fixed_rerun_runs/cerebra_fusion_memory_20260620_162759/` |
| lightweight | 17/20 (85%) | procedural | `2-benchmark/results/xbench_output/lightweight_results_runs/lightweight_memory_20260613_172237/` |
| agent_kb | 17/20 (85%) | semantic | `2-benchmark/results/xbench_output/agent_kb_results_runs/agent_kb_20260613_212231/` |
| evolved | 17/20 (85%) | graph+temporal | `2-benchmark/results/xbench_output/evolved_results_runs/temporal_experience_graph_20260620_144017/` |
| cerebra原版 | 15/20 (75%) | graph(损坏) | `2-benchmark/results/xbench_output/cerebra_results_runs/cerebra_fusion_memory_20260613_195425/` |

### 逐题错题分布

| Task | 题目 | voyager | cerebra修复 | lightweight | agent_kb | evolved | cerebra原版 |
|------|------|---------|------------|-------------|----------|---------|------------|
| 5 | 肖申克→羽毛→动物数 | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| 8 | 伦敦奥运奖牌 | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| 9 | 地铁站距第二近 | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| 10 | 三点等距几何 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 12 | 人口年均增长% | ❌ | ✅ | ✅ | ✅ | ❌ | ✅ |
| 16 | 喜人奇妙夜演员 | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| 17 | 中财大校长姓氏 | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| 18 | 蔚来轿车数 | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |

> Task 10 三点等距几何：5/6 系统全错，仅 voyager 巧合答对（搜到了别人算好的结果）。这是 Flash-Searcher 工具能力天花板（缺少坐标计算工具），非 memory 问题。

### Serper 无效运行（已排除）

| 运行 | 准确率 | Serper 错误 | 有效？ |
|------|--------|------------|--------|
| cerebra修复 #1 | 10/20 (50%) | 20/20 | ❌ 废弃 |
| voyager #1 | 6/11 (55%) | 11/11 | ❌ 废弃 |

---

## 六、MemEvolve 进化产物

### 进化工作目录: `1-evolution/results/memevolve_work/`

```
memevolve_work/
├── evolve_state.json              # 进化状态：3轮，best=temporal_experience_graph
├── round_00/                      # Round 0: seed=agent_kb
│   ├── analysis_report.json       # LLM 诊断报告
│   ├── generated_system_{1,2,3}.json  # 3个候选设计
│   ├── created_system.json        # winner: cogni_graph_memory
│   ├── validated_systems.json     # 验证结果
│   ├── round_summary.json
│   └── base_logs/                 # agent_kb 的 10 条 task 轨迹 (1.json~10.json)
├── round_01/
│   ├── analysis_report.json
│   ├── generated_system_{1,2,3}.json
│   ├── created_system.json        # winner: temporal_experience_graph
│   ├── validated_systems.json
│   ├── round_summary.json
│   └── base_logs/                 # task 21-30 的轨迹
└── round_02/                      # Round 2: 失败
    ├── checkpoint.json            # 中断检查点，可用 --resume_from 恢复
    ├── analysis_report.json
    ├── validated_systems.json     # 验证失败（语法错误）
    └── base_logs/                 # task 41-50 的轨迹
```

### 进化产物
- **Round 0 winner**: `cogni_graph_memory` → `EvolveLab/providers/cogni_graph_memory_provider.py`
- **Round 1 winner**: `temporal_experience_graph` → `EvolveLab/providers/temporal_experience_graph_provider.py`（458 行）
- **Round 2**: 失败，LLM 生成的代码有语法错误，未修复

---

## 七、数据集

| 数据集 | 路径 | 格式 | 用途 |
|--------|------|------|------|
| xBench DeepSearch | `2-benchmark/code/data/xbench/DeepSearch.csv` | 加密(base64+XOR) | 20题测评 |
| WebWalkerQA | `1-evolution/code/data/webwalkerqa/webwalkerqa_subset_170.jsonl` | JSONL | 进化阶段评估 |

---

## 八、所有 provider 名称

| 命令行参数 | 对应文件 |
|-----------|---------|
| `voyager` | `memory/EvolveLab/providers/voyager_memory_provider.py` |
| `cerebra_fusion_memory` | `memory/EvolveLab/providers/cerebra_fusion_memory_provider.py` |
| `lightweight_memory` | `memory/EvolveLab/providers/lightweight_memory_provider.py` |
| `agent_kb` | `memory/EvolveLab/providers/agent_kb_provider.py` |
| `temporal_experience_graph` | `memory/EvolveLab/providers/temporal_experience_graph_provider.py` |
| `dilu_memory` | `memory/EvolveLab/providers/dilu_memory_provider.py` |

---

## 九、已知问题

1. **Serper 额度极易耗尽**：每次 20 题约消耗 200-400 次搜索。充值: https://serper.dev
2. **cerebra 原版编码器 bug**：`_safe_get_model_response` 调用顺序错误，已修复
3. **进化产物未超越 baseline**：temporal_experience_graph(85%) = lightweight/agent_kb(85%)
4. **Task 10 天花板**：三点等距几何需要坐标计算工具，5/6 系统全错
5. **concurrency 必须为 1**：多线程会导致记忆系统序列积累逻辑混乱
6. **report.txt 有 bug**：总显示 Correct:0，准确率需从各 JSON `score` 字段手动统计
