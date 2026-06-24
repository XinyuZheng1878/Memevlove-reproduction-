# 第二步：xBench 评测 —— 代码分类

## 目录

| 目录 | 分类 | 内容 | 作用 |
|------|------|------|------|
| `agent/` | Agent引擎 | `FlashOAgents/` | 搜索agent的plan-search-crawl循环 |
| `memory/` | 记忆系统 | `EvolveLab/` | 23种memory provider定义+配置（含进化产物） |
| `judge/` | 判分 | `xbench-evals-main/` | Judge判分（`eval_grader.py`） |
| `scripts/` | 入口 | `run_flash_searcher_mm_xbench.py` + 工具函数 | 在xBench上跑agent+memory评测 |
| `data/` | 数据集 | `xbench/DeepSearch.csv` | xBench加密数据集（100题，取前20） |
| `config/` | 配置 | `.env` + `requirements.txt` | API密钥+依赖 |
| `storage/` | 记忆库 | 各memory system的记忆文件 + `models/`(模型缓存) | 运行时自动读写 |

## 运行

```bash
# 方式1：一键脚本
bash run_xbench.sh <provider_name>

# 方式2：手动
python scripts/run_flash_searcher_mm_xbench.py --memory_provider voyager --sample_num 20 ...
```
