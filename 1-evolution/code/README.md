# 第一步：MemEvolve 进化 —— 代码分类

## 目录

| 目录 | 分类 | 内容 | 作用 |
|------|------|------|------|
| `agent/` | Agent引擎 | `FlashOAgents/` | 搜索agent的plan-search-crawl循环 |
| `memory/` | 记忆系统 | `EvolveLab/` | 23种memory provider定义+配置 |
| `evolution/` | 进化引擎 | `MemEvolve/` + `evolve_cli.py` + `mini-swe-agent/` | 分析→生成→创建→验证 进化循环 |
| `judge/` | 判分 | `xbench-evals-main/` | 标准答案比对 |
| `scripts/` | 入口 | `run_flash_searcher_webwalkerqa.py` + 工具函数 | 在WebWalkerQA上跑agent采集轨迹 |
| `data/` | 数据集 | `webwalkerqa/` | WebWalkerQA 170题 |
| `config/` | 配置 | `.env` + `requirements.txt` | API密钥+依赖 |
| `storage/` | 运行时 | `models/`(模型缓存) + `tools/`(工具包装) | 进化过程自动读写 |

## 运行

```bash
# 方式1：一键脚本
bash run_evolution.sh

# 方式2：手动
python evolution/evolve_cli.py --dataset webwalkerqa --seed_provider agent_kb ...
```
