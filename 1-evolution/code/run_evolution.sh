#!/bin/bash
# ============================================================
# MemEvolve 进化 —— 从零生成一个新的 memory system
# ============================================================
# 代码目录：$(pwd)/
# 结果输出：../results/memevolve_work/
# ============================================================

source /opt/miniconda3/etc/profile.d/conda.sh && conda activate memevolve
cd $(dirname "$0")

python evolution/evolve_cli.py \
    --dataset webwalkerqa \
    --seed_provider agent_kb \
    --analysis_model deepseek-v4-flash \
    --generation_model deepseek-v4-flash \
    --num_rounds 3 \
    --tasks_per_round 10 \
    --pareto true

# 注意：如需从 checkpoint 恢复：
# python evolution/evolve_cli.py \
#     --resume_from ../results/memevolve_work/round_02/checkpoint.json
