#!/bin/bash
# ============================================================
# xBench 评测 —— 在 20 条 DeepSearch 题上测评 memory system
# ============================================================
# 用法: bash run_xbench.sh <provider_name> [task_indices]
# 例:   bash run_xbench.sh voyager
#       bash run_xbench.sh cerebra_fusion_memory
#       bash run_xbench.sh temporal_experience_graph
#       bash run_xbench.sh lightweight_memory 12-20
# ============================================================

PROVIDER=${1:-voyager}
INDICES=${2:-}

source /opt/miniconda3/etc/profile.d/conda.sh && conda activate memevolve
cd $(dirname "$0")

# 清空对应 storage（保证从头积累记忆）
case $PROVIDER in
    lightweight_memory)
        rm -rf ../storage/lightweight_memory/*
        ;;
    agent_kb)
        rm -rf ../storage/agent_kb/*
        ;;
    cerebra_fusion_memory)
        rm -rf ../storage/cerebra_fusion_memory/*
        ;;
    voyager)
        rm -f ../storage/voyager/voyager_memory.json
        ;;
    temporal_experience_graph)
        rm -f ../storage/temporal_experience_graph/data.json
        ;;
esac

# 构造命令
CMD="python scripts/run_flash_searcher_mm_xbench.py \
    --infile ../data/xbench/DeepSearch.csv \
    --outfile ../results/xbench_output/${PROVIDER}_results.jsonl \
    --memory_provider ${PROVIDER} \
    --judge_model deepseek-v4-flash \
    --max_steps 40 \
    --concurrency 1"

if [ -n "$INDICES" ]; then
    CMD="$CMD --task_indices $INDICES"
else
    CMD="$CMD --sample_num 20"
fi

echo "Running: $CMD"
eval $CMD
