"""
Configuration for EvolveLab memory system
"""

from typing import Dict, Any
import os
from .memory_types import MemoryType

# Default configuration for different memory providers
STORAGE_BASE_DIR = "../storage"

DEFAULT_CONFIG = {
    "EvolveLab": {
        "default_top_k": 3,
        "active_provider": "agent_kb",  # Default active provider
        "storage_base_dir": STORAGE_BASE_DIR,
    },
    
    "providers": {
        MemoryType.AGENT_KB: {
            "kb_database_path": os.path.join(STORAGE_BASE_DIR, "agent_kb", "agent_kb_database.json"),
            "top_k": 3,
            "search_weights": {'text': 0.5, 'semantic': 0.5},
        },
        
        MemoryType.SKILLWEAVER: {
            "skills_file_path": os.path.join(STORAGE_BASE_DIR, "skillweaver", "skillweaver_generated_skills.py"),
            "skills_dir": os.path.join(STORAGE_BASE_DIR, "skillweaver"),
        },
        
        MemoryType.MOBILEE: {
            "tips_file_path": os.path.join(STORAGE_BASE_DIR, "mobilee", "tips", "tips.json"),
            "shortcuts_file_path": os.path.join(STORAGE_BASE_DIR, "mobilee", "shortcuts", "shortcuts.json"),
        },
        
        MemoryType.EXPEL: {
            "insights_file_path": os.path.join(STORAGE_BASE_DIR, "expel", "insights.json"),
            "success_trajectories_file_path": os.path.join(STORAGE_BASE_DIR, "expel", "success_trajectories.json"),
            "top_k": 3,
            "search_weights": {'text': 0.3, 'semantic': 0.7},
            # embedding model id for sentence-transformers (optional override)
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        },
        MemoryType.LIGHTWEIGHT_MEMORY: {
            "model": None,
            "storage_dir": "../storage/lightweight_memory",
            "max_strategic_memories": 30,
            "max_operational_memories": 30,
            "max_shortterm_items": 15,
            "shortterm_provision_interval": 5,
            "top_k_longterm": 3,
            "enable_longterm_provision": False,
        },
        MemoryType.CEREBRA_FUSION_MEMORY: {
            "model": None,
            "storage_dir": "../storage/cerebra_fusion_memory",
            "db_path": "../storage/cerebra_fusion_memory/cf_database.json",
            "model_cache_dir": "../storage/models",
            "top_k": 3,
            "search_weights": {"text": 0.2, "semantic": 0.8},
            "min_score": 0.22,
            "min_score_in_phase": 0.22,
            "semantic_edge_threshold": 0.75,
            "max_neighbors_expand": 3,
            "enable_graph_expansion": True,
            "enable_tool_memory": True,
            "tools_storage_path": "../storage/cerebra_fusion_memory/tools_storage.py",
            "max_tool_candidates": 3,
            "consolidation_interval": 50,
        },
        MemoryType.DILU: {
            "db_path": "../storage/dilu/dilu_memory.json",      
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_model_cache": "../storage/models"
        },
        MemoryType.GENERATIVE: {
            "db_path": "../storage/generative/generative_memory.json",
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_model_cache": "../storage/models"
        },
        MemoryType.VOYAGER: {
            "db_path": "../storage/voyager/voyager_memory.json",
            "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_model_cache": "../storage/models"
        },
        MemoryType.MEMP: {
            "store_path": os.path.join(STORAGE_BASE_DIR, "memp"),
            "records_file": "procedural_records.json",
        },
        MemoryType.DYNAMIC_CHEATSHEET: {
            "store_path": "../storage/dynamic_cheatsheet",
            "records_file": "dynamic_cheatsheet.json",
            "cheatsheet_file": "global_cheatsheet.txt",
            "top_k": 1,
        },
        MemoryType.AGENT_WORKFLOW_MEMORY: {
            "store_path": "../storage/agent_workflow_memory/workflow_memory.json",
            "index_dir": "../storage/agent_workflow_memory/index",
            "top_k": 1,
            "enable_induction": True,
        },
        MemoryType.EVOLVER: {
            "store_path": "../storage/evolver",
            "records_file": "principle_records.json",
            "search_top_k": 1,
            "max_pos_examples": 1,
            "max_neg_examples": 1,
            "prune_threshold": 0.3,
        },
        MemoryType.COGNI_GRAPH_MEMORY: {
            "database_path": './storage/cogni_graph_memory/graph.json',
            "top_k": 5,
            "buffer_size": 10,
            "model_cache_dir": './storage/models',
            "prune_max_nodes": 1000
        },
        MemoryType.CORTEX_RESONANCE_MEMORY: {
            "planning": 'The strategic approach, decomposition, tool‑use decisions, step‑by‑step reasoning.',
            "experience": 'Key lessons, pitfalls avoided, best practices, tactical insights.',
            "database_path": './storage/cortex_resonance_memory/entries.json',
            "top_k": 5,
            "similarity_threshold": 0.65,
            "merge_threshold": 0.85,
            "model_cache_dir": './storage/models'
        },
        MemoryType.COGNITION_LATTICE: {
            "database_path": './storage/cognition_lattice/memory_database.json',
            "model_cache_dir": './storage/models',
            "top_k": 5,
            "min_utility": 0.1,
            "enable_utility_pruning": True,
            "prune_interval": 50,
            "levels_to_extract": ['level1_specific', 'level2_strategic', 'level3_universal', 'failure_pattern']
        },
        MemoryType.TEMPORAL_EXPERIENCE_GRAPH: {
            "database_path": './storage/temporal_experience_graph/data.json',
            "top_k": 5,
            "similarity_threshold": 0.3,
            "min_utility_threshold": 0.2,
            "search_weights": {'text': 0.4, 'semantic': 0.6},
            "pruning_interval": 50,
            "model_cache_dir": './storage/models'
        },
        MemoryType.ADAPTIVE_ECHO_MEMORY: {
            "raw_db_path": './storage/adaptive_echo_memory/raw_trajectories.json',
            "core_db_path": './storage/adaptive_echo_memory/core_memories.json',
            "top_k": 3,
            "max_core_size": 500,
            "prune_interval": 100,
            "staging_threshold": 10,
            "similarity_threshold": 0.95,
            "merge_threshold": 0.8,
            "phase_weights_begin": {'planning': 0.7, 'experience': 0.3},
            "phase_weights_in": {'planning': 0.3, 'experience': 0.7},
            "model_cache_dir": './storage/models'
        },
        MemoryType.THOUGHT_LOOM_MEMORY: {
            "database_path": './storage/thought_loom_memory/memories.json',
            "graph_path": './storage/thought_loom_memory/graph.json',
            "top_k": 5,
            "similarity_threshold": 0.7,
            "max_memories": 1000,
            "prune_interval": 50,
            "seed_on_cold_start": True
        },
        MemoryType.CONTEXTUAL_RESONANCE_MEMORY: {
            "database_path": './storage/contextual_resonance_memory/data.json',
            "embed_model": 'sentence-transformers/all-MiniLM-L6-v2',
            "top_k": 5,
            "tfidf_weight": 0.3,
            "semantic_weight": 0.7,
            "dedup_threshold": 0.95,
            "enable_phase_aware": True,
            "enable_failure_learning": True,
            "model_cache_dir": './storage/models'
        },
        MemoryType.RESONANT_EXPERIENCE: {
            "Search_Strategy": 'Any patterns in how searches were conducted',
            "Tags": '2-3 keywords',
            "Primary_Failure_Reason": 'What went wrong',
            "Avoidance_Strategy": 'How to avoid this in future',
            "database_path": './storage/resonant_experience/resonant_experiences.json',
            "top_k": 3,
            "partial_success_threshold": 0.4,
            "min_utility_before_prune": 0.2,
            "enable_tool_extraction": True,
            "fallback_guidance_enabled": True,
            "prune_interval": 100,
            "utility_boost_on_success": 0.1,
            "utility_penalty_on_failure": 0.05
        },
        MemoryType.ADAPTIVE_INSIGHT_MEMORY: {
            "database_path": './storage/adaptive_insight_memory/insight_db.json',
            "top_k": 5,
            "in_phase_top_k": 3,
            "score_threshold": 0.25,
            "search_weights": {'text': 0.3, 'semantic': 0.7},
            "field_search_weights": {'query': 1.0, 'planning': 1.2, 'experience': 1.5},
            "model_cache_dir": './storage/models'
        },
        # add new memory type upside this line
}
}


def get_memory_config(provider_type: MemoryType) -> Dict[str, Any]:
    """Get configuration for a specific memory provider"""
    return DEFAULT_CONFIG["providers"].get(provider_type, {}).copy()


def get_evolve_lab_config() -> Dict[str, Any]:
    """Get configuration for the EvolveLab memory system"""
    return DEFAULT_CONFIG["EvolveLab"].copy()