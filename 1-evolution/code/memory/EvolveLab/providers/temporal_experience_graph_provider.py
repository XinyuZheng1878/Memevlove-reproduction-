import json
import os
import uuid
import sys
import re
import time
from typing import List, Optional, Dict, Any
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryRequest,
    MemoryResponse,
    TrajectoryData,
    MemoryType,
    MemoryItem,
    MemoryItemType,
    MemoryStatus
)


@dataclass
class MemoryEntry:
    id: str
    query: str
    strategy: str          # high-level essence (1 sentence)
    methodology: str       # mid-level workflow (2-3 sentences)
    detailed_steps: str    # numbered steps
    experience: str        # key lessons learned
    support_count: int = 1
    retrieval_count: int = 0
    success_count: int = 0
    last_retrieved: float = 0.0
    created: float = 0.0
    # cached embeddings for each field
    query_embedding: Optional[np.ndarray] = None
    strategy_embedding: Optional[np.ndarray] = None
    methodology_embedding: Optional[np.ndarray] = None
    detailed_embedding: Optional[np.ndarray] = None
    experience_embedding: Optional[np.ndarray] = None


class TemporalExperienceGraph:
    """
    Core memory storage with multi-level abstraction, hybrid retrieval,
    redundancy integration, and utility tracking.
    """
    def __init__(self, embedding_model: SentenceTransformer,
                 tfidf_vectorizer: TfidfVectorizer = None):
        self.entries: Dict[str, MemoryEntry] = {}
        self.embedding_model = embedding_model
        self.tfidf_vectorizer = tfidf_vectorizer or TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = None
        self.entry_ids_tfidf: List[str] = []

    def add_entry(self, entry: MemoryEntry):
        self.entries[entry.id] = entry

    def remove_entry(self, entry_id: str):
        if entry_id in self.entries:
            del self.entries[entry_id]

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        return self.entries.get(entry_id)

    def rebuild_indices(self):
        """Rebuild TF-IDF matrix and compute all missing embeddings."""
        if not self.entries:
            self.tfidf_matrix = None
            self.entry_ids_tfidf = []
            return

        # Rebuild TF-IDF on queries
        ids = []
        texts = []
        for eid, entry in self.entries.items():
            ids.append(eid)
            texts.append(entry.query)
        self.entry_ids_tfidf = ids
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        # Compute embeddings for any entries missing them
        entries_to_encode = [e for e in self.entries.values()
                             if e.query_embedding is None]
        if entries_to_encode:
            queries = [e.query for e in entries_to_encode]
            embs = self.embedding_model.encode(queries, convert_to_numpy=True,
                                               show_progress_bar=False)
            for i, e in enumerate(entries_to_encode):
                e.query_embedding = embs[i]

    def hybrid_search(self, query_embedding: np.ndarray,
                      query_text: str,
                      top_k: int = 5,
                      weights: Optional[Dict[str, float]] = None) -> List[MemoryEntry]:
        weights = weights or {'text': 0.4, 'semantic': 0.6}
        score_board = defaultdict(float)

        # Semantic similarity on query embeddings
        if self.entries:
            all_ids = list(self.entries.keys())
            all_embs = np.array([self.entries[eid].query_embedding
                                 for eid in all_ids
                                 if self.entries[eid].query_embedding is not None],
                                dtype=np.float32)
            if len(all_embs) > 0:
                sem_scores = cosine_similarity([query_embedding], all_embs)[0]
                for idx, eid in enumerate(all_ids):
                    score_board[eid] += weights['semantic'] * sem_scores[idx]

        # Text similarity via TF-IDF
        if self.tfidf_matrix is not None and self.entry_ids_tfidf:
            query_vec = self.tfidf_vectorizer.transform([query_text])
            text_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            for idx, eid in enumerate(self.entry_ids_tfidf):
                score_board[eid] += weights['text'] * text_scores[idx]

        # Sort by score
        sorted_eids = sorted(score_board.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for eid, score in sorted_eids:
            entry = self.entries[eid]
            # apply adaptive threshold based on utility
            if entry.retrieval_count > 0:
                success_rate = entry.success_count / entry.retrieval_count
                # boost threshold if low success, lower if high success
                adjusted_threshold = 0.3 - (success_rate - 0.5) * 0.2
                adjusted_threshold = max(0.1, min(0.7, adjusted_threshold))
            else:
                adjusted_threshold = 0.3
            if score >= adjusted_threshold:
                results.append(entry)
        return results


class TemporalExperienceGraphProvider(BaseMemoryProvider):
    """
    Provides phase-aware, multi-level memory guidance with adaptive utility pruning.
    """
    def __init__(self, config: Optional[dict] = None):
        super().__init__(MemoryType.TEMPORAL_EXPERIENCE_GRAPH, config)
        self.db_path = self.config.get("database_path",
                                       "../storage/temporal_experience_graph/data.json")
        self.top_k = self.config.get("top_k", 5)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.3)
        self.min_utility_threshold = self.config.get("min_utility_threshold", 0.2)
        self.search_weights = self.config.get("search_weights", {"text": 0.4, "semantic": 0.6})
        self.pruning_interval = self.config.get("pruning_interval", 50)
        self.model_cache_dir = self.config.get("model_cache_dir", "../storage/models")
        self.model = self.config.get("model", None)

        self.embedding_model = self._load_embedding_model()
        self.graph: Optional[TemporalExperienceGraph] = None
        self._ingestion_count = 0
        self._recently_retrieved: Dict[str, List[str]] = {}

    def _load_embedding_model(self) -> SentenceTransformer:
        cache_dir = self.model_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        local_path = os.path.join(cache_dir, model_name.replace('/', '_'))
        try:
            if os.path.exists(local_path) and os.listdir(local_path):
                return SentenceTransformer(local_path)
        except:
            pass
        model = SentenceTransformer(model_name)
        model.save(local_path)
        return model

    def initialize(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            if not os.path.exists(self.db_path):
                with open(self.db_path, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            self.graph = TemporalExperienceGraph(
                embedding_model=self.embedding_model,
                tfidf_vectorizer=TfidfVectorizer(stop_words='english')
            )
            self._load_from_disk()
            self.graph.rebuild_indices()
            self._run_pruning_if_needed()
            return True
        except Exception as e:
            print(f"Error initializing TemporalExperienceGraphProvider: {e}")
            return False

    def _load_from_disk(self):
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    entry = MemoryEntry(
                        id=item['id'],
                        query=item['query'],
                        strategy=item.get('strategy', ''),
                        methodology=item.get('methodology', ''),
                        detailed_steps=item.get('detailed_steps', ''),
                        experience=item.get('experience', ''),
                        support_count=item.get('support_count', 1),
                        retrieval_count=item.get('retrieval_count', 0),
                        success_count=item.get('success_count', 0),
                        last_retrieved=item.get('last_retrieved', 0.0),
                        created=item.get('created', time.time())
                    )
                    self.graph.add_entry(entry)
        except Exception as e:
            print(f"Error loading from disk: {e}")

    def _save_to_disk(self):
        try:
            data = []
            for entry in self.graph.entries.values():
                data.append({
                    'id': entry.id,
                    'query': entry.query,
                    'strategy': entry.strategy,
                    'methodology': entry.methodology,
                    'detailed_steps': entry.detailed_steps,
                    'experience': entry.experience,
                    'support_count': entry.support_count,
                    'retrieval_count': entry.retrieval_count,
                    'success_count': entry.success_count,
                    'last_retrieved': entry.last_retrieved,
                    'created': entry.created
                })
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving to disk: {e}")

    def _run_pruning_if_needed(self):
        """Remove low-utility entries periodically."""
        if len(self.graph.entries) < 10:
            return
        # every pruning_interval ingestions
        if self._ingestion_count % self.pruning_interval != 0:
            return
        current_time = time.time()
        to_remove = []
        for eid, entry in self.graph.entries.items():
            # low retrieval count and never retrieved recently
            if entry.retrieval_count < 3 and current_time - entry.last_retrieved > 86400 * 30:
                to_remove.append(eid)
                continue
            # if retrieval count >0 but success rate < 0.3
            if entry.retrieval_count > 0:
                success_rate = entry.success_count / entry.retrieval_count
                if success_rate < 0.3 and current_time - entry.created > 86400 * 7:
                    to_remove.append(eid)
                    continue
        for eid in to_remove:
            self.graph.remove_entry(eid)
        if to_remove:
            self.graph.rebuild_indices()
            self._save_to_disk()

    def _refine_query(self, query: str) -> str:
        """Use LLM to extract key concepts for better retrieval."""
        if not self.model:
            return query
        prompt = f"""Extract the core concepts and keywords from the following user query. Output only the refined query (short phrase or list of keywords), no explanation.

Query: {query}
Refined:"""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            response = self.model(messages)
            refined = getattr(response, "content", str(response)).strip()
            return refined if refined else query
        except:
            return query

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        if not self.graph or not self.graph.entries:
            return MemoryResponse(memories=[], memory_type=self.memory_type,
                                  total_count=0, request_id=str(uuid.uuid4()))

        try:
            refined_query = self._refine_query(request.query)
            query_embedding = self.embedding_model.encode(refined_query, convert_to_numpy=True)
            candidates = self.graph.hybrid_search(
                query_embedding=query_embedding,
                query_text=refined_query,
                top_k=self.top_k,
                weights=self.search_weights
            )

            if not candidates:
                return MemoryResponse(memories=[], memory_type=self.memory_type,
                                      total_count=0, request_id=str(uuid.uuid4()))

            # Update retrieval stats
            for entry in candidates:
                entry.retrieval_count += 1
                entry.last_retrieved = time.time()

            # Generate memory content based on phase
            if request.status == MemoryStatus.BEGIN:
                # Use strategy & methodology (high/mid level)
                text_lines = []
                for i, entry in enumerate(candidates[:3]):
                    text_lines.append(f"## Strategy {i+1}")
                    text_lines.append(entry.strategy)
                    text_lines.append("")
                    text_lines.append(f"## Methodology {i+1}")
                    text_lines.append(entry.methodology)
                    text_lines.append("")
                content = "\n".join(text_lines) if text_lines else "No guidance available."
            else:  # IN phase
                # Use detailed steps & experience
                text_lines = []
                for i, entry in enumerate(candidates[:3]):
                    text_lines.append(f"## Detailed Steps {i+1}")
                    text_lines.append(entry.detailed_steps)
                    text_lines.append("")
                    text_lines.append(f"## Key Experience {i+1}")
                    text_lines.append(entry.experience)
                    text_lines.append("")
                content = "\n".join(text_lines) if text_lines else "No execution guidance available."

            memory_item = MemoryItem(
                id=f"teg_{uuid.uuid4()}",
                content=content,
                metadata={
                    'num_sources': len(candidates),
                    'source_ids': [e.id for e in candidates],
                    'phase': request.status.value,
                    'refined_query': refined_query
                },
                score=sum(len(e.strategy) for e in candidates) / max(len(candidates), 1)
            )
            # save stats to disk eventually
            self._save_to_disk()

            return MemoryResponse(
                memories=[memory_item],
                memory_type=self.memory_type,
                total_count=1,
                request_id=str(uuid.uuid4())
            )
        except Exception as e:
            print(f"Error in provide_memory: {e}")
            return MemoryResponse(memories=[], memory_type=self.memory_type,
                                  total_count=0, request_id=str(uuid.uuid4()))

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        try:
            if not self._is_task_successful(trajectory_data):
                return False, "Skipping ingestion - task not successful"

            # Generate multi-level abstraction using LLM
            summary = self._summarize_trajectory(trajectory_data)
            if not summary:
                return False, "Failed to generate memory summary"

            # Check for redundancy with existing entries
            query_emb = self.embedding_model.encode(trajectory_data.query, convert_to_numpy=True)
            best_match = None
            best_score = 0.0
            for entry in self.graph.entries.values():
                if entry.query_embedding is not None:
                    sim = cosine_similarity([query_emb], [entry.query_embedding])[0][0]
                    if sim > 0.85 and sim > best_score:
                        best_score = sim
                        best_match = entry

            if best_match:
                # Merge new information into existing entry
                best_match.support_count += 1
                # Append new methodology and detailed steps if not already present
                if summary['methodology'] and summary['methodology'] not in best_match.methodology:
                    best_match.methodology += "\n" + summary['methodology']
                if summary['detailed_steps'] and summary['detailed_steps'] not in best_match.detailed_steps:
                    best_match.detailed_steps += "\n" + summary['detailed_steps']
                if summary['experience'] and summary['experience'] not in best_match.experience:
                    best_match.experience += "\n" + summary['experience']
                if summary['strategy'] and summary['strategy'] not in best_match.strategy:
                    best_match.strategy = summary['strategy']  # keep latest strategy
                self._save_to_disk()
                self._ingestion_count += 1
                return True, f"Merged into existing entry {best_match.id}"

            # Create new entry
            entry = MemoryEntry(
                id=str(uuid.uuid4()),
                query=trajectory_data.query,
                strategy=summary.get('strategy', ''),
                methodology=summary.get('methodology', ''),
                detailed_steps=summary.get('detailed_steps', ''),
                experience=summary.get('experience', ''),
                support_count=1,
                created=time.time()
            )
            # Compute embeddings
            entry.query_embedding = self.embedding_model.encode(entry.query, convert_to_numpy=True)
            self.graph.add_entry(entry)
            self._ingestion_count += 1
            self.graph.rebuild_indices()
            self._save_to_disk()
            self._run_pruning_if_needed()
            return True, f"Stored new memory entry {entry.id}"
        except Exception as e:
            return False, f"Error taking in memory: {e}"

    def _is_task_successful(self, td: TrajectoryData) -> bool:
        metadata = td.metadata or {}
        return metadata.get('is_correct', False) or metadata.get('success', False)

    def _summarize_trajectory(self, td: TrajectoryData) -> Optional[Dict[str, str]]:
        if not self.model:
            return None
        trajectory_text = self._format_trajectory(td)
        prompt = f"""You are an expert AI trainer analyzing a successful task execution. Extract memory at three abstraction levels.

TASK: {td.query}
RESULT: {td.result}
TRAJECTORY:
{trajectory_text}

Generate the following JSON output:
{{
  "strategy": "One-sentence high-level strategy essence.",
  "methodology": "2-3 sentence mid-level workflow description.",
  "detailed_steps": "Numbered step-by-step execution with tool choices and reasoning.",
  "experience": "Key lessons learned, best practices, and pitfalls to avoid."
}}

Ensure each field is substantial. Output ONLY the JSON."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            response = self.model(messages)
            text = getattr(response, "content", str(response)).strip()
            # Extract JSON
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                required = ['strategy', 'methodology', 'detailed_steps', 'experience']
                if all(k in data and len(data[k]) >= 50 for k in required):
                    return data
        except:
            pass
        return None

    def _format_trajectory(self, td: TrajectoryData) -> str:
        if not td.trajectory:
            return "No steps."
        lines = []
        for i, step in enumerate(td.trajectory):
            lines.append(f"Step {i+1}: {step.get('type','')} - {step.get('content','')}")
        return "\n".join(lines)