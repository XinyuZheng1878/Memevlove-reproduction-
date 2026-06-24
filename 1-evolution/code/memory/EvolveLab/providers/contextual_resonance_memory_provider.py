import json
import os
import uuid
import sys
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from ..base_memory import BaseMemoryProvider
from ..memory_types import (
    MemoryRequest,
    MemoryResponse,
    TrajectoryData,
    MemoryType,
    MemoryItem,
    MemoryStatus,
    MemoryItemType
)


@dataclass
class ResonanceEntry:
    """A single memory entry with multiple fields and metadata."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    planning: str = ""
    experience: str = ""
    failure_patterns: str = ""
    embedding_query: Optional[np.ndarray] = None
    embedding_planning: Optional[np.ndarray] = None
    embedding_experience: Optional[np.ndarray] = None
    embedding_failure: Optional[np.ndarray] = None
    is_successful: bool = True
    version: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    success_count: int = 0


class ContextualResonanceEngine:
    """
    Core engine managing memory storage, indexing, retrieval and ingestion.
    Implements multi-field indexing, phase-aware retrieval, incremental updates, and deduplication.
    """

    def __init__(self, db_path: str, embed_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 model_cache_dir: str = '../storage/models', top_k: int = 5,
                 tfidf_weight: float = 0.3, semantic_weight: float = 0.7,
                 dedup_threshold: float = 0.95):
        self.db_path = db_path
        self.tfidf_weight = tfidf_weight
        self.semantic_weight = semantic_weight
        self.top_k = top_k
        self.dedup_threshold = dedup_threshold

        # Embedding model (cached locally)
        self.embedding_model = self._load_embedding_model(embed_model_name, model_cache_dir)

        # In-memory store
        self.entries: Dict[str, ResonanceEntry] = {}
        self.entry_order: List[str] = []  # maintain insertion order

        # TF-IDF index (only for query field)
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self.tfidf_entries: List[str] = []

        # Version tracking for incremental indexing
        self.global_version = 0
        self.last_indexed_version = -1

        # Field list for multi-field retrieval
        self.fields = ['query', 'planning', 'experience', 'failure_patterns']

        # Load existing data
        self._load()

    def _load_embedding_model(self, model_name: str, cache_dir: str) -> SentenceTransformer:
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, model_name.replace('/', '_'))
        try:
            if os.path.exists(local_path) and os.listdir(local_path):
                model = SentenceTransformer(local_path)
                return model
        except Exception:
            pass
        model = SentenceTransformer(model_name)
        model.save(local_path)
        return model

    def _load(self):
        """Load memory from JSON file, if exists."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    raw_list = json.load(f)
                for raw in raw_list:
                    entry = ResonanceEntry(
                        entry_id=raw.get('entry_id', str(uuid.uuid4())),
                        query=raw.get('query', ''),
                        planning=raw.get('planning', ''),
                        experience=raw.get('experience', ''),
                        failure_patterns=raw.get('failure_patterns', ''),
                        is_successful=raw.get('is_successful', True),
                        version=raw.get('version', 1),
                        timestamp=raw.get('timestamp', datetime.now().isoformat()),
                        access_count=raw.get('access_count', 0),
                        success_count=raw.get('success_count', 0)
                    )
                    self.entries[entry.entry_id] = entry
                    self.entry_order.append(entry.entry_id)
                self.global_version = len(self.entries)
            except Exception as e:
                print(f"Warning: Failed to load memory database: {e}")

    def _save(self):
        """Persist all entries to JSON file."""
        data = []
        for entry in self.entries.values():
            data.append({
                'entry_id': entry.entry_id,
                'query': entry.query,
                'planning': entry.planning,
                'experience': entry.experience,
                'failure_patterns': entry.failure_patterns,
                'is_successful': entry.is_successful,
                'version': entry.version,
                'timestamp': entry.timestamp,
                'access_count': entry.access_count,
                'success_count': entry.success_count
            })
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_entry(self, entry: ResonanceEntry) -> bool:
        """Add a new memory entry with deduplication check."""
        # Deduplication
        if self._is_duplicate(entry):
            return False

        # Compute embeddings lazily
        self._compute_embeddings(entry)

        self.entries[entry.entry_id] = entry
        self.entry_order.append(entry.entry_id)
        self.global_version = len(self.entries)
        self._save()
        return True

    def _compute_embeddings(self, entry: ResonanceEntry):
        """Compute embeddings for all fields of the entry."""
        texts = [entry.query, entry.planning, entry.experience, entry.failure_patterns]
        if any(texts):
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            entry.embedding_query = embeddings[0]
            entry.embedding_planning = embeddings[1]
            entry.embedding_experience = embeddings[2]
            entry.embedding_failure = embeddings[3]

    def _is_duplicate(self, entry: ResonanceEntry) -> bool:
        """Check if entry is a near-duplicate of an existing entry."""
        if len(self.entries) < 2:
            return False
        # Use query embedding for comparison (most discriminative)
        if entry.embedding_query is None:
            self._compute_embeddings(entry)
        for existing in self.entries.values():
            if existing.embedding_query is None:
                self._compute_embeddings(existing)
            sim = cosine_similarity([entry.embedding_query], [existing.embedding_query])[0][0]
            if sim > self.dedup_threshold:
                # Merge new insights into existing entry
                if entry.failure_patterns and entry.failure_patterns not in existing.failure_patterns:
                    existing.failure_patterns += "\n" + entry.failure_patterns
                existing.access_count += 1
                self._save()
                return True
        return False

    def _build_tfidf_index(self):
        """Build TF-IDF index on all query texts."""
        queries = [entry.query for entry in self.entries.values()]
        if not queries:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            self.tfidf_entries = []
            return
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(queries)
        self.tfidf_entries = list(self.entries.keys())

    def _ensure_index(self):
        """Rebuild TF-IDF index if versions mismatch (incremental lazy rebuild)."""
        if self.global_version != self.last_indexed_version:
            self._build_tfidf_index()
            self.last_indexed_version = self.global_version

    def _retrieve_by_field(self, query: str, field: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve entries using both TF-IDF (query only) and embedding similarity on the given field.
        Returns list of (entry_id, combined_score) sorted descending.
        """
        self._ensure_index()
        candidate_scores = defaultdict(float)

        # 1. TF-IDF (only valid for 'query' field)
        if field == 'query' and self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
            query_vec = self.tfidf_vectorizer.transform([query])
            tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            for idx, score in enumerate(tfidf_scores):
                eid = self.tfidf_entries[idx]
                candidate_scores[eid] += self.tfidf_weight * score

        # 2. Semantic embedding on the specific field
        query_emb = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        embed_attr = f'embedding_{field}'
        for eid, entry in self.entries.items():
            emb = getattr(entry, embed_attr, None)
            if emb is not None:
                sim = cosine_similarity([query_emb], [emb])[0][0]
                candidate_scores[eid] += self.semantic_weight * sim

        # Sort, filter low scores, take top_k
        sorted_eids = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        # Remove entries with zero or negative score
        filtered = [(eid, score) for eid, score in sorted_eids if score > 0.05][:top_k]
        return filtered

    def hybrid_retrieve(self, query: str, phase: MemoryStatus, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Multi-field retrieval with phase-aware field weighting.
        Returns list of memory content dictionaries.
        """
        # Phase-aware field priorities
        if phase == MemoryStatus.BEGIN:
            field_weights = {'query': 0.4, 'planning': 0.4, 'experience': 0.1, 'failure_patterns': 0.1}
        else:  # IN phase
            field_weights = {'query': 0.2, 'planning': 0.2, 'experience': 0.3, 'failure_patterns': 0.3}

        combined = defaultdict(float)
        for field, weight in field_weights.items():
            results = self._retrieve_by_field(query, field, top_k * 2)
            for eid, score in results:
                combined[eid] += weight * score

        # Sort and return details
        sorted_eids = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        output = []
        for eid, total_score in sorted_eids:
            entry = self.entries.get(eid)
            if not entry:
                continue
            entry.access_count += 1
            output.append({
                'entry_id': eid,
                'score': total_score,
                'query': entry.query,
                'planning': entry.planning,
                'experience': entry.experience,
                'failure_patterns': entry.failure_patterns,
                'is_successful': entry.is_successful
            })
        return output

    def get_statistics(self) -> Dict:
        return {
            'total_entries': len(self.entries),
            'version': self.global_version,
            'last_indexed_version': self.last_indexed_version
        }


class ContextualResonanceMemoryProvider(BaseMemoryProvider):

    DEFAULT_PROMPTS = {
        'query_refinement': """You are a query analyzer for a memory retrieval system. Extract the core concepts, entities, and intentions from the user query. Output a concise search query (max 50 words) that captures the essence for both semantic and keyword matching.

User query: {{query}}
Refined query:""",
        'student_guidance_begin': """You are an expert assistant analyzing similar past tasks to provide strategic planning guidance. Based on the retrieved memories, produce 2-3 concise, actionable suggestions for the agent's initial planning phase.

Current task: {{query}}

Retrieved memory entries (each with query, planning, experience, failure patterns):
{{memories}}

Output numbered suggestions using gentle language. Do not include markdown or headings.""",
        'teacher_guidance_begin': """You are an experienced AI agent synthesizing multiple experience entries. Provide a unified paragraph (2-3 sentences) of operational guidance, focusing on common pitfalls, best practices, and proven techniques.

Task: {{query}}

Retrieved experiences:
{{memories}}

Output only the guidance text.""",
        'execution_guidance_in': """You are an AI agent assistant providing real-time execution guidance. The agent is currently working on the task. Based on retrieved memories (including cautionary failure patterns), produce 2-3 specific warnings, tips, or intermediate steps.

Task: {{query}}

Agent context (last steps): {{context}}

Retrieved memories (including failure patterns):
{{memories}}

Output numbered items with gentle language. If failure patterns are present, highlight them as 'Caution:' items.""",
        'extraction': """You are a memory extraction expert. Analyze the following task trajectory and produce structured memory fields in JSON format.

Task: {{query}}
Trajectory:
{{trajectory}}
Result: {{result}}

Extract:
1. planning: Detailed, actionable planning steps the agent used (if any). If no planning, output empty string.
2. experience: Key lessons learned, successful methodologies, best practices.
3. failure_patterns: If task failed, extract specific negative patterns (e.g., tool errors, wrong approaches). If task succeeded, output empty string.

Important: Be specific and concrete. Each field must contain at least 1-2 sentences if content exists.

Output JSON only: