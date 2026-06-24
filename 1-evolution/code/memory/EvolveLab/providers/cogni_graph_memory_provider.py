import json
import os
import uuid
import sys
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

# -----------------------------------------------------------------------------
# Graph-based Memory Structures
# -----------------------------------------------------------------------------

@dataclass
class GraphNode:
    node_id: str
    label: str               # short concept name
    text: str                # full content (e.g., strategy description)
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_accessed: float = field(default_factory=lambda: datetime.now().timestamp())
    access_count: int = 0
    utility: float = 1.0     # learned utility score

@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    relation: str            # e.g., "uses", "followed_by", "results_in"
    weight: float = 1.0

@dataclass
class ExperienceGraph:
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: List[GraphEdge] = field(default_factory=list)
    adjacency: Dict[str, List[Tuple[str, str, float]]] = field(default_factory=lambda: defaultdict(list))
    # adjacency: source_id -> [(target_id, relation, weight)]

    def add_node(self, node: GraphNode):
        self.nodes[node.node_id] = node
        if node.node_id not in self.adjacency:
            self.adjacency[node.node_id] = []

    def add_edge(self, edge: GraphEdge):
        self.edges.append(edge)
        self.adjacency[edge.source_id].append((edge.target_id, edge.relation, edge.weight))

    def get_neighbors(self, node_id: str, relation_filter: Optional[str] = None) -> List[Tuple[str, str, float]]:
        neighbors = self.adjacency.get(node_id, [])
        if relation_filter:
            return [(n, r, w) for n, r, w in neighbors if r == relation_filter]
        return neighbors

# -----------------------------------------------------------------------------
# Embedding manager with caching and incremental support
# -----------------------------------------------------------------------------

class EmbeddingManager:
    def __init__(self, cache_dir: str = '../storage/models'):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=cache_dir)
        self.embedding_cache: Dict[str, np.ndarray] = {}

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        uncached_texts = [t for t in texts if t not in self.embedding_cache]
        if uncached_texts:
            new_embs = self.model.encode(uncached_texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
            for t, emb in zip(uncached_texts, new_embs):
                self.embedding_cache[t] = emb
        return np.array([self.embedding_cache[t] for t in texts])

# -----------------------------------------------------------------------------
# Main Memory Manager
# -----------------------------------------------------------------------------

class CogniGraphManager:
    def __init__(self, database_path: str, model, embedding_manager: EmbeddingManager):
        self.database_path = database_path
        self.model = model
        self.embedding_manager = embedding_manager
        self.graph = ExperienceGraph()
        self.indexed_fields = ['query', 'agent_planning', 'search_agent_planning', 'agent_experience', 'search_agent_experience']
        self.text_index: Dict[str, np.ndarray] = {}  # field -> embeddings matrix
        self.text_ids: Dict[str, List[str]] = {}      # field -> list of node_ids
        self.buffer: List[dict] = []
        self.buffer_size = 10   # flush after N ingests
        self.load()

    def load(self):
        if os.path.exists(self.database_path):
            with open(self.database_path, 'r') as f:
                data = json.load(f)
            for entry in data:
                self._ingest_entry(entry)
            self._rebuild_index()

    def save(self):
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        # Serialize graph nodes and edges in a flat list
        serializable = []
        for node in self.graph.nodes.values():
            serializable.append({
                'node_id': node.node_id,
                'label': node.label,
                'text': node.text,
                'metadata': node.metadata,
                'last_accessed': node.last_accessed,
                'access_count': node.access_count,
                'utility': node.utility
            })
        # edges are reconstructed from adjacency; for persistence we store edges separately
        edge_list = []
        for edge in self.graph.edges:
            edge_list.append({
                'source_id': edge.source_id,
                'target_id': edge.target_id,
                'relation': edge.relation,
                'weight': edge.weight
            })
        out = {'nodes': serializable, 'edges': edge_list}
        with open(self.database_path, 'w') as f:
            json.dump(out, f, indent=2)

    def _ingest_entry(self, entry: dict):
        """Add a single experience entry to the graph, merging duplicates."""
        query = entry.get('query', '')
        query_embed = self.embedding_manager.encode([query])[0]

        # Check for duplicate query (cosine > 0.95)
        existing_ids = self.text_ids.get('query', [])
        if existing_ids:
            existing_embs = self.text_index.get('query', np.empty((0, query_embed.shape[0])))
            if existing_embs.shape[0] > 0:
                similarities = cosine_similarity([query_embed], existing_embs)[0]
                max_idx = np.argmax(similarities)
                if similarities[max_idx] > 0.95:
                    # Merge with existing node
                    existing_node_id = existing_ids[max_idx]
                    node = self.graph.nodes.get(existing_node_id)
                    if node:
                        # Append new insights to text (with separator)
                        new_text_parts = []
                        for field in self.indexed_fields:
                            val = entry.get(field, '')
                            if val:
                                new_text_parts.append(val)
                        if new_text_parts:
                            node.text += '\n---\n' + '\n'.join(new_text_parts)
                        node.last_accessed = datetime.now().timestamp()
                        node.access_count += 1
                        # Recompute embedding for combined text
                        combined_embed = self.embedding_manager.encode([node.text])[0]
                        # Update index entry
                        idx = existing_ids.index(existing_node_id)
                        self.text_index['query'][idx] = combined_embed
                        return  # merged, no new node

        # No duplicate: create new node
        node_id = str(uuid.uuid4())
        # Combine all fields into a single text for the node
        combined_text = ' '.join([entry.get(f, '') for f in self.indexed_fields if entry.get(f, '')])
        node = GraphNode(
            node_id=node_id,
            label=query[:50],
            text=combined_text,
            embedding=query_embed,
            metadata=entry.get('metadata', {}),
            last_accessed=datetime.now().timestamp(),
            access_count=1,
            utility=1.0
        )
        self.graph.add_node(node)

        # Extract entities and relations using LLM (for graph structure)
        self._extract_graph_relations(node, entry)

        # Index the node's text for each field (but we store only combined embedding for simplicity)
        for field in self.indexed_fields:
            field_text = entry.get(field, '')
            if field_text:
                # store field-specific embedding in the node metadata
                if 'field_embeddings' not in node.metadata:
                    node.metadata['field_embeddings'] = {}
                emb = self.embedding_manager.encode([field_text])[0]
                node.metadata['field_embeddings'][field] = emb.tolist()
                # Also add to global field index
                if field not in self.text_ids:
                    self.text_ids[field] = []
                    self.text_index[field] = np.empty((0, query_embed.shape[0]))
                self.text_ids[field].append(node_id)
                self.text_index[field] = np.vstack([self.text_index[field], emb])

    def _extract_graph_relations(self, node: GraphNode, entry: dict):
        """Use LLM to extract linked concepts from the entry and add edges."""
        prompt = f"""You are an expert knowledge graph builder. Given the following experience entry, identify key entities (concepts, techniques, tools) and their relationships. Output a JSON list of edges: each edge has "source_label", "target_label", "relation". Use concise labels.

Entry query: {entry.get('query', '')}
Planning: {entry.get('agent_planning', '')}
Experience: {entry.get('agent_experience', '')}

Example output:
[{{"source_label": "web_search", "target_label": "scrape_dynamic_content", "relation": "precedes"}}]

Only output the JSON array, no extra text."""
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            response = self.model(messages)
            resp_text = getattr(response, "content", str(response)).strip()
            import re
            json_match = re.search(r'\[.*?\]', resp_text, re.DOTALL)
            if json_match:
                edges_data = json.loads(json_match.group(0))
            else:
                edges_data = json.loads(resp_text)
            for ed in edges_data:
                src_label = ed.get('source_label', '')
                tgt_label = ed.get('target_label', '')
                rel = ed.get('relation', 'related')
                if not src_label or not tgt_label:
                    continue
                # Find or create nodes for these labels
                src_id = self._get_or_create_label_node(src_label)
                tgt_id = self._get_or_create_label_node(tgt_label)
                if src_id and tgt_id:
                    self.graph.add_edge(GraphEdge(source_id=src_id, target_id=tgt_id, relation=rel))
        except Exception as e:
            print(f"Graph extraction error: {e}")

    def _get_or_create_label_node(self, label: str) -> str:
        """Find existing node with same label or create a new one."""
        for nid, node in self.graph.nodes.items():
            if node.label == label:
                return nid
        node_id = str(uuid.uuid4())
        node = GraphNode(node_id=node_id, label=label, text=label)
        self.graph.add_node(node)
        return node_id

    def _rebuild_index(self):
        """Rebuild text index from graph nodes (used after batch flush)."""
        self.text_index = {f: np.empty((0, 384)) for f in self.indexed_fields}
        self.text_ids = {f: [] for f in self.indexed_fields}
        for node in self.graph.nodes.values():
            node_emb = self.embedding_manager.encode([node.text])[0]
            for field in self.indexed_fields:
                if field in node.metadata.get('field_embeddings', {}):
                    emb = np.array(node.metadata['field_embeddings'][field])
                else:
                    emb = node_emb  # fallback
                self.text_ids[field].append(node.node_id)
                self.text_index[field] = np.vstack([self.text_index[field], emb]) if self.text_index[field].size else emb.reshape(1, -1)

    def add_entry(self, entry: dict):
        self.buffer.append(entry)
        self._ingest_entry(entry)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        self._rebuild_index()
        self.save()
        self.buffer = []

    def search(self, query: str, status: MemoryStatus, top_k: int = 5) -> List[Dict[str, Any]]:
        query_emb = self.embedding_manager.encode([query])[0]

        # Multi-field search with weights
        field_weights = {'query': 1.0, 'agent_planning': 0.8, 'agent_experience': 0.7, 'search_agent_planning': 0.6, 'search_agent_experience': 0.6}
        scores = defaultdict(float)

        for field, weight in field_weights.items():
            ids = self.text_ids.get(field, [])
            emb_mat = self.text_index.get(field)
            if emb_mat is None or emb_mat.shape[0] == 0:
                continue
            sims = cosine_similarity([query_emb], emb_mat)[0]
            for idx, sid in enumerate(ids):
                scores[sid] += weight * sims[idx]

        # Graph-based boost: if status is IN, traverse from top results to find related concepts
        if status == MemoryStatus.IN:
            # Get initial top candidates
            sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            for node_id, base_score in sorted_nodes:
                neighbors = self.graph.get_neighbors(node_id)
                for neighbor_id, relation, w in neighbors:
                    # Boost neighbor with decay factor
                    if neighbor_id in scores:
                        scores[neighbor_id] += 0.3 * base_score * w
                    else:
                        scores[neighbor_id] = 0.2 * base_score * w

        # Sort final scores
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for node_id, score in sorted_nodes:
            node = self.graph.nodes.get(node_id)
            if node:
                node.last_accessed = datetime.now().timestamp()
                node.access_count += 1
                results.append({
                    'node_id': node_id,
                    'label': node.label,
                    'text': node.text,
                    'score': score,
                    'utility': node.utility
                })
        return results

    def prune(self, max_nodes: int = 1000):
        """Remove low-utility nodes when graph exceeds limit."""
        if len(self.graph.nodes) <= max_nodes:
            return
        # compute utility score (access_count * utility / (time_decay))
        now = datetime.now().timestamp()
        node_utils = []
        for nid, node in self.graph.nodes.items():
            time_factor = 1.0 / (1 + 0.01 * (now - node.last_accessed))
            score = node.utility * node.access_count * time_factor
            node_utils.append((nid, score))
        node_utils.sort(key=lambda x: x[1])
        # remove lowest 10% of nodes
        to_remove = set(nid for nid, _ in node_utils[:int(len(node_utils)*0.1)])
        for nid in to_remove:
            self.graph.nodes.pop(nid, None)
        self.graph.edges = [e for e in self.graph.edges if e.source_id not in to_remove and e.target_id not in to_remove]
        for nid in to_remove:
            if nid in self.graph.adjacency:
                del self.graph.adjacency[nid]
        # Rebuild index
        self._rebuild_index()
        self.save()


# -----------------------------------------------------------------------------
# Provider Implementation
# -----------------------------------------------------------------------------

class CogniGraphMemoryProvider(BaseMemoryProvider):
    def __init__(self, config: Optional[dict] = None):
        super().__init__(MemoryType.COGNI_GRAPH_MEMORY, config)
        self.database_path = self.config.get("database_path", "../storage/cogni_graph_memory/graph.json")
        self.top_k = self.config.get("top_k", 5)
        self.buffer_size = self.config.get("buffer_size", 10)
        self.model = self.config.get("model", None)
        self.model_cache_dir = self.config.get("model_cache_dir", "../storage/models")
        self.graph_manager: Optional[CogniGraphManager] = None

    def initialize(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            embed_manager = EmbeddingManager(cache_dir=self.model_cache_dir)
            self.graph_manager = CogniGraphManager(
                database_path=self.database_path,
                model=self.model,
                embedding_manager=embed_manager
            )
            return True
        except Exception as e:
            print(f"Error initializing CogniGraphMemory: {e}")
            return False

    def _build_retrieval_prompt(self, query: str, context: str, status: MemoryStatus) -> str:
        """Prepare the query for searching. If status is IN, include recent context."""
        if status == MemoryStatus.BEGIN:
            return query
        # For IN phase, create a composite query from context (last few steps)
        # Use LLM to extract current sub-goal
        prompt = f"""Given the current step context, what is the immediate sub-task or goal the agent should focus on? Return a short phrase.

Current query: {query}
Recent context (last 2 steps):
{context[:500]}
Sub-task:"""
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            response = self.model(messages)
            sub_task = getattr(response, "content", str(response)).strip()
            return sub_task if sub_task else query
        except:
            return query

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        if not self.graph_manager:
            return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0, request_id=str(uuid.uuid4()))

        try:
            # Use query refinement for BEGIN, sub-goal extraction for IN
            refined = self._build_retrieval_prompt(request.query, request.context, request.status)

            results = self.graph_manager.search(refined, request.status, self.top_k)

            if not results:
                return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0, request_id=str(uuid.uuid4()))

            # Filter low utility or very low score results (adaptive threshold)
            filtered = [r for r in results if r['utility'] >= 0.3 or r['score'] >= 0.2]

            if not filtered:
                return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0, request_id=str(uuid.uuid4()))

            # Synthesize memories from top results using LLM
            synthesized = self._synthesize(filtered, request)
            memory_item = MemoryItem(
                id=f"cogni_{uuid.uuid4()}",
                content=synthesized,
                metadata={
                    'num_sources': len(filtered),
                    'top_score': filtered[0]['score'],
                    'status': request.status.value
                },
                score=filtered[0]['score']
            )
            return MemoryResponse(memories=[memory_item], memory_type=self.memory_type, total_count=1, request_id=str(uuid.uuid4()))
        except Exception as e:
            print(f"Error in provide_memory: {e}")
            return MemoryResponse(memories=[], memory_type=self.memory_type, total_count=0, request_id=str(uuid.uuid4()))

    def _synthesize(self, results: List[Dict], request: MemoryRequest) -> str:
        if not results:
            return ""

        context_str = '\n\n'.join([f"Source {i+1}: {r['text'][:500]}" for i, r in enumerate(results)])

        if request.status == MemoryStatus.BEGIN:
            prompt = f"""You are an AI agent receiving guidance for an upcoming task. Based on the following retrieved past experiences, provide a concise strategic overview and a step-by-step plan for the new task.

Task: {request.query}

Relevant memories:
{context_str}

Output:
- Key strategies from past experiences
- Step-by-step action plan (numbered)
- Common pitfalls to avoid"""
        else:  # IN phase
            prompt = f"""You are an AI agent in the middle of executing a task. Based on the following similar sub-task experiences, provide immediate actionable advice to overcome the current step.

Current task: {request.query}
Recent context: {request.context[:300]}

Relevant memories:
{context_str}

Output:
- Immediate next step suggestion
- Alternative approaches if stuck
- Specific tool or search query recommendations"""

        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            response = self.model(messages)
            return getattr(response, "content", str(response)).strip()
        except:
            return results[0]['text'][:200] if results else ""

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        try:
            # Accept partial success as well (allow bootstrapping)
            success = self._is_task_successful(trajectory_data)
            partial = trajectory_data.metadata.get('partial_success', False)
            if not success and not partial:
                return False, "Task not successful or partial"

            # Use LLM to summarize trajectory into structured fields (similar to AgentKB but add graph-friendly extraction)
            summary = self._summarize_trajectory(trajectory_data)
            if not summary:
                return False, "Summarization failed"

            entry = {
                'query': trajectory_data.query,
                'agent_planning': summary.get('agent_planning', ''),
                'search_agent_planning': summary.get('search_agent_planning', ''),
                'agent_experience': summary.get('agent_experience', ''),
                'search_agent_experience': summary.get('search_agent_experience', ''),
                'metadata': {
                    **trajectory_data.metadata,
                    'ingestion_time': datetime.now().isoformat(),
                    'partial': partial
                }
            }

            self.graph_manager.add_entry(entry)
            # Prune if necessary
            self.graph_manager.prune(max_nodes=1000)
            return True, "Ingested with graph integration"
        except Exception as e:
            return False, f"Error: {e}"

    def _is_task_successful(self, trajectory_data: TrajectoryData) -> bool:
        md = trajectory_data.metadata or {}
        for key in ['is_correct', 'success', 'task_success']:
            if md.get(key) is True:
                return True
        return False

    def _summarize_trajectory(self, trajectory_data: TrajectoryData) -> Optional[Dict[str, str]]:
        if not self.model:
            return None
        # Build trajectory text
        traj_text = ''
        if trajectory_data.trajectory:
            for step in trajectory_data.trajectory:
                traj_text += f"{step.get('type', 'step')}: {step.get('content', '')}\n"
        if trajectory_data.result:
            traj_text += f"Result: {trajectory_data.result}\n"

        prompt = f"""You are an expert AI agent trainer. Analyze the following task execution trajectory and extract structured memory patterns. Output JSON with fields: agent_planning (strategic plan), search_agent_planning (search strategy), agent_experience (key lessons), search_agent_experience (search-specific insights). Each field must be detailed (≥2 sentences).

Trajectory:
{traj_text}

Task query: {trajectory_data.query}

Output only JSON object."""
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            response = self.model(messages)
            resp_text = getattr(response, "content", str(response)).strip()
            import re
            json_match = re.search(r'\{.*?\}', resp_text, re.DOTALL)
            if json_match:
                summary = json.loads(json_match.group(0))
            else:
                summary = json.loads(resp_text)
            required_fields = ['agent_planning', 'search_agent_planning', 'agent_experience', 'search_agent_experience']
            if all(f in summary and len(summary[f].strip()) >= 50 for f in required_fields):
                return summary
            return None
        except:
            return None