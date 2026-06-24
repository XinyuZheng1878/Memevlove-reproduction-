import json
import os
import uuid
import sys
from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import re
import numpy as np

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
from storage.tools.tool_wrapper import ToolWrapper


@dataclass
class ExperienceNode:
    """Represents a stored experience with multiple abstraction levels"""
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Multi-level abstractions
    concrete_plan: str = ""  # Task-specific steps
    abstract_plan: str = ""  # Generalizable strategy
    concrete_experience: str = ""  # Specific lessons
    abstract_experience: str = ""  # General principles
    search_strategy: str = ""  # Search-specific patterns
    
    # Utility tracking
    utility_score: float = 0.5  # Initial neutral score
    retrieval_count: int = 0
    success_count: int = 0
    last_retrieved: Optional[str] = None
    
    # Metadata
    task_success: bool = True
    partial_success: bool = False
    failure_reason: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Tool memory
    extracted_tools: List[Dict[str, str]] = field(default_factory=list)


class ResonantExperienceEngine:
    """Core engine that manages experience storage, retrieval, and evolution"""

    def __init__(self, database_path: str, model=None, tool_wrapper: Optional[ToolWrapper] = None):
        self.database_path = database_path
        self.model = model
        self.tool_wrapper = tool_wrapper
        self.experiences: Dict[str, ExperienceNode] = {}
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        self._load_database()

    def _load_database(self):
        """Load existing experiences from disk"""
        try:
            if os.path.exists(self.database_path):
                with open(self.database_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        node = ExperienceNode(
                            experience_id=item.get('experience_id', str(uuid.uuid4())),
                            query=item.get('query', ''),
                            timestamp=item.get('timestamp', datetime.now().isoformat()),
                            concrete_plan=item.get('concrete_plan', ''),
                            abstract_plan=item.get('abstract_plan', ''),
                            concrete_experience=item.get('concrete_experience', ''),
                            abstract_experience=item.get('abstract_experience', ''),
                            search_strategy=item.get('search_strategy', ''),
                            utility_score=item.get('utility_score', 0.5),
                            retrieval_count=item.get('retrieval_count', 0),
                            success_count=item.get('success_count', 0),
                            task_success=item.get('task_success', True),
                            partial_success=item.get('partial_success', False),
                            failure_reason=item.get('failure_reason'),
                            tags=item.get('tags', []),
                            extracted_tools=item.get('extracted_tools', [])
                        )
                        self.experiences[node.experience_id] = node
                        for tag in node.tags:
                            self.tag_index[tag].append(node.experience_id)
        except (FileNotFoundError, json.JSONDecodeError):
            self.experiences = {}
            self.tag_index = defaultdict(list)

    def _save_database(self):
        """Persist experiences to disk"""
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        data = []
        for node in self.experiences.values():
            data.append({
                'experience_id': node.experience_id,
                'query': node.query,
                'timestamp': node.timestamp,
                'concrete_plan': node.concrete_plan,
                'abstract_plan': node.abstract_plan,
                'concrete_experience': node.concrete_experience,
                'abstract_experience': node.abstract_experience,
                'search_strategy': node.search_strategy,
                'utility_score': node.utility_score,
                'retrieval_count': node.retrieval_count,
                'success_count': node.success_count,
                'task_success': node.task_success,
                'partial_success': node.partial_success,
                'failure_reason': node.failure_reason,
                'tags': node.tags,
                'extracted_tools': node.extracted_tools
            })
        with open(self.database_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def retrieve_relevant_experiences(self, query: str, context: str, phase: str, top_k: int = 3) -> List[ExperienceNode]:
        """
        LLM-driven retrieval that reasons about query and context to select most relevant experiences.
        No hardcoded matching - uses model to think about what's needed.
        """
        if not self.model or not self.experiences:
            return []
        
        # Build compact experience catalog for LLM to reason about
        catalog = []
        for eid, node in self.experiences.items():
            entry = {
                'id': eid[:8],
                'query': node.query[:150],
                'utility': round(node.utility_score, 2)
            }
            # Add abstraction levels based on phase
            if phase == 'BEGIN':
                entry['strategy'] = node.abstract_plan[:200] if node.abstract_plan else node.concrete_plan[:200]
            else:
                entry['guidance'] = node.abstract_experience[:200] if node.abstract_experience else node.concrete_experience[:200]
            
            if node.tags:
                entry['tags'] = node.tags[:3]
            if node.failure_reason:
                entry['failure'] = node.failure_reason[:100]
            
            catalog.append(entry)
        
        retriever_prompt = f"""You are an experience retrieval specialist for an AI agent. Your task is to analyze the current task and determine which past experiences are most relevant.

CURRENT TASK:
Query: {query[:300]}
Phase: {phase}
Context: {context[:500]}

EXPERIENCE CATALOG:
{json.dumps(catalog, indent=2)}

RETRIEVAL INSTRUCTIONS:
1. Analyze the current task's core challenges, domain, and requirements
2. Evaluate each experience for relevance based on:
   - Semantic similarity to the query
   - Complementary strategies that could help
   - Past failure patterns to avoid
   - Utility score (higher is better)
3. Select the TOP {top_k} most relevant experiences
4. For each selected experience, explain briefly why it's relevant

Output ONLY a JSON object with format:
{{"selected_ids": ["id1", "id2", ...], "reasoning": "brief explanation of selections"}}"""

        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": retriever_prompt}]}]
            response = self.model(messages)
            result = getattr(response, "content", str(response)).strip()
            
            # Extract JSON
            json_match = re.search(r'\{.*?\}', result, re.DOTALL)
            if json_match:
                selection = json.loads(json_match.group(0))
            else:
                selection = json.loads(result)
            
            selected_ids = selection.get('selected_ids', [])
            full_ids = []
            for sid in selected_ids:
                for eid, node in self.experiences.items():
                    if eid.startswith(sid):
                        full_ids.append(eid)
                        break
            
            results = [self.experiences[eid] for eid in full_ids if eid in self.experiences]
            
            # Update retrieval counts
            for node in results:
                node.retrieval_count += 1
                node.last_retrieved = datetime.now().isoformat()
            
            self._save_database()
            return results[:top_k]
            
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

    def add_experience(self, node: ExperienceNode):
        """Add new experience with deduplication"""
        # Check for near-duplicate queries
        for existing in self.experiences.values():
            if existing.query.strip().lower() == node.query.strip().lower():
                # Merge: keep higher quality content
                if node.task_success and not existing.task_success:
                    self.experiences[existing.experience_id] = node
                    return True
                elif node.partial_success and not existing.partial_success:
                    existing.partial_success = True
                    existing.concrete_plan = existing.concrete_plan or node.concrete_plan
                    existing.concrete_experience = existing.concrete_experience or node.concrete_experience
                else:
                    return False  # Duplicate, skip
        
        self.experiences[node.experience_id] = node
        for tag in node.tags:
            self.tag_index[tag].append(node.experience_id)
        self._save_database()
        return True

    def update_utility(self, experience_id: str, contributed_to_success: bool):
        """Update utility score based on retrieval outcome"""
        node = self.experiences.get(experience_id)
        if node:
            if contributed_to_success:
                node.success_count += 1
                node.utility_score = min(1.0, node.utility_score + 0.1)
            else:
                node.utility_score = max(0.0, node.utility_score - 0.05)
            
            node.retrieval_count += 1
            self._save_database()

    def prune_low_utility(self, threshold: float = 0.2):
        """Remove experiences with very low utility"""
        to_remove = []
        for eid, node in self.experiences.items():
            if node.utility_score < threshold and node.retrieval_count > 5:
                to_remove.append(eid)
        
        for eid in to_remove:
            node = self.experiences.pop(eid)
            for tag in node.tags:
                if eid in self.tag_index.get(tag, []):
                    self.tag_index[tag].remove(eid)
            print(f"Pruned low-utility experience: {eid[:8]}")
        
        if to_remove:
            self._save_database()


class ResonantExperienceProvider(BaseMemoryProvider):
    """
    A self-evolving memory system that stores multi-level abstractions from trajectories
    and provides decision-oriented retrieval using LLM reasoning.
    
    Key Innovations:
    1. LLM-driven retrieval that reasons about relevance (no embeddings required)
    2. Multi-level abstraction storage (concrete steps + general principles)
    3. Partial success ingestion (learn from near-misses)
    4. Experience-to-tool extraction pipeline
    5. Utility-based pruning and evolution
    """

    def __init__(self, config: Optional[dict] = None):
        super().__init__(MemoryType.RESONANT_EXPERIENCE, config)
        
        self.database_path = self.config.get(
            "database_path",
            "../storage/resonant_experience/resonant_experiences.json"
        )
        self.top_k = self.config.get("top_k", 3)
        self.model = self.config.get("model", None)
        self.partial_success_threshold = self.config.get("partial_success_threshold", 0.4)
        self.min_utility_before_prune = self.config.get("min_utility_before_prune", 0.2)
        self.enable_tool_extraction = self.config.get("enable_tool_extraction", True)
        
        self.engine: Optional[ResonantExperienceEngine] = None
        self.tool_wrapper: Optional[ToolWrapper] = None
        
        # Fallback memories for cold start
        self.fallback_memories = {
            "BEGIN": [
                "When approaching a new task, first identify the core question and break it down into sub-problems.",
                "Use progressive information gathering: start with broad searches, then narrow down based on findings.",
                "If a search tool fails or returns no results, try alternative queries with different keywords or sources.",
                "Consider the domain of the question: academic, factual, or procedural, and adapt your search strategy accordingly."
            ],
            "IN": [
                "If current approach is failing, consider switching to alternative data sources or search methods.",
                "When encountering tool errors (quota limits, SSL failures), try alternative approaches like direct URL access.",
                "Verify information from multiple sources before committing to an answer.",
                "If stuck, backtrack to the last known successful step and try a different path."
            ]
        }
    
    def initialize(self) -> bool:
        """Initialize the memory system"""
        try:
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            self.engine = ResonantExperienceEngine(
                database_path=self.database_path,
                model=self.model,
                tool_wrapper=self.tool_wrapper
            )
            
            if self.enable_tool_extraction and self.model:
                try:
                    self.tool_wrapper = ToolWrapper(model=self.model, logger=self.logger)
                except Exception as e:
                    print(f"Tool wrapper initialization warning: {e}")
            
            return True
        except Exception as e:
            print(f"Initialization error: {e}")
            return False
    
    def _generate_retrieval_goal(self, query: str, context: str, phase: str) -> str:
        """
        Use LLM to generate a focused retrieval goal that guides what kind of memories to seek.
        This provides better targeting than raw query matching.
        """
        if not self.model:
            return query
        
        goal_prompt = f"""
Given the current task and agent context, determine what type of past experience would be most helpful.

Task Query: {query[:200]}
Current Phase: {phase}
Agent Context: {context[:300]}

Based on this analysis, generate a concise retrieval goal (1-2 sentences) describing what kind of 
past experiences would be most valuable. Focus on:
- What strategies might work
- What pitfalls to avoid
- What tools or approaches to try

Output only the retrieval goal, nothing else.
"""
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": goal_prompt}]}]
            response = self.model(messages)
            goal = getattr(response, "content", str(response)).strip()
            return goal if goal else query
        except Exception as e:
            print(f"Retrieval goal generation error: {e}")
            return query
    
    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        """Provide memories based on phase and context"""
        if not self.engine:
            return MemoryResponse(
                memories=[],
                memory_type=self.memory_type,
                total_count=0,
                request_id=str(uuid.uuid4())
            )
        
        try:
            phase = request.status.value
            retrieval_goal = self._generate_retrieval_goal(
                request.query, 
                request.context or "", 
                phase
            )
            
            # Retrieve relevant experiences
            experiences = self.engine.retrieve_relevant_experiences(
                query=retrieval_goal,
                context=request.context or "",
                phase=phase,
                top_k=self.top_k
            )
            
            text_memories = []
            tool_memories = []
            
            if not experiences:
                # Intelligent fallback: decide dynamically whether to use generic memories
                if self.model:
                    fallback_decision = self._decide_fallback(request)
                    if fallback_decision:
                        text_memories = self._create_fallback_memories(request)
                else:
                    # Without model, use fallback as safety net
                    text_memories = self._create_fallback_memories(request)
            else:
                # Synthesize retrieved experiences into actionable guidance
                guidance = self._synthesize_guidance(experiences, request)
                if guidance:
                    text_memories.append(MemoryItem(
                        id=f"exp_synth_{uuid.uuid4()}",
                        content=guidance,
                        metadata={
                            'num_sources': len(experiences),
                            'phase': phase,
                            'retrieval_goal': retrieval_goal,
                            'experience_ids': [e.experience_id[:8] for e in experiences]
                        },
                        score=sum(e.utility_score for e in experiences) / len(experiences),
                        type=MemoryItemType.TEXT
                    ))
                
                # Extract tool memories if available
                if self.enable_tool_extraction and self.tool_wrapper:
                    tool_memories = self._extract_tool_memories(experiences, request)
            
            # Combine memories
            all_memories = text_memories + tool_memories
            
            return MemoryResponse(
                memories=all_memories,
                memory_type=self.memory_type,
                total_count=len(all_memories),
                request_id=str(uuid.uuid4())
            )
            
        except Exception as e:
            print(f"Provide memory error: {e}")
            return self._fallback_response(request)
    
    def _decide_fallback(self, request: MemoryRequest) -> bool:
        """Dynamically decide whether to use fallback memories"""
        decision_prompt = f"""
Task Query: {request.query[:200]}
Phase: {request.status.value}
Context: {request.context[:300]}

Should we provide generic fallback guidance to help the agent? Consider:
- Is this a complex task requiring strategy?
- Is the agent at risk of making common mistakes?
- Would generic guidance be better than nothing?

Answer only 'yes' or 'no' with brief reason.
"""
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": decision_prompt}]}]
            response = self.model(messages)
            result = getattr(response, "content", str(response)).strip().lower()
            return 'yes' in result[:10]
        except:
            return True  # Safe default
    
    def _create_fallback_memories(self, request: MemoryRequest) -> List[MemoryItem]:
        """Create intelligent fallback memories"""
        phase = request.status.value
        fallback_content = self.fallback_memories.get(phase, [])
        
        if self.model and fallback_content:
            # Dynamically select most relevant fallback
            selection_prompt = f"""
Query: {request.query[:200]}
Phase: {phase}

Available guidelines:
{chr(10).join(f'{i+1}. {g}' for i, g in enumerate(fallback_content))}

Select the 2 most relevant guidelines for this task. Return only numbers separated by commas.
"""
            try:
                messages = [{"role": "user", "content": [{"type": "text", "text": selection_prompt}]}]
                response = self.model(messages)
                result = getattr(response, "content", str(response)).strip()
                indices = [int(x) for x in re.findall(r'\d+', result) if 1 <= int(x) <= len(fallback_content)]
                selected = [fallback_content[i-1] for i in indices[:2]]
            except:
                selected = fallback_content[:2]
        else:
            selected = fallback_content[:2]
        
        return [
            MemoryItem(
                id=f"fallback_{uuid.uuid4().hex[:8]}",
                content=content,
                metadata={'type': 'fallback', 'phase': phase},
                score=0.3,
                type=MemoryItemType.TEXT
            )
            for content in selected
        ]
    
    def _synthesize_guidance(self, experiences: List[ExperienceNode], request: MemoryRequest) -> Optional[str]:
        """Synthesize multiple experiences into coherent guidance"""
        if not experiences:
            return None
        
        if not self.model:
            # Simple concatenation without model
            parts = []
            for exp in experiences[:2]:
                if request.status == MemoryStatus.BEGIN:
                    content = exp.abstract_plan or exp.concrete_plan
                else:
                    content = exp.abstract_experience or exp.concrete_experience
                if content:
                    parts.append(content[:200])
            return "\n---\n".join(parts) if parts else None
        
        # Build synthesis prompt
        experiences_text = []
        for i, exp in enumerate(experiences[:3]):
            entry = f"Experience {i+1} (utility: {exp.utility_score:.2f}):\n"
            if request.status == MemoryStatus.BEGIN:
                entry += f"Strategy: {exp.abstract_plan[:300]}\n"
                entry += f"Details: {exp.concrete_plan[:300]}"
            else:
                entry += f"Guidance: {exp.abstract_experience[:300]}\n"
                entry += f"Details: {exp.concrete_experience[:300]}"
            
            if exp.search_strategy:
                entry += f"\nSearch: {exp.search_strategy[:200]}"
            if exp.tags:
                entry += f"\nTags: {', '.join(exp.tags[:3])}"
            experiences_text.append(entry)
        
        prompt = f"""
You are synthesizing past experiences to guide an AI agent on a current task.

Current Task: {request.query[:200]}
Current Phase: {request.status.value}

Relevant Past Experiences:
{chr(10).join(experiences_text)}

Synthesize a concise, actionable guidance (2-4 sentences) that:
1. Focuses on the most relevant strategies/insights
2. Provides specific recommendations applicable to the current task
3. Warns about any pitfalls to avoid
4. Is written in imperative but helpful tone

Output only the synthesized guidance, no explanations.
"""
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            response = self.model(messages)
            guidance = getattr(response, "content", str(response)).strip()
            return guidance if guidance else None
        except Exception as e:
            print(f"Synthesis error: {e}")
            return None
    
    def _extract_tool_memories(self, experiences: List[ExperienceNode], request: MemoryRequest) -> List[MemoryItem]:
        """Extract and wrap tool memories from experiences"""
        tool_items = []
        for exp in experiences:
            for tool_info in exp.extracted_tools:
                try:
                    tool_code = tool_info.get('code', '')
                    tool_name = tool_info.get('name', 'unknown_tool')
                    
                    if not tool_code:
                        continue
                    
                    # Execute the tool code to get the function
                    exec_globals = {}
                    exec(tool_code, exec_globals)
                    func = exec_globals.get(tool_name)
                    
                    if func and self.tool_wrapper:
                        wrapped_tool = self.tool_wrapper.wrap_function(func, tool_name)
                        tool_items.append(MemoryItem(
                            id=f"tool_{tool_name}_{uuid.uuid4().hex[:8]}",
                            content=tool_code,
                            metadata={
                                "wrapped_tool": wrapped_tool,
                                "callable": func,
                                "tool_name": tool_name,
                                "source_experience": exp.experience_id[:8]
                            },
                            score=exp.utility_score,
                            type=MemoryItemType.API
                        ))
                except Exception as e:
                    print(f"Tool extraction error: {e}")
                    continue
        
        return tool_items[:3]  # Limit to 3 tools
    
    def _fallback_response(self, request: MemoryRequest) -> MemoryResponse:
        """Create a safe fallback response"""
        fallback_memories = self._create_fallback_memories(request)
        return MemoryResponse(
            memories=fallback_memories,
            memory_type=self.memory_type,
            total_count=len(fallback_memories),
            request_id=str(uuid.uuid4())
        )
    
    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        """Ingest trajectory data as experience"""
        if not self.engine:
            return False, "Engine not initialized"
        
        try:
            # Determine task success and extract insights
            task_metadata = trajectory_data.metadata or {}
            is_success = task_metadata.get('is_correct', False) or task_metadata.get('success', False)
            partial_score = task_metadata.get('partial_score', 0.0)
            confidence = task_metadata.get('confidence', 0.0)
            
            # Fuzzy success detection
            is_partial_success = False
            if not is_success and partial_score >= self.partial_success_threshold:
                is_partial_success = True
                is_success = False
            
            if not is_success and not is_partial_success:
                # Still extract knowledge from failure
                return self._ingest_failure(trajectory_data)
            
            # Extract multi-level abstractions
            abstractions = self._extract_abstractions(
                trajectory_data, 
                is_success, 
                is_partial_success
            )
            
            if not abstractions:
                return False, "Failed to extract abstractions"
            
            # Extract tools if successful
            extracted_tools = []
            if is_success and self.enable_tool_extraction and self.model:
                extracted_tools = self._extract_tools_from_trajectory(trajectory_data)
            
            # Create experience node
            node = ExperienceNode(
                query=trajectory_data.query,
                concrete_plan=abstractions.get('concrete_plan', ''),
                abstract_plan=abstractions.get('abstract_plan', ''),
                concrete_experience=abstractions.get('concrete_experience', ''),
                abstract_experience=abstractions.get('abstract_experience', ''),
                search_strategy=abstractions.get('search_strategy', ''),
                task_success=is_success,
                partial_success=is_partial_success,
                tags=abstractions.get('tags', []),
                extracted_tools=extracted_tools
            )
            
            # Add to engine
            added = self.engine.add_experience(node)
            if added:
                return True, f"Experience stored: {node.experience_id[:8]}"
            else:
                return False, "Duplicate experience, not stored"
            
        except Exception as e:
            print(f"Take-in memory error: {e}")
            return False, str(e)
    
    def _extract_abstractions(self, trajectory_data: TrajectoryData, is_success: bool, is_partial: bool) -> Optional[Dict]:
        """Extract multi-level abstractions from trajectory using LLM"""
        if not self.model:
            # Simple extraction without model
            trajectory_str = self._format_trajectory(trajectory_data)
            return {
                'concrete_plan': trajectory_str[:500],
                'abstract_plan': "Follow the step-by-step approach.",
                'concrete_experience': trajectory_str[:500],
                'abstract_experience': "Use systematic approach with verification.",
                'tags': ['general'],
                'search_strategy': "Use search to gather information."
            }
        
        trajectory_text = self._format_trajectory(trajectory_data)
        
        extraction_prompt = f"""
You are analyzing an AI agent's task execution to extract structured knowledge. 

Task Query: {trajectory_data.query}
Execution Status: {'Successful' if is_success else 'Partial Success'} 
Result: {trajectory_data.result if trajectory_data.result else 'Completed'}

Execution Trajectory:
{trajectory_text[:2000]}

Extract knowledge at two abstraction levels:

1. **Concrete Plan** (task-specific steps): The exact steps, tool calls, and decisions made
2. **Abstract Plan** (generalizable strategy): The underlying strategy that could apply to similar tasks  
3. **Concrete Experience** (specific lessons): What specifically worked or didn't work
4. **Abstract Experience** (general principles): Broader lessons and best practices
5. **Search Strategy**: Any patterns in how searches were conducted
6. **Tags**: 2-4 relevant keywords/tags

Output ONLY a JSON object:
{{
    "concrete_plan": "detailed step-by-step...",
    "abstract_plan": "general approach...",
    "concrete_experience": "specific lessons...",
    "abstract_experience": "general principles...",
    "search_strategy": "search patterns...",
    "tags": ["tag1", "tag2"]
}}

Each field should have substantial content (2-5 sentences)."""
        
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": extraction_prompt}]}]
            response = self.model(messages)
            result = getattr(response, "content", str(response)).strip()
            
            json_match = re.search(r'\{.*?\}', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return json.loads(result)
        except Exception as e:
            print(f"Abstraction extraction error: {e}")
            return None
    
    def _extract_tools_from_trajectory(self, trajectory_data: TrajectoryData) -> List[Dict[str, str]]:
        """Extract reusable tool functions from successful trajectory"""
        if not self.model:
            return []
        
        trajectory_text = self._format_trajectory(trajectory_data)
        
        extraction_prompt = f"""
Analyze this successful task execution and extract reusable Python functions that capture common patterns.

Task: {trajectory_data.query}

Trajectory:
{trajectory_text[:1500]}

Identify 1-2 useful tool functions that:
1. Parameterize a common operation (search, crawl, extract)
2. Could be reused in similar tasks
3. Are self-contained with clear inputs/outputs

For each function, output:
{{
    "tools": [
        {{
            "name": "function_name",
            "code": "def function_name(param1, param2):\\n    # docstring\\n    # implementation",
            "description": "Brief description of what this tool does"
        }}
    ]
}}

Focus on practical, reusable operations. Output ONLY the JSON."""
        
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": extraction_prompt}]}]
            response = self.model(messages)
            result = getattr(response, "content", str(response)).strip()
            
            json_match = re.search(r'\{.*?\}', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data.get('tools', [])
            return []
        except Exception as e:
            print(f"Tool extraction error: {e}")
            return []
    
    def _ingest_failure(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        """Ingest knowledge from failed executions"""
        if not self.model:
            return False, "Not processing failed task"
        
        trajectory_text = self._format_trajectory(trajectory_data)
        
        failure_prompt = f"""
Analyze this failed task execution to extract valuable negative knowledge.

Task: {trajectory_data.query}
Context: {str(trajectory_data.metadata or {})[:200]}
Result: {str(trajectory_data.result)[:200]}

Trajectory:
{trajectory_text[:1500]}

Extract:
1. **Primary Failure Reason**: What went wrong (1-2 sentences)
2. **Avoidance Strategy**: How to avoid this in future (1-2 sentences)
3. **Tags**: 2-3 keywords

Output ONLY JSON:
{{
    "failure_reason": "what went wrong...",
    "avoidance_strategy": "how to avoid...",
    "tags": ["tag1", "tag2"]
}}
"""
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": failure_prompt}]}]
            response = self.model(messages)
            result = getattr(response, "content", str(response)).strip()
            
            json_match = re.search(r'\{.*?\}', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                data = json.loads(result)
            
            # Create a low-utility experience node with failure knowledge
            node = ExperienceNode(
                query=trajectory_data.query,
                concrete_plan="",
                abstract_plan=f"Avoid: {data.get('avoidance_strategy', 'Unknown failure')}",
                concrete_experience=f"Failed because: {data.get('failure_reason', 'No reason extracted')}",
                abstract_experience=data.get('avoidance_strategy', ''),
                tags=data.get('tags', []),
                task_success=False,
                partial_success=False,
                failure_reason=data.get('failure_reason', ''),
                utility_score=0.1  # Start low
            )
            
            self.engine.add_experience(node)
            return True, f"Failure knowledge stored: {node.experience_id[:8]}"
            
        except Exception as e:
            print(f"Failure ingestion error: {e}")
            return False, str(e)
    
    def _format_trajectory(self, trajectory_data: TrajectoryData) -> str:
        """Format trajectory into readable text"""
        if not trajectory_data.trajectory:
            return "No trajectory data"
        
        parts = [f"Task: {trajectory_data.query}"]
        for i, step in enumerate(trajectory_data.trajectory[:15], 1):  # Limit to 15 steps
            step_type = step.get('type', 'step')
            content = step.get('content', '')
            parts.append(f"Step {i} ({step_type}): {str(content)[:200]}")
        
        if trajectory_data.result:
            parts.append(f"Result: {str(trajectory_data.result)[:200]}")
        
        return "\n".join(parts)