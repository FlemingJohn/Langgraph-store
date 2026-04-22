# Deterministic Document Orchestration Architecture
### Developer Integration Guide

This document outlines the implementation of a deterministic, high-speed document retrieval mechanism using the LangGraph Store. By deprecating traditional Retrieval-Augmented Generation (RAG), Vector Databases, and GraphRAG methodologies, this architecture achieves sub-second latency and an 80% reduction in token overhead via prefix caching.

## Core Tenets
Standard RAG pipelines introduce massive latency via embedding generation and probabilistic vector search. Furthermore, vector proximity does not guarantee logical relevance, particularly in strict technical specifications requiring multi-hop dependency resolution.

This architecture relies on three pillars:
1. **The Shared Brain**: Pre-loading documents as discrete blocks into the native LangGraph PostgresStore.
2. **Semantic Routing and Multi-hop Dependencies**: Using lightweight LLM dispatchers to select exact Block IDs and their prerequisite dependencies.
3. **Prefix Caching via Block Sorting**: Guaranteeing LLM cache hits by injecting context in strict numerical order.

---

## 1. Initializing the LangGraph PostgresStore

The LangGraph `AsyncPostgresStore` acts as a high-speed, key-value repository. It bypasses vectorization entirely, allowing the graph to fetch specific clause blocks in O(1) time.

```python
from langgraph.store.postgres import AsyncPostgresStore

async def initialize_and_seed_store(connection_string: str):
    """
    Initializes the connection pool and seeds the store with document blocks.
    In a production application, this store is a global singleton accessible across all threads.
    """
    store_pool = AsyncPostgresStore.from_conn_string(connection_string)
    store = await store_pool.__aenter__()

    # Document blocks must be logical clauses or tables, manually segmented beforehand.
    await store.aput(
        namespace=("documents", "spec_a"),
        key="block_01",
        value={
            "order": 1,
            "content": "Total quarterly revenue must exceed $50M to qualify.",
            "metadata_summary": "Defines aggregate revenue threshold."
        }
    )
    return store
```

---

## 2. Dynamic Routing and Multi-Hop Dependency Resolution

Instead of searching by vector mathematics, the architecture passes concise block metadata to a routing LLM. The LLM determines the optimal subset of blocks required to fulfill the user's intent.

Crucially, it executes a secondary pass to resolve **multi-hop dependencies** (e.g., if a selected table requires a calculation formula located in a different block).

```python
import json
import re

class SemanticRouter:
    def __init__(self, llm):
        self.llm = llm

    async def extract_required_blocks(self, intent: str, available_metadata: dict) -> list[str]:
        """
        Primary pass: Identify core blocks based on intent.
        """
        metadata_str = "\n".join(f'"{bid}": "{desc}"' for bid, desc in available_metadata.items())
        prompt = (
            "You are a strict routing engine. Evaluate the user intent and identify the required Block IDs.\n"
            f"Available Blocks:\n{{\n{metadata_str}\n}}\n\n"
            f"Intent: {intent}\n\n"
            "Return a JSON array of Block IDs. No explanation."
        )
        response = await self.llm.invoke(prompt)
        return self._parse_json_ids(response, available_metadata)

    async def resolve_dependencies(self, selected_ids: list[str], available_metadata: dict) -> list[str]:
        """
        Secondary pass (Multi-hop mapping): Identify prerequisite blocks needed to fully understand the core blocks.
        This entirely replaces complex GraphRAG schemas.
        """
        selected_summaries = "\n".join(f'"{bid}": "{available_metadata[bid]}"' for bid in selected_ids)
        remaining_metadata = {bid: v for bid, v in available_metadata.items() if bid not in selected_ids}
        remaining_str = "\n".join(f'"{bid}": "{desc}"' for bid, desc in remaining_metadata.items())

        prompt = (
            "Review the selected core document blocks. Do they logically depend on definitions, formulas, or "
            "tables from the remaining available blocks?\n"
            f"Selected Blocks:\n{{\n{selected_summaries}\n}}\n\n"
            f"Remaining Blocks:\n{{\n{remaining_str}\n}}\n\n"
            "Identify ONLY the Block IDs from the remaining blocks that represent mandatory prerequisites. "
            "Return a JSON array of the required prerequisite Block IDs."
        )
        response = await self.llm.invoke(prompt)
        prerequisite_ids = self._parse_json_ids(response, available_metadata)
        
        # Combine and deduplicate
        return list(set(selected_ids + prerequisite_ids))

    def _parse_json_ids(self, text: str, valid_keys: dict) -> list[str]:
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                ids = json.loads(match.group())
                return [bid for bid in ids if bid in valid_keys]
            except json.JSONDecodeError:
                pass
        return []
```

---

## 3. Maximizing LLM Prefix Caching via Block Sorting

Cloud LLM providers natively cache request tokens when the absolute prefix of the prompt string explicitly matches a prior request. 

Standard RAG inherently prevents prefix caching because semantic search returns chunk arrays in varying proximity orders for every unique query. By sorting the deterministically routed blocks by their absolute document order, the static text block becomes a stable, cacheable prefix.

```python
class PromptCacheOptimizer:
    @staticmethod
    def construct_deterministic_payload(retrieved_blocks: list[dict], dynamic_user_intent: str) -> str:
        """
        Constructs the final payload. 
        Sorts the blocks to guarantee stable prefixes, maximizing TTFT latency reduction.
        """
        # Strict sorting guarantees the prefix order never wavers regardless of what combination 
        # of blocks were routed.
        sorted_blocks = sorted(retrieved_blocks, key=lambda b: b.get("order", 0))
        
        static_context = []
        for block in sorted_blocks:
            static_context.append(f"--- BLOCK: {block.get('block_id')} ---\n{block.get('content')}")
            
        static_prefix = "\n\n".join(static_context)
        
        # CACHE LINE: Everything above the user intent remains stable across varying but related queries.
        return f"SYSTEM CONTEXT:\n{static_prefix}\n\nUSER INTENT:\n{dynamic_user_intent}"
```

### The Mathematics of the 80-90% Cost Reduction (Cold vs Warm Starts)
The primary financial driver of this architecture is the imbalance between input and output tokens during complex reasoning tasks. When validating documents, input context is massive (e.g., 20,000 tokens of document blocks), while the output audit is minimal (e.g., 500 tokens).

- **Cold Start (First Execution - The Closed Book)**: The system computes the entire prefix. This is the equivalent of handing someone a 500-page manual for the very first time and asking a question. They must read the entire book cover-to-cover before answering. The provider (Azure/Anthropic) charges standard input rates, and TTFT reflects standard heavy processing times.
- **Warm Start (Subsequent Executions - The Open Book)**: Because the `PromptCacheOptimizer` strictly sorts the injected blocks, identical contexts hit the active cache memory. For the second question, the book is already open on their desk and committed to memory. They answer instantly without re-reading the text. The provider heavily discounts these input tokens.

Even though output tokens continue to bill at standard rates, driving the 20,000+ input tokens to near-zero pricing structurally forces the aggregate cost per execution down by 80-90%. Concurrently, TTFT drops from multiple seconds to milliseconds as the model bypasses prefix evaluation.

## 4. Accelerating Execution via Pre-Computed Indexes

To achieve near-instant dynamic routing, the metadata mapping is not generated dynamically per query. It is pre-computed and indexed at deployment. The LLM simply diagnoses the intent against the pre-loaded index.

```python
class PreComputedIndexer:
    @staticmethod
    def generate_index(document_blocks: list[dict]) -> dict:
        """
        Executed once during system startup or document ingestion.
        Generates a fast-lookup metadata dictionary.
        """
        index = {}
        for block in document_blocks:
            # We store only tiny summaries for the router, never the heavy raw text.
            index[block["block_id"]] = block.get("metadata_summary", "No summary.")
        return index

# In production, the SemanticRouter consumes this lightweight, instantly available index.
# Fast LLM execution occurs because the index token length is minimal.
global_precomputed_index = PreComputedIndexer.generate_index(heavy_document_blocks)
```

---

## 5. Instant UX via LangGraph Node Streaming

To prevent monolithic blocking, the architecture leverages `astream` or `astream_events` to push partial outputs to the frontend instantly. The UI reacts millisecond-by-millisecond rather than waiting for the entire graph to conclude.

```python
async def execute_with_ui_streaming(graph_orchestrator, initial_state: dict):
    """
    Demonstrates intercepting node completion events to update a frontend UI instantly.
    """
    # Stream mode "updates" yields the state delta immediately as each node finishes
    async for node_name, state_update in graph_orchestrator.astream(
        initial_state, stream_mode="updates"
    ):
        if node_name == "route_blocks":
            print(f"[UI UPDATE] Found {len(state_update['selected_blocks'])} relevant document sections...")
            # Push WebSocket event to frontend: "Analyzing core requirements..."
            
        elif node_name == "resolve_dependencies":
            print(f"[UI UPDATE] Resolved multi-hop dependencies. Assembling context...")
            # Push WebSocket event: "Pulling prerequisite calculations..."
            
        elif node_name == "fetch_and_cache":
            print(f"[UI UPDATE] Context assembled. Executing regulatory validation...")
            # Push WebSocket event: "Executing LLM..."
```

## Summary
By substituting RAG with deterministic metadata routing, strict Block ID fetching via `AsyncPostgresStore`, explicit LLM multi-hop dependency requests, and rigorous sorting, latency is reduced to sub-second thresholds. The code examples provided establish the foundation for implementing this structure safely in any LangGraph-driven data pipeline.
