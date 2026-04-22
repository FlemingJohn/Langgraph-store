import asyncio
import json
from src.domain.interfaces import ILLMProvider, IStoreManager, IBlockRouter
from src.core.routing import SemanticBlockRouter
from src.graph.orchestrator import GraphOrchestrator

# Mocks to test the architecture without an actual LLM/Postgres footprint
class MockLLM(ILLMProvider):
    async def invoke(self, prompt: str) -> str:
        # Simulate returning selected blocks
        return '["block_01", "block_03"]'

class MockStoreManager(IStoreManager):
    async def connect(self): pass
    async def save_block(self, namespace, doc_id, block_id, value): pass
    async def get_blocks(self, namespace, doc_id, block_ids):
        # Mocks a fast O(1) DB fetch
        repo = {
            "block_01": {"block_id": "block_01", "content": "Engineering Data 1", "order": 1},
            "block_02": {"block_id": "block_02", "content": "Engineering Data 2", "order": 2},
            "block_03": {"block_id": "block_03", "content": "Engineering Data 3", "order": 3},
        }
        fetched = [repo[b] for b in block_ids if b in repo]
        return sorted(fetched, key=lambda x: x["order"])

async def main():
    llm = MockLLM()
    router = SemanticBlockRouter(llm)
    store = MockStoreManager()

    orchestrator = GraphOrchestrator(store, router)
    graph = orchestrator.build_graph()

    print("Executing Architecture Test...")
    
    state = {
        "document_id": "doc_100",
        "intent": "Find engineering specifications for power.",
        "available_metadata": {
            "block_01": "General Definitions",
            "block_02": "Geometry Constraints",
            "block_03": "Power Constraints"
        }
    }

    result = await graph.ainvoke(state)
    
    print("\n--- SELECTED BLOCKS ---")
    print(result.get("selected_blocks"))
    
    print("\n--- FINAL PROMPT FOR CACHING ---")
    print(result.get("final_prompt_payload"))

if __name__ == "__main__":
    asyncio.run(main())
