from langgraph.graph import StateGraph, START, END
from src.graph.state import AgentState
from src.domain.interfaces import IStoreManager, IBlockRouter
from src.core.prompt_caching import PromptCacheHelper

class GraphOrchestrator:
    """
    Law of Demeter (LoD): Coordinates components without delving into their sub-methods.
    Composition over Inheritance: Takes dependencies rather than inheriting them.
    """
    
    def __init__(self, store_manager: IStoreManager, router: IBlockRouter):
        self.store = store_manager
        self.router = router
        self.builder = StateGraph(AgentState)
        self._setup_nodes()
        self._setup_edges()

    def _setup_nodes(self):
        self.builder.add_node("route_blocks", self._node_route_blocks)
        self.builder.add_node("fetch_and_cache", self._node_fetch_and_cache)

    def _setup_edges(self):
        self.builder.add_edge(START, "route_blocks")
        self.builder.add_edge("route_blocks", "fetch_and_cache")
        self.builder.add_edge("fetch_and_cache", END)

    async def _node_route_blocks(self, state: AgentState) -> dict:
        """Dynamically picks blocks via Language Model routing."""
        intent = state.get("intent", "")
        # Assuming metadata is passed in state for this example
        meta = state.get("available_metadata", {})
        
        selected = await self.router.route_intent_to_blocks(intent, meta)
        # Node Caching point: The workflow avoids re-fetching if 'selected' doesn't change
        # in a checkpoint-enabled graph environment.
        return {"selected_blocks": selected}

    async def _node_fetch_and_cache(self, state: AgentState) -> dict:
        """Fetches directly from PostgresStore and formats for Prompt Caching."""
        doc_id = state.get("document_id")
        selected_ids = state.get("selected_blocks", [])
        intent = state.get("intent", "")
        
        # O(1) Fetch precisely what we need from Shared Brain
        blocks = await self.store.get_blocks("docs", doc_id, selected_ids)
        
        # Heavy Prefix Caching formatting
        payload = PromptCacheHelper.construct_prefix_heavy_payload(blocks, intent)
        
        return {
            "retrieved_context": blocks,
            "final_prompt_payload": payload
        }

    def build_graph(self):
        """Compiles the workflow."""
        return self.builder.compile()
