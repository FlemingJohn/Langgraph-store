import logging
from typing import Dict, Any, List

# Core LangGraph postgres store dependency
from langgraph.store.postgres import AsyncPostgresStore
from src.domain.interfaces import IStoreManager

logger = logging.getLogger(__name__)

class PostgresStoreManager(IStoreManager):
    """
    Integration with LangGraph's PostgresStore.
    Follows SRP (Single Responsibility Principle) by entirely abstracting persistence logic.
    """
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._store: AsyncPostgresStore = None
        self._pool_context = None

    async def connect(self):
        """Initializes the connection pool lazily."""
        if self._store is None:
            logger.info("Initializing LangGraph AsyncPostgresStore...")
            self._pool_context = AsyncPostgresStore.from_conn_string(self.connection_string)
            self._store = await self._pool_context.__aenter__()

    async def save_block(self, namespace: str, doc_id: str, block_id: str, value: Dict[str, Any]):
        """Persists exact text blocks. No vectors, no embeddings."""
        if self._store is None:
            await self.connect()
        # In LangGraph Store, namespaces are tuples.
        ns = (namespace, doc_id)
        await self._store.aput(namespace=ns, key=block_id, value=value)

    async def get_blocks(self, namespace: str, doc_id: str, block_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieves precisely required blocks in constant time.
        Determinism inherently solves the shifting-prefix cache miss issue.
        """
        if self._store is None:
            await self.connect()
            
        ns = (namespace, doc_id)
        results = []
        for bid in block_ids:
            item = await self._store.aget(namespace=ns, key=bid)
            if item and item.value:
                results.append(item.value)
                
        # Ensuring fixed order output which heavily favors Prefix Caching
        return sorted(results, key=lambda x: x.get("order", 0))

    async def close(self):
        """Gracefully release the pool."""
        if self._pool_context:
            await self._pool_context.__aexit__(None, None, None)
            self._store = None
