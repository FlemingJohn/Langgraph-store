from abc import ABC, abstractmethod
from typing import Dict, Any, List

class ILLMProvider(ABC):
    """
    Interface Segregation Principle (ISP) & Dependency Inversion Principle (DIP).
    Isolates external LLM packages (LangChain, OpenAI, Anthropic) from core logic.
    """
    @abstractmethod
    async def invoke(self, prompt: str) -> str:
        """Invokes the LLM with a highly optimized, cache-friendly prompt."""
        pass

class IStoreManager(ABC):
    """
    Abstract interface for managing persistence and retrieval of state/knowledge.
    """
    @abstractmethod
    async def connect(self):
        """Initializes database/store connections cleanly."""
        pass

    @abstractmethod
    async def save_block(self, namespace: str, doc_id: str, block_id: str, value: Dict[str, Any]):
        """Persists a canonical data block."""
        pass

    @abstractmethod
    async def get_blocks(self, namespace: str, doc_id: str, block_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieves exact blocks deterministically without vector search."""
        pass

class IBlockRouter(ABC):
    """
    Abstract interface for the dynamic router.
    """
    @abstractmethod
    async def route_intent_to_blocks(self, intent: str, available_blocks_metadata: Dict[str, str]) -> List[str]:
        """Maps an intent to specific block IDs using ILLMProvider."""
        pass
