import json
import re
from typing import Dict, List
from src.domain.interfaces import IBlockRouter, ILLMProvider

class SemanticBlockRouter(IBlockRouter):
    """
    Open/Closed Principle (OCP): Router accepts any ILLMProvider.
    Analyzes intent and selectively picks exact document blocks.
    Replaces brittle vector retrieval with dynamic semantic selection.
    """
    def __init__(self, llm: ILLMProvider):
        self.llm = llm

    async def route_intent_to_blocks(self, intent: str, available_blocks_metadata: Dict[str, str]) -> List[str]:
        """
        Takes summarized metadata of all blocks and the user intent.
        Returns a JSON array of block IDs.
        """
        metadata_str = "\n".join(f'"{bid}": "{desc}"' for bid, desc in available_blocks_metadata.items())
        
        prompt = (
            "You are a routing dispatcher. Select the minimum required document blocks "
            "to answer the following intent. Rules:\n"
            "- Optimize for minimal token usage.\n"
            "- Output ONLY a JSON array of block IDs. No other text.\n\n"
            f"Available Blocks:\n{{\n{metadata_str}\n}}\n\n"
            f"Intent: {intent}"
        )

        response = await self.llm.invoke(prompt)
        
        # Robust extraction
        json_match = re.search(r"\[.*?\]", response, re.DOTALL)
        if json_match:
            try:
                selected_ids = json.loads(json_match.group())
                return [bid for bid in selected_ids if bid in available_blocks_metadata]
            except json.JSONDecodeError:
                pass
                
        return []
