from typing import List, Dict, Any

class PromptCacheHelper:
    """
    KISS Principle. 
    Strict formatting enforces deterministic LLM prompt prefixes.
    For Azure (OpenAI) and Anthropic Prefix Caching to hit >90%, the exact same bits
    must form the start of the payload.
    """
    
    @staticmethod
    def construct_prefix_heavy_payload(blocks: List[Dict[str, Any]], user_query: str) -> str:
        """
        Orders blocks consistently to maximize Prefix Caching hits.
        The dynamic (user_query) is intentionally placed at the VERY END.
        """
        # Ensure consistent deterministic sorting
        sorted_blocks = sorted(blocks, key=lambda b: b.get("order", 0))
        
        prefix_context_sections = []
        for block in sorted_blocks:
            prefix_context_sections.append(
                f"--- BLOCK: {block.get('block_id')} ---\n{block.get('content')}"
            )
            
        full_static_prefix = "\n\n".join(prefix_context_sections)
        
        # Format payload: Static Context FIRST, Dynamic Context LAST.
        return f"SYSTEM CONTEXT (CACHEABLE):\n{full_static_prefix}\n\nUSER INTENT:\n{user_query}"
