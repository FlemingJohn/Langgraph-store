from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    """
    Strongly typed state dictionary for LangGraph workflow.
    """
    document_id: str
    intent: str
    
    # Internal states payload tracking
    available_metadata: Dict[str, str]
    selected_blocks: List[str]
    retrieved_context: List[Dict[str, Any]]
    
    # Output
    final_prompt_payload: str
