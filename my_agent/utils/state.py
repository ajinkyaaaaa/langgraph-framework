from typing import List, Dict, Annotated, Any
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: Annotated[str, "Metadata for nodes."]
    control_number: str
    scope: Dict
    files: Any
    llm_decision: Any
    ipe_checkpoints: Dict[str, Dict]
    tally_checkpoints: Dict[str, Dict]