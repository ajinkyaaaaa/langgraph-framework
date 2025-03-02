from typing import TypedDict, List, Dict, Any, Annotated
import time

class get_control_number(TypedDict):
    control_number_to_validate: str
class OutputState(TypedDict):
    answer: str

class AgentState(TypedDict, get_control_number, OutputState):
    # Step 1: get the control number and establish scope
    control_number: str
    scope_details: Any
    files = List[str]
    # checkpoints
    ipe_checkpoints: Dict[str, Dict[str, Any]]
    tally_checkpoints: Dict[str, Dict[str, Any]]

    message: Annotated[List[str], "Metadata for nodes"]