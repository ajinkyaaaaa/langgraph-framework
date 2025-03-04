from typing import List, Dict, Annotated, Any
from typing_extensions import TypedDict
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class AgentState(TypedDict):
    messages: Annotated[str, "Metadata for nodes."]
    control_number: str
    scope: Dict
    files: Any
    llm_decision: Any
    ipe_checkpoints: Dict[str, Dict]
    tally_checkpoints: Dict[str, Dict]
    summary: str

class ImageAnalysisInput(BaseModel):
    query: Optional[str] = None
    image_path: Optional[str] = None
    data: Optional[Any] = None
    context: Optional[str] = None
    for_image: bool = False
    for_text: bool = False
    return_amount:bool = False