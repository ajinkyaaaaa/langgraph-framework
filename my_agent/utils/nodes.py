# nodes.py
from typing import Dict, Any, TypedDict
from my_agent.utils.tools import *
from my_agent.utils.state import AgentState

class GetControlNumber(TypedDict):
    control_number: Any

# get control number from user 
def user_input(state: GetControlNumber) -> AgentState:
    """ Start node initializes the workflow. """
    return AgentState(control_number=state["control_number"], scope={}, files=None, 
                      llm_decision=None, ipe_checkpoints={}, tally_checkpoints={})

# establish scope for complete process
def establish_scope(state: AgentState) -> AgentState:
    file_path = r"/content/Landing/CTRL0037345.xlsx"
    state["scope"] = gather_scope(file_path=file_path, control_number=state["control_number"])
    return state

# ---> remove
def fetch_evidences(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Fetches the required evidences. """
    state["evidences"] = extract_evidence.run(state["query"])
    state["files"] = fetch_evidence.run(state["evidences"])
    return state

# IPE Agent
def ipe_validator(state: AgentState) -> AgentState:
    """ Performs image analysis on evidences. """
    queries = state["scope"]["IPE"]
    print(state["scope"])
    results = []
    for query in queries:
        evidences = extract_evidence.run(query)
        state["files"] = fetch_evidence.run(evidences)
        for file in state["files"]["image_path"]:
            results.append(image_analysis.run({"query": query, "image_path": file}))
    state["ipe_checkpoints"] = results
    return state

# Analysing agent
def supervisor(state: AgentState) -> AgentState:
    """Uses LLM to make a decision based on the evidence."""
    all_valid = all(entry.get("valid") == "yes" for entry in state.get("ipe_checkpoints", []))
    state["llm_decision"] = all_valid  
    return state

# Tally excel and image totals 
def tally_totals_node(state: AgentState) -> AgentState:
    """ Reads an Excel file and sums relevant values. """
    query = state["scope"]["Tally"]
    print(query)
    evidences = extract_evidence.run(query)
    state["files"] = fetch_evidence.run(evidences)
    excel_path = r"/content/Landing/37345_IPE3.xlsx"
    excel_total = excel_reader.run(excel_path)
    result = []
    for file in state["files"]["image_path"].values():
        tally_result = image_analysis.run({"query": query, "image_path": file})
        result.append(tally_result)
    state["tally_checkpoints"] = result
    return state

def evaluator(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Ends the workflow. """
    state["message"] = "Process completed."
    return state

def should_continue(state):
    messages = state["llm_decision"]
    # If there are no tool calls, then we finish
    if not messages:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"