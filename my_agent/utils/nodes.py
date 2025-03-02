# nodes.py
from typing import Dict, Any
from typing_extensions import TypedDict
from utils.tools import extract_evidence, fetch_evidence, image_analysis, excel_reader, gather_scope

class get_control_number(TypedDict):
    control_number: str

class summary(TypedDict):
    summary: str

def fetch_scope(state: get_control_number) -> Dict[str, Any]:
    """ Gets the scope for the workflow. """
    base_file = r"C:\Users\Ajinkya\Desktop\Projects\lang-graph-poc\my_app\my_agent\Landing\CTRL0037345.xlsx"
    state["scope"] = gather_scope(file_path=base_file, control_number=state["control_number"])
    return state

def fetch_evidences(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Fetches the required evidences. """
    state["evidences"] = extract_evidence.run(state["query"])
    state["files"] = fetch_evidence.run(state["evidences"])
    return state

def perform_image_analysis(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Performs image analysis on evidences. """
    results = []
    for file in state["files"]["files"].values():
        results.append(image_analysis.run({"query": state["query"], "image_path": file}))
    state["image_results"] = results
    return state

def llm_decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Uses LLM to make a decision based on the evidence. """
    decision = "approve" if "Processed" in str(state["image_results"]) else "reject"
    state["llm_decision"] = decision == "approve"
    return state

def tally_totals_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Reads an Excel file and sums relevant values. """
    state["totals"] = excel_reader.run("/data/financials.xlsx")
    return state

def end_node(state: Dict[str, Any]) -> Dict[str, Any]:
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