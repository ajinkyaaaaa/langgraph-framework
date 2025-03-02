# nodes.py
from typing import Dict, Any
from my_agent.utils.tools import *
from typing_extensions import TypedDict, Literal
from utils.state import *
from langgraph.types import Command

def user_input(state: get_control_number):
    state["control_number"] = state["control_number_to_validate"]

def fetch_scope(state: get_control_number) -> AgentState:
    """ Gets the scope for the workflow. """
    base_file = r"C:\Users\Ajinkya\Desktop\Projects\lang-graph-poc\my_app\my_agent\Landing\CTRL0037345.xlsx"
    state["scope"] = gather_scope(file_path=base_file, control_number=state["control_number"])
    return state

def fetch_evidences(state: AgentState) -> AgentState:
    """ Fetches the required evidences. """
    query = state["ipe_info"]
    state["evidences"] = extract_evidence.run(query)
    state["files"] = fetch_evidence.run(state["evidences"])
    return state

def perform_image_analysis(state: AgentState) -> Command[Literal["Distributor"]]:
    """ Performs image analysis on evidences. """
    ipe_queries = state["scope"]["IPE"]
    no_of_ipes = len(ipe_queries)
    for i in range(ipe_queries):
        if len(state["ipe_checkpoints"]) < no_of_ipes:
            return Command(update={"ipe_info": ipe_queries[i]}, goto="Distributor")
        else:
            






    results = []
    for query in ipe_queries:
        files = process_evidence.run(query)
        for file in files["file_paths"].values():
            results.append(image_analysis.run({"query": query, "image_path": file}))
    state["ipe_checkpoints"] = results
    return state

def llm_decision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """ Uses LLM to make a decision based on the evidence. """
    decision = "approve" if "Processed" in str(state["ipe_checkpoints"]) else "reject"
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