# agent.py
from langgraph.graph import StateGraph
from my_agent.utils.nodes import (
    start_node,
    fetch_evidences,
    perform_image_analysis,
    llm_decision_node,
    tally_totals_node,
    end_node
)

workflow = StateGraph(dict)

workflow.add_node("start_node", start_node)
workflow.add_node("fetch_evidences", fetch_evidences)
workflow.add_node("perform_image_analysis", perform_image_analysis)
workflow.add_node("llm_decision_node", llm_decision_node)
workflow.add_node("tally_totals_node", tally_totals_node)
workflow.add_node("end_node", end_node)

workflow.set_entry_point("start_node")
workflow.add_edge("start_node", "fetch_evidences")
workflow.add_edge("fetch_evidences", "perform_image_analysis")
workflow.add_edge("perform_image_analysis", "llm_decision_node")
workflow.add_conditional_edges(
    "llm_decision_node",
    lambda state: "tally_totals_node" if state["llm_decision"] else "end_node"
)
workflow.add_edge("tally_totals_node", "end_node")

executor = workflow.compile()
