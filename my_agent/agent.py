# agent.py
from langgraph.graph import StateGraph, START, END
from my_agent.utils.nodes import (
    start_node,
    fetch_evidences,
    perform_image_analysis,
    llm_decision_node,
    tally_totals_node,
    end_node,
    # should_continue
)
# from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
# perform_image_analysis = create_react_agent()
# memory = MemorySaver()

workflow = StateGraph(dict)

# workflow.add_node("User Input", start_node)
workflow.add_node("Distributor", fetch_evidences)
workflow.add_node("[Agent] IPE", perform_image_analysis)
workflow.add_node("[Agent] Analyser", llm_decision_node)
workflow.add_node("[Tool] Tally Totals", tally_totals_node)
workflow.add_node("Evaluator", end_node)

# workflow.set_entry_point(START)
# workflow.add_edge(START, "User Input")
workflow.add_edge(START, "Distributor")
workflow.add_edge("Distributor", "[Agent] IPE")
workflow.add_edge("[Agent] IPE", "[Agent] Analyser")
workflow.add_conditional_edges(
    "[Agent] Analyser",
    # should_continue,
    # {
    #     "continue":"[Tool] Tally Totals",
    #     "end":END
    # }
    lambda state: "[Tool] Tally Totals" if state["llm_decision"] else END
)
workflow.add_edge("[Tool] Tally Totals", "Evaluator")
workflow.add_edge("Evaluator", END)

# executor = workflow.compile(checkpointer=memory)
executor = workflow.compile()
