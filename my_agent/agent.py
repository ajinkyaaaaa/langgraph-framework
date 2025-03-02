# agent.py
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from utils.nodes import (
    get_control_number,
    fetch_scope,
    fetch_evidences,
    perform_image_analysis,
    llm_decision_node,
    tally_totals_node,
    end_node,
    should_continue
)
# from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
# perform_image_analysis = create_react_agent()
# memory = MemorySaver()

workflow = StateGraph(dict, input=get_control_number)

# workflow.add_node("User Input", start_node)
workflow.add_node("Initialise", fetch_scope)
workflow.add_node("Distributor", fetch_evidences)
workflow.add_node("[Agent] IPE", perform_image_analysis)
workflow.add_node("[Agent] Analyser", llm_decision_node)
workflow.add_node("[Tool] Tally Totals", tally_totals_node)
workflow.add_node("Evaluator", end_node)

# workflow.set_entry_point(START)
# workflow.add_edge(START, "User Input")
workflow.add_edge(START, "Initialise")
workflow.add_edge("Initialise", "Distributor")
workflow.add_edge("Distributor", "[Agent] IPE")
workflow.add_edge("[Agent] IPE", "[Agent] Analyser")
workflow.add_conditional_edges(
    "[Agent] Analyser",
    should_continue,
    {
        "continue":"[Tool] Tally Totals",
        "end":"Evaluator"
    }
    # lambda state: "[Tool] Tally Totals" if state["llm_decision"] else END
)
workflow.add_edge("[Tool] Tally Totals", "Evaluator")
workflow.add_edge("Evaluator", END)

# executor = workflow.compile(checkpointer=memory)
executor = workflow.compile()


# Invoke the graph with an input and print the result
print(executor.invoke({"control_number": "CTRL0037345"}))