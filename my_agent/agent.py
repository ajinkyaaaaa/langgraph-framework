# agent.py
from langgraph.graph import StateGraph, START, END
from my_agent.utils.nodes import *
# from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
# perform_image_analysis = create_react_agent()
# memory = MemorySaver()
from langgraph.types import Command

workflow = StateGraph(dict)


############################
# # workflow.add_node("User Input", start_node)
# # workflow.add_node("Initialise", fetch_scope)
# workflow.add_node("Distributor", fetch_evidences)
# workflow.add_node("[Agent] IPE", perform_image_analysis)
# workflow.add_node("[Agent] Analyser", llm_decision_node)
# workflow.add_node("[Tool] Tally Totals", tally_totals_node)
# workflow.add_node("Evaluator", end_node)


# # workflow.add_edge(START, "Initialise")
# # workflow.add_edge("Initialise", "Distributor")
# # start from distibutor
# workflow.add_edge(START, "Distributor")
# # rest of the workflow
# workflow.add_edge("Distributor", "[Agent] IPE")
# workflow.add_edge("[Agent] IPE", "[Agent] Analyser")
# workflow.add_conditional_edges(
#     "[Agent] Analyser",
#     should_continue,
#     {
#         "continue":"[Tool] Tally Totals",
#         "end":"Evaluator"
#     }
#     # lambda state: "[Tool] Tally Totals" if state["llm_decision"] else END
# )
# workflow.add_edge("[Tool] Tally Totals", "Evaluator")
# workflow.add_edge("Evaluator", END)

# executor = workflow.compile(checkpointer=memory)
###########################

workflow.add_node(user_input)

workflow.add_node("Run Agent Workflow", user_input)
workflow.add_edge("Establish Scope", gather_scope)
workflow.add_node("Distributor", process_evidence)
workflow.add_node("[Agent] IPE", perform_image_analysis)
workflow.add_node("[Agent] Analyser", llm_decision_node)
workflow.add_node("[Tool] Tally Totals", tally_totals_node)
workflow.add_node("Evaluator", end_node)


# workflow.add_edge(START, "Initialise")
# workflow.add_edge("Initialise", "Distributor")
# start from distibutor
workflow.add_edge(START, "Run Agent Workflow")
# State to update: state["control_number"]
workflow.add_edge("Run Agent Workflow", "Establish Scope")
# takes state["control_number"] -> output -> state["scope"]
workflow.add_edge("Establish Scope", "[Agent] IPE")
# takes state["scope"] -> fetch_evidences() -> append state["ipe_checkpoints"]
workflow.add_edge("[Agent] IPE", "Distributor")
# takes state["files_required"] 
workflow.add_edge("Distributor", "[Agent] IPE")
# takes state["files_required"] -> output -> state["file_paths"]
workflow.add_edge("[Agent] IPE", "[Agent] Analyser")
# takes state["ipe_checkpoints"] -> performs rest of the analysis 

workflow.add_conditional_edges(
    "[Agent] Analyser",
    should_continue,
    {
        "continue":"[Tool] Tally Totals",
        "end":"Evaluator"
    }
)
workflow.add_edge("[Tool] Tally Totals", "Distributor")
workflow.add_edge("Distibutor", "[Tool] Tally Totals")
workflow.add_edge("Evaluator", END)
executor = workflow.compile()
