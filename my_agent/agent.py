# agent.py
from langgraph.graph import StateGraph
from langgraph.constants import START, END

# from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
# perform_image_analysis = create_react_agent()
# memory = MemorySaver()
from utils.state import AgentState
from my_agent.utils.nodes import *

workflow = StateGraph(AgentState, input=get_control_number)
# Define edges
workflow.add_node("User Input", user_input)
workflow.add_node("Gather Scope", establish_scope)
workflow.add_node("[agent] IPE Validator", ipe_validator)
workflow.add_node("[agent] Supervisor", supervisor)
workflow.add_node("[tool] Tally Totals", tally_totals_node)
workflow.add_node("Evaluator", evaluator)
# Define flow
workflow.add_edge(START, "User Input")
workflow.add_edge("User Input", "Gather Scope")
workflow.add_edge("Gather Scope", "[agent] IPE Validator")
workflow.add_edge("[agent] IPE Validator", "[agent] Supervisor")
workflow.add_conditional_edges(
    "[agent] Supervisor",
    should_continue,
    {
        "continue":"[tool] Tally Totals",
        "end":"Evaluator"
    }
)
workflow.add_edge("[tool] Tally Totals", "Evaluator")
workflow.add_edge("Evaluator", END)
# executor = workflow.compile(checkpointer=memory)
executor = workflow.compile()
