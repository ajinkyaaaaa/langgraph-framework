# nodes.py
from typing import Dict, Any, TypedDict


#################################################
# Run on cloud
# from my_agent.utils.tools import *
# from my_agent.utils.state import AgentState
# Run locally
from utils.tools import *
from utils.state import AgentState
#################################################

class GetControlNumber(TypedDict):
    control_number: Any
  
class ShowSummary(TypedDict):
    summary: str

# get control number from user 
def user_input(state: GetControlNumber) -> AgentState:
    """ Start node initializes the workflow. """
    return AgentState(control_number=state["control_number"], scope={}, files=None, 
                      llm_decision=None, ipe_checkpoints={}, tally_checkpoints={})

# establish scope for complete process
def establish_scope(state: AgentState) -> AgentState:
    file_path = r"C:\Users\UV172XK\code@ey\Agentic AI\my_app\langgraph-framework\my_agent\Landing\CTRL0037345.xlsx"
    try:
        state["scope"] = gather_scope(file_path=file_path, control_number=state["control_number"])
    except Exception as e:
        state["scope"] = "Process failed."
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
    guidelines = """
    "When analyzing the image, you must register only the fields and values that have the RED font color-ignore everything else.
    The image follows a horizontal layout, where each field's corresponding value appears in the same row.
    You do not need to check for exact formatting—only compare the values to see if they are similar.
    Do not extract or mention any fields which do not have the font color as RED.
    In the comments, explicitly state the values found and the expected values."
    """
    results = []
    for query in queries:
        evidences = extract_evidence.run(query)
        state["files"] = fetch_evidence.run(evidences)
        for file in state["files"]["image_path"]:
            output = llm_call.run({"query": query, "image_path": file, "context":guidelines, "for_image":True})
            data = f"""
            I have received the following JSON response from my LLM: {str(output)}. However, the response contains inaccuracies because the LLM is failing to correctly compare values.

            Your task:
            - Carefully **verify** if the mentioned values are actually incorrect.
            - If a value is **already correct**, do **not** flag it as incorrect.
            - If **all values are correct**, update `"valid": "yes"` in the response.

            Important:
            - The final response **must strictly follow the same JSON structure** as the one provided.
            - Do **not** add any explanations, additional formatting, or extra text—only return the corrected JSON.
            """
            guidelines = f"""
            You must strictly return a **JSON object** in the following format:

            {{
                "valid": "yes" or "no",
                "comments": "Comment on the observation (mention the image being analyzed)"
            }}

            Rules:
            1. **Verify each value carefully** before marking it as incorrect.
            2. If a value is already **correct**, explicitly state that it is correct.
            3. If **all values are correct**, update `"valid": "yes"`, since this means everything is valid.
            4. **Do not** include any explanations, extra text, or formatting—return only the corrected JSON.
            """
            time.sleep(5)
            temp = llm_call.run({"context":guidelines, "for_text":True, "data":data})
            refined_output = json.loads(temp)
            print("\nrefined_output:\n", refined_output)
            results.append(refined_output)
    state["ipe_checkpoints"] = results
    return state

# Analysing agent
def supervisor(state: AgentState) -> AgentState:
    """Uses LLM to make a decision based on the evidence."""
    print("IPE CHECKPOINTS: ", state.get("ipe_checkpoints"))
    all_valid = all(entry.get("valid") == "yes" for entry in state.get("ipe_checkpoints", []))
    state["llm_decision"] = all_valid  
    print("Supervisor:\n\n", state["llm_decision"])
    return state

# Tally excel and image totals 
def tally_totals_node(state: AgentState) -> AgentState:
    """ Reads an Excel file and sums relevant values. """
    guidelines = """
    When analyzing the image, register only the fields and values highlighted in yellow.
    You will find the required data at the bottom of the image.
    You will only return the integer part of the number that is found in the image as your response.
    """
    query = state["scope"]["Tally"]
    
    evidences = extract_evidence.run(query[0])
    state["files"] = fetch_evidence.run(evidences)
    print(f"Tally evidences: {state['files']}")
    excel_path = r"C:\Users\UV172XK\code@ey\Agentic AI\my_app\langgraph-framework\my_agent\Landing\37345_IPE3.xlsx"
    excel_output = excel_reader.run(excel_path)
    print(f"Excel Total", excel_output)

    result = {
        "valid": None,
        "comments": None
    }

    for file in state["files"]["image_path"]:
        if file.lower().endswith(".jpg"):
            print("\nTally TOTALS IMAGE USED\n", file)
            try:
                temp_ = llm_call.run({"image_path": file, "context": guidelines, "return_amount":True})
            except Exception as e:
                temp_ = 0
            print("type temp", type(temp_))
            print("temp", temp_)
            image_total = int(float(temp_))
            excel_total = int(float(excel_output["excel_total"]))
            if abs(image_total-excel_total) < 1000:    
                result["valid"] = "yes"
                result["comments"] = "The totals of the image has match with the totals from the evidence excel."
            else:
                result["valid"] = "no"
                result["comments"] = "The totals of the image does not match with the totals from the evidence excel."

    state["tally_checkpoints"] = result
    print(f"Tally results: {state['tally_checkpoints']}")
    return state

def evaluator(state: AgentState) -> AgentState:
    """ Ends the workflow. """
    ipe_data = state["ipe_checkpoints"]
    tally_data = state["tally_checkpoints"]
    data = f"""
    This is the observed outputs (checkpoints) for the individual ipe and tally totals checkpoints:
    1. ipe_checkpoints = {str(ipe_data)}
    2. tally_checkpoints = {str(tally_data)}
    """
    summary = llm_call.run({"for_text":True, "data":data})
    state["summary"] = summary
    print("\n\nOVERALL SUMMARY:\n", state["summary"])
    return state


def should_continue(state: AgentState):
    messages = state["llm_decision"]
    # If there are no tool calls, then we finish
    if state["llm_decision"]:
        return "continue"
    # Otherwise if there is, we continue
    else:
        return "end"
    
def scope_continue(state:AgentState):
    status = state["scope"]
    if status!="Process failed.":
        return "continue"
    else:
        return "end"