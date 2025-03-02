# tools.py
import os
import openai
import pandas as pd
from typing import Dict, Any, List
from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
import re
import base64
import json


openai.api_key = os.getenv("openai_api_key")
openai.azure_endpoint = os.getenv("openai_azure_endpoint")

llm = AzureChatOpenAI(
    deployment_name="gpt-4o-mini",
    model="gpt-4o-mini",
    temperature=0.01,
    max_tokens=4000,
    openai_api_key=openai.api_key,
    azure_endpoint=openai.azure_endpoint,
)

# Tool functions
def gather_scope(file_path: str, control_number: str) -> Dict[str, List[str]]:
    """
    Extracts relevant details from an Excel file based on the given control number.

    :param file_path: Path to the Excel file.
    :param control_number: The control number to filter data.
    :return: A dictionary with categorized details.
    """
    df = pd.read_excel(file_path, dtype=str) 
    row = df[df["control_number"] == control_number]
    ipe_columns = [col for col in df.columns if col.startswith("IPE") and "totals" not in col]
    tally_totals_column = [col for col in df.columns if "totals" in col]
    def clean_values(series):
        return [str(value).replace("\n", " ").strip() for value in series.values.flatten() if pd.notna(value)]
    ipe_details = clean_values(row[ipe_columns]) if not row.empty else []
    tally_totals = clean_values(row[tally_totals_column]) if not row.empty else []
    scope = {
        "IPE": ipe_details,
        "Tally totals": tally_totals,
        "CPT": []  
    }
    return scope

# def extract_evidence_tool(user_query: str) -> Dict[str, List[str]]:
#     """Extracts evidence filenames from user query using regex."""
#     match = re.search(r"\b(\d+_IPE\d+\.jpg)\b", user_query)
#     return {"evidences": [match.group(1)]} if match else {"evidences": []}

# def fetch_evidence_tool(evidences_required: Dict[str, Any]) -> Dict[str, Any]:
#     """Fetches file paths for evidence images, verifying their existence."""
#     landing_folder = "/content/Landing/"
#     file_paths = [os.path.join(landing_folder, filename) for filename in evidences_required]
#     existing_files = [path for path in file_paths if os.path.exists(path)]
#     return {"image_path": existing_files[0]} if existing_files else {"image_path": None}

def process_evidence(user_query: str) -> Dict[str, Any]:
    """Extracts multiple evidence filenames from user query and fetches their valid file paths."""

    # Step 1: Extract all evidence filenames using regex
    matches = re.findall(r"\b(\d+_IPE\d+\.jpg)\b", user_query)
    evidences = matches if matches else []

    # Step 2: Fetch file paths for valid evidence files
    landing_folder = "/content/Landing/"
    file_paths = [os.path.join(landing_folder, filename) for filename in evidences]

    # Step 3: Filter only the files that exist
    existing_files = [path for path in file_paths if os.path.exists(path)]

    return {
        "evidences": evidences,  # List of extracted filenames
        "file_paths": existing_files  # List of valid file paths
    }

def image_analysis_tool(query: str, image_path: str) -> Dict[str, Any]:
    """Encodes the image to Base64 and sends it to GPT-4o for validation."""
    if not image_path:
        return {"error": "Image path missing."}
    
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": query},
            {"type": "text", "text": "You will return your reponse strictly in this json format result = {{ 'valid': 'success' or 'failure', 'comments': your comments for the process. Make sure you have addressed in sufficient detail.}}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}
    ]
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.01,
        max_tokens=500
    )
    
    answer = response.choices[0].message.content.lower()
    answer = re.sub(r"```json\n(.*?)\n```", r"\1", answer, flags=re.DOTALL).strip()
    # result = "yes" in answer or "correct" in answer or "valid" in answer
    # return {"valid": result, "errors": [] if result else ["Validation failed."]}
    try:
        json_output = json.loads(answer)
        return json_output  # Ensure we return the exact JSON response
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format returned by the model.", "raw_response": answer}

class ImageAnalysisInput(BaseModel):
    query: str
    image_path: str

# Tool instances
extract_evidence = Tool(
    name="ExtractEvidence",
    func=extract_evidence_tool,
    description="Extracts evidence filenames from user query."
)

fetch_evidence = Tool(
    name="FetchEvidence",
    func=fetch_evidence_tool,
    description="Fetches file paths of required evidences."
)

image_analysis = StructuredTool(
    name="ImageAnalysis",
    func=image_analysis_tool,
    args_schema=ImageAnalysisInput,
    description="Analyzes image fields using GPT-4."
)

def process_excel_calculations(file_path: str, filter_column: str = "Account", filter_value: str = "23040100", calculation_column: str = "Amount") -> Dict[str, Any]:
    try:
        df = pd.read_excel(file_path, dtype=str)
        if filter_column not in df.columns or calculation_column not in df.columns:
            return {"error": f"Missing required columns: '{filter_column}' or '{calculation_column}'"}
        filtered_df = df[df[filter_column] == str(filter_value)].copy()
        if filtered_df.empty:
            return {"error": f"No matching rows found for {filter_column} = {filter_value}"}
        total = 0
        for value in filtered_df[calculation_column].dropna(): 
            value = str(value).replace(",", "") 

            if value.startswith("-"):
                total -= float(value[1:]) 
            else:
                total += float(value)  
        total_ = int(abs(total))
        print("Total:", total_)
        return {"excel_total": total_}
    except Exception as e:
        return {"error": str(e)}

excel_reader = Tool(
    name="ExcelReader",
    func=process_excel_calculations,
    description="Reads an Excel file, filters, and calculates totals."
)
