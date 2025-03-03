# tools.py
import os
import openai
import pandas as pd
from typing import Dict, Any, List
from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel
# from langchain.chat_models import AzureChatOpenAI
from langchain_openai import AzureChatOpenAI
import re
import base64
import json


openai.api_key = os.getenv("openai_api_key")
openai.azure_endpoint = os.getenv("openai_azure_endpoint")
openai.api_type = "azure"

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
    print(f"Gathering scope from {file_path} for control number {control_number}")
    df = pd.read_excel(file_path, dtype=str) 
    row = df[df["control_number"] == control_number]
    
    if row.empty:
        print("No matching rows found for control number.")
    
    ipe_columns = [col for col in df.columns if col.startswith("IPE") and "totals" not in col]
    tally_totals_column = [col for col in df.columns if "totals" in col]

    def clean_values(series):
        return [str(value).replace("\n", " ").strip() for value in series.values.flatten() if pd.notna(value)]
    
    ipe_details = clean_values(row[ipe_columns]) if not row.empty else []
    tally_totals = clean_values(row[tally_totals_column]) if not row.empty else []

    scope = {
        "IPE": ipe_details,
        "Tally": tally_totals,
        "CPT": []  
    }
    print(f"Extracted scope: {scope}")
    return scope

def extract_evidence_tool(user_query: str) -> Dict[str, List[str]]:
    print(f"Extracting evidence from query: {user_query}")
    matches = re.findall(r"\b(\d+_IPE\d+\.(jpg|jpeg|png|xls|xlsx))\b", user_query, re.IGNORECASE)
    evidences = [match[0] for match in matches] if matches else []
    print(f"Extracted evidences: {evidences}")
    return {"evidences": evidences}

def fetch_evidence_tool(evidences_required: Dict[str, Any]) -> Dict[str, Any]:
    print(f"Fetching evidences: {evidences_required}")
    landing_folder = r"C:\Users\UV172XK\code@ey\Agentic AI\my_app\langgraph-framework\my_agent\Landing"
    file_paths = [os.path.join(landing_folder, filename) for filename in evidences_required]
    existing_files = [path for path in file_paths if os.path.exists(path)]
    print(f"Existing files found: {existing_files}")
    return {"image_path": existing_files} 

def image_analysis_tool(query: str, image_path: str) -> Dict[str, Any]:
    print(f"Analyzing image: {image_path} for query: {query}")
    
    if not image_path:
        return {"error": "Image path missing."}
    
    with open(image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    
    messages = [
        {"role": "system", "content": """You must strictly respond in JSON format as follows:

        {{
            "valid": "yes" or "no",
            "comments": "Comment on the observation (always mention the image being analyzed)"
        }}

        Do not include any other text, explanations, or code blocks. Just return the JSON.
        """},
        {"role": "user", "content": [
            {"type": "text", "text": query},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}
    ]

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.01,
        max_tokens=4000
    )
    
    answer = response.choices[0].message.content.lower()
    print(f"Raw LLM response: {answer}")
    
    try:
        json_output = json.loads(answer)
        print(f"Parsed JSON output: {json_output}")
        return json_output  
    except json.JSONDecodeError:
        print("Error: Invalid JSON format returned by the model.")
        return {"error": "Invalid JSON format returned by the model.", "raw_response": answer}

class ImageAnalysisInput(BaseModel):
    query: str
    image_path: str

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

def process_excel_calculations(file_path: str) -> Dict[str, Any]:
    print(f"Processing Excel file: {file_path}")
    
    try:
        df = pd.read_excel(file_path, dtype=str)
        print(f"Columns in Excel: {df.columns}")
        
        total = sum(float(str(value).replace(",", "").strip()) for value in df["Amount in local currency"].dropna())
        total_ = int(abs(total))
        print(f"Calculated total: {total_}")
        return {"excel_total": total_}
    
    except Exception as e:
        print(f"Error processing Excel: {e}")
        return {"error": str(e)}

excel_reader = Tool(
    name="ExcelReader",
    func=process_excel_calculations,
    description="Reads an Excel file, filters, and calculates totals."
)