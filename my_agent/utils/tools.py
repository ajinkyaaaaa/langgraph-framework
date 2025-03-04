# tools.py
import os
import openai
import pandas as pd
from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel
# from langchain.chat_models import AzureChatOpenAI
# from langchain_openai import AzureChatOpenAI
# from langchain_community.chat_models import AzureChatOpenAI
import re
import base64
import json
import time
from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict, Annotated
from utils.state import ImageAnalysisInput

os.environ["OPENAI_API_KEY"] = "cb4b1a0311454198ad4c9c42e9c4e5d7"
os.environ["OPENAI_AZURE_ENDPOINT"] = "https://swcdoai2x2aoa01.openai.azure.com"
os.environ["OPENAI_API_BASE"] = "https://swcdoai2x2aoa01.openai.azure.com"

openai.api_key = os.getenv("openai_api_key")
openai.azure_endpoint = os.getenv("openai_azure_endpoint")
openai.api_type = "azure"


# from langchain.chat_models import AzureChatOpenAI
# llm = AzureChatOpenAI(
#     deployment_name="gpt-4o-mini",
#     model="gpt-4o-mini",
#     temperature=0.01,
#     max_tokens=4000,
#     openai_api_key=openai.api_key,
#     azure_endpoint=openai.azure_endpoint,
# )

from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(
    azure_endpoint=os.environ["OPENAI_AZURE_ENDPOINT"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    deployment_name="gpt-4o-mini",
    openai_api_version="2023-05-15"  
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


def llm_function(
        query: Optional[str] = None,
        image_path: Optional[str] = None,
        data: Optional[Any] = None,
        context: Optional[str] = None,
        for_image: bool = False,
        for_text: bool = False,
        return_amount: bool = False,
    ) -> Dict[str, Any]:
    """
    Analyzes an image or text using GPT-4o.

    :param query: Query for image analysis (Required if for_image=True).
    :param image_path: Path to the image file (Required if for_image=True).
    :param data: Data for text analysis (Required if for_text=True).
    :param context: Additional context (Optional).
    :param for_image: Set to True for image-based analysis.
    :param for_text: Set to True for text-based analysis.
    :return: Dictionary with extracted insights or error messages.
    """

    if for_image and not (query and image_path):
        return {"error": "Missing required parameters: 'query' and 'image_path' are needed for image analysis."}

    if for_text and data is None:
        return {"error": "Missing required parameter: 'data' is needed for text analysis."}

    print(f"[INFO] Processing request | for_image: {for_image} | for_text: {for_text}")

    messages = []

    # Image Analysis
    if for_image:
        print(f"[INFO] Analyzing image: {image_path} | Query: {query}")

        # Convert image to Base64
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            return {"error": f"Failed to read image: {str(e)}"}

        # Define image-processing system instructions
        messages.append(
            {
                "role": "system",
                "content": (
                    "You must strictly respond in JSON format as follows:\n"
                    '{\n  "valid": "yes" or "no" (yes if all fields are filled correctly),\n  "comments": "Comment on the observation (mention the image being analyzed)"\n}\n'
                    "Do not include any other text, explanations, or code blocks—only return valid JSON."
                ),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                ],
            }
        )

    # Text Analysis
    if for_text:
        print(f"[INFO] Analyzing text data.")

        messages.append({"role": "system", "content": "You are a helpful assistant that summarizes workflows."})
        messages.append({"role": "user", "content": data})
        messages.append(
            {
                "role": "system",
                "content": (
                    "Summarize the given information by specifying which tests have passed and which have failed, along with reasons for failure.\n"
                    "Important: All IPEs must pass or be valid for the Tally Totals process to begin. If any fails, Tally Totals cannot proceed.\n"
                    "Return only the required output without explanations or additional text."
                ),
            }
        )

    if return_amount:
        print(f"[INFO] Analyzing image: {image_path} | Query: {query}")

        # Convert image to Base64
        try:
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as e:
            return {"error": f"Failed to read image: {str(e)}"}
        messages.append(
              {
                  "role": "user",
                  "content": [
                      {"type": "text", "text": f"This is what I need to validate: {query}. For this, return the amount that is found at the bottom portion of the screen, highlighted in YELLOW color. Your response should only be a number. Eg. 112345"},
                      {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                  ],
              }
          )

    # Add guidelines if provided
    if context:
        messages.append({"role": "system", "content": context})

    # Retry logic for API call
    max_retries = 30
    attempt = 0
    response = None
    while attempt < max_retries:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.01,
                max_tokens=500,
            )
            break  # Exit loop on success
        except Exception as e:
            print(f"[ERROR] API Call Failed (Attempt {attempt + 1}): {str(e)}")
            time.sleep(5)
            attempt += 1

    if not response:
        return {"error": "Failed to get a response from the model after multiple attempts."}

    answer = response.choices[0].message.content.strip()
    print(f"[INFO] Raw LLM Response: {answer}")

    # Process response for image analysis
    if for_image:
        try:
            json_output = json.loads(answer)
            print(f"[INFO] Parsed JSON Output: {json_output}")
            return json_output
        except json.JSONDecodeError:
            print("[ERROR] Invalid JSON format returned by the model.")
            return {"error": "Invalid JSON format from the model.", "raw_response": answer}

    # Return text response for text analysis
    return answer

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

llm_call = StructuredTool(
    name="ImageAnalysis",
    func=llm_function,
    args_schema=ImageAnalysisInput,
    description="Analyzes image fields using GPT-4."
)

def process_excel_calculations(file_path: str) -> Dict[str, Any]:
    try:
        filter_value = "23040100"
        filter_column = "Account"
        calculation_column = "Amount in local currency"
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
        return {"excel_total": total_}
    except Exception as e:
        return {"error": str(e)}

excel_reader = Tool(
    name="ExcelReader",
    func=process_excel_calculations,
    description="Reads an Excel file, filters, and calculates totals."
)