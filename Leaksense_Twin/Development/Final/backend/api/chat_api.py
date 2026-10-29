from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
import os
import pandas as pd
from openai import OpenAI
import json

router = APIRouter(prefix="/api/chat", tags=["Chat Advisor"])

# Configuration — read from environment, never hardcode a credential here again.
# Set NVIDIA_API_KEY in your local .env (gitignored) or shell environment.
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "google/gemma-3-12b-it"

client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=NVIDIA_API_KEY or "missing-key"
) if NVIDIA_API_KEY else None

class ChatRequest(BaseModel):
    message: str
    history: list = []

def get_project_context():
    """Load context from GEMINI.md and Excel files."""
    context = ""
    
    # 1. Load GEMINI.md
    try:
        # chat_api.py is in backend/api/
        # base_dir should be the project root
        api_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(api_dir)
        root_dir = os.path.dirname(backend_dir)
        
        gemini_path = os.path.join(root_dir, "GEMINI.md")
        if os.path.exists(gemini_path):
            with open(gemini_path, "r", encoding="utf-8") as f:
                context += "PROJECT DOCUMENTATION:\n" + f.read()[:3000] + "\n\n"
        
        data_dir = os.path.join(backend_dir, "Data")
        files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
        context += "DATASET OVERVIEW:\n"
        for file in files:
            file_path = os.path.join(data_dir, file)
            df = pd.read_excel(file_path, nrows=5)
            context += f"File: {file}\nColumns: {', '.join(df.columns)}\n"
            context += f"Sample Data:\n{df.to_string(index=False)}\n\n"
    except Exception as e:
        print(f"Error loading Excel data: {e}")

    return context

@router.post("/ask")
async def ask_chatbot(request: Request, body: ChatRequest):
    if client is None:
        raise HTTPException(503, "Chat advisor not configured — set NVIDIA_API_KEY in the environment.")
    try:
        predictor = request.app.state.predictor
        
        # Get live sensor data if available
        live_sensors = {}
        if predictor and hasattr(predictor, 'window') and len(predictor.window) > 0:
            # Get latest sensor reading from window
            latest = predictor.window[-1]
            # Map back to names based on predictor.predict logic
            # [maf, map_boost, map_cac_out, t_cac_out, t_exh, dP_dpf, rpm]
            live_sensors = {
                "MAF": latest[0],
                "MAP_boost": latest[1],
                "MAP_cac_out": latest[2],
                "T_cac_out": latest[3],
                "T_exh": latest[4],
                "dP_dpf": latest[5],
                "RPM": latest[6]
            }

        project_context = get_project_context()
        
        system_prompt = f"""You are the LeakSense Twin AI Advisor, a specialized expert in Cat C18 diesel engine diagnostics.
Your goal is to help users understand engine health, leak detection results, and the underlying data.

{project_context}

CURRENT LIVE SENSOR STATUS:
{json.dumps(live_sensors, indent=2)}

Guidelines:
1. Use technical but accessible language.
2. If the user asks about live sensors, refer to the CURRENT LIVE SENSOR STATUS provided above.
3. If they ask about the dataset, refer to the DATASET OVERVIEW.
4. Keep answers concise and professional.
5. You can explain the Digital Twin approach and Energy Field Analysis if asked.
"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Clean and add history to ensure alternating user/assistant roles
        # Most APIs require the first message after system to be 'user'
        clean_history = []
        for msg in body.history:
            if not clean_history:
                if msg.get("role") == "user":
                    clean_history.append(msg)
            else:
                if msg.get("role") != clean_history[-1].get("role"):
                    clean_history.append(msg)
        
        # Append the cleaned history (limit to last 10 messages)
        for msg in clean_history[-10:]:
            messages.append(msg)
            
        # Ensure the next message is a user message
        # If the last message was already 'user', we should merge or skip
        if messages[-1].get("role") == "user":
            # Merge current message into the last user message
            messages[-1]["content"] += f"\n\n{body.message}"
        else:
            messages.append({"role": "user", "content": body.message})

        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=1024,
        )

        response = completion.choices[0].message.content
        return {"response": response}

    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
