import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing")

client = genai.Client(api_key=API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # okay for testing, tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    resume_text: str
    job_description: str

@app.get("/")
async def root():
    return {"message": "Resume analyzer API is running"}

@app.post("/analyze")
async def analyze_resume(data: AnalysisRequest):
    try:
        prompt = f"""
You are a resume analysis assistant.

Score the resume against the job description using this exact rubric:

- Python/FastAPI: 20
- Frontend framework match: 15
- REST API design/integration: 10
- LLM/AI experience: 10
- AWS Cloud: 15
- Databases (PostgreSQL/Redis/SQL): 10
- Docker/containerization: 10
- Agile/collaboration: 10

Rules:
- Award only integer points
- Use partial credit when relevant
- match_score must equal the sum of the rubric categories
- Return ONLY valid JSON

Required JSON:
{{
  "match_score": 0,
  "score_breakdown": {{
    "python_fastapi": 0,
    "frontend_framework": 0,
    "rest_api": 0,
    "llm_ai": 0,
    "aws": 0,
    "databases": 0,
    "docker_containerization": 0,
    "agile_collaboration": 0
  }},
  "missing_keywords": [],
  "suggested_skills": [],
  "strengths": []
}}

Resume:
{data.resume_text}

Job Description:
{data.job_description}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        return json.loads(response.text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))