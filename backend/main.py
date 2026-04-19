import os
import json
import re
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
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    resume_text: str
    job_description: str


@app.get("/")
async def root():
    return {"message": "Resume analyzer API is running"}


def normalize_score_breakdown(score_breakdown):
    """
    Converts AI-generated score breakdown into a clean object with
    dynamic keys and integer values.
    """
    if not isinstance(score_breakdown, dict):
        return {}

    cleaned = {}

    for key, value in score_breakdown.items():
        if not isinstance(key, str):
            continue

        normalized_key = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
        if not normalized_key:
            continue

        try:
            cleaned[normalized_key] = int(value)
        except (TypeError, ValueError):
            cleaned[normalized_key] = 0

    return cleaned


@app.post("/analyze")
async def analyze_resume(data: AnalysisRequest):
    try:
        prompt = f"""
You are a resume analysis assistant.

Analyze how well the resume matches the job description for ANY profession or industry.

Instructions:
- Read the resume and job description carefully.
- Identify the 6 to 8 most important qualifications, skills, experience areas, tools, certifications, responsibilities, or traits from the job description.
- Create a score_breakdown object using category names based on the actual job description.
- Each key in score_breakdown must be a short snake_case category name.
- Each value in score_breakdown must be an integer score.
- The total of all category scores must equal match_score.
- Award only integer points.
- Use partial credit when relevant.
- Score only based on evidence clearly present in the resume.
- Do not assume qualifications that are not explicitly stated.
- missing_keywords should list important missing terms, tools, skills, credentials, or qualifications.
- suggested_skills should list the highest-value areas the user could strengthen.
- strengths should list the strongest matches between the resume and the role.
- Return ONLY valid JSON.
- Do not include markdown fences.
- Do not include explanation before or after the JSON.

Required JSON format:
{{
  "match_score": 0,
  "score_breakdown": {{
    "category_name_here": 0
  }},
  "missing_keywords": [],
  "suggested_skills": [],
  "strengths": []
}}

Important rules for score_breakdown:
- Must contain 6 to 8 categories
- Categories must be specific to the job description
- Keys must be snake_case
- Values must be integers
- Do not use generic labels like category_1
- Do not hardcode software-only categories unless the job description actually requires them

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

        result = json.loads(response.text)

        score_breakdown = normalize_score_breakdown(result.get("score_breakdown", {}))
        match_score = sum(score_breakdown.values())

        formatted_result = {
            "matchScore": match_score,
            "scoreBreakdown": score_breakdown,
            "missingKeywords": result.get("missing_keywords", []),
            "suggestedSkills": result.get("suggested_skills", []),
            "strengths": result.get("strengths", []),
            "jobDescription": data.job_description,
            "resumeText": data.resume_text,
        }

        return formatted_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))