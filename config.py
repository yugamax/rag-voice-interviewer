import os
from dotenv import load_dotenv
import base64

load_dotenv()

# All Groq API keys (gr_api_key1 ... gr_api_key6)
GROQ_API_KEYS = [os.getenv(f"gr_api_key{i}") for i in range(1, 7)]
NON_EMPTY_GROQ_KEYS = [k for k in GROQ_API_KEYS if k]

GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# Service account JSON (full JSON string) - use in deployments where file upload isn't possible
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
# Support base64-encoded JSON in env (useful for GitHub Secrets / CI)
GOOGLE_CREDENTIALS_JSON_B64 = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON_B64")
if not GOOGLE_CREDENTIALS_JSON and GOOGLE_CREDENTIALS_JSON_B64:
    try:
        # some systems may include surrounding quotes; strip them
        raw = GOOGLE_CREDENTIALS_JSON_B64.strip("\"'\n \r")
        GOOGLE_CREDENTIALS_JSON = base64.b64decode(raw).decode("utf-8")
    except Exception as e:
        # leave as None on failure; firebase_client will handle fallback and log
        print(f"[config] Failed to decode GOOGLE_APPLICATION_CREDENTIALS_JSON_B64: {e}")
        GOOGLE_CREDENTIALS_JSON = None

# Firestore collections (can override via env if you want)
# INTERVIEW_CONTEXT_COLLECTION = os.getenv("INTERVIEW_CONTEXT_COLLECTION", "interview_context")
# INTERVIEW_QUESTIONS_COLLECTION = os.getenv("INTERVIEW_QUESTIONS_COLLECTION", "interview_questions")

# LLM key (can override via GROQ_LLM_API_KEY env)
GROQ_LLM_API_KEY = (
    os.getenv("GROQ_LLM_API_KEY")
    or (NON_EMPTY_GROQ_KEYS[0] if NON_EMPTY_GROQ_KEYS else None)
)

if GROQ_LLM_API_KEY is None:
    raise RuntimeError("No Groq API key found for LLM. Set gr_api_key1..6 or GROQ_LLM_API_KEY.")
