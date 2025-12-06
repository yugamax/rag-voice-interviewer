import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# Load .env (so GOOGLE_APPLICATION_CREDENTIALS is available)
load_dotenv()

GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not GOOGLE_CREDENTIALS_PATH:
    raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set in .env")

# Init Firebase Admin
if not firebase_admin._apps:
    cred = credentials.Certificate(GOOGLE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ============================
# CONFIG – CHANGE THESE
# ============================

INTERVIEW_ID = "INTERVIEW_ID_124"   # 🔁 set this to whatever you want

QUESTIONS = [
    "Can you briefly introduce yourself?",
    "Why are you interested in this role?",
    "Explain a challenging technical problem you solved recently.",
    "How do you handle tight deadlines and pressure?",
    "Where do you see yourself in the next two to three years?"
]

# These are the RAG context snippets the interviewer can use
CONTEXT_DOCS = [
    {
        "content": "This role primary for people with experience in Python and machine learning.",
        "interviewId": INTERVIEW_ID,
        "topic": "tech-stack",
    },
    {
        "content": "The company values teamwork, innovation, and a strong commitment to customer satisfaction.",
        "interviewId": INTERVIEW_ID,
        "topic": "culture",
    },
    {
        "content": "The applicant must know backend deployment workflows, including Docker.",
        "interviewId": INTERVIEW_ID,
        "topic": "workflow",
    },
]

# ============================
# SEEDING LOGIC
# ============================

def seed_questions(interview_id, questions):
    questions_ref = (
        db.collection("interviews")
        .document(interview_id)
        .collection("questions")
    )

    for i, q in enumerate(questions, start=1):
        questions_ref.add(
            {
                "order": i,
                "text": q,
            }
        )
        print(f"✅ Added question {i}: {q}")

    print(f"\n🎉 Seeded {len(questions)} questions for interview: {interview_id}")


def seed_context(context_docs):
    ctx_ref = db.collection("interview_context")

    for doc in context_docs:
        # Make sure it's a dict and has content
        if isinstance(doc, dict):
            content = doc.get("content") or ""
        else:
            # If someone passes just a string instead of dict
            content = str(doc)
            doc = {"content": content}

        ctx_ref.add(doc)
        preview = content[:80].replace("\n", " ")
        print(f"✅ Added context: {preview}...")

    print(f"\n🎉 Seeded {len(context_docs)} context docs into 'interview_context'")


if __name__ == "__main__":
    seed_questions(INTERVIEW_ID, QUESTIONS)
    seed_context(CONTEXT_DOCS)
    print("\n✅ All seeding done.")
