import uuid
import json
import os
from typing import List, Optional, Dict, Any

import firebase_admin
from firebase_admin import credentials, firestore

from config import (
    GOOGLE_CREDENTIALS_PATH,
    # JSON string form of credentials for env-only deployments
    GOOGLE_CREDENTIALS_JSON,
    INTERVIEW_QUESTIONS_COLLECTION,
)

# ---- Firebase init ----
if not firebase_admin._apps:
    # Prefer an explicit file path if it exists
    if GOOGLE_CREDENTIALS_PATH and os.path.exists(GOOGLE_CREDENTIALS_PATH):
        print(f"[Firebase] Initializing from credentials file: {GOOGLE_CREDENTIALS_PATH}")
        cred = credentials.Certificate(GOOGLE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
    # If a JSON string is provided (useful for deployments), parse and use it
    elif GOOGLE_CREDENTIALS_JSON:
        try:
            print("[Firebase] Initializing from credentials provided in environment (GOOGLE_APPLICATION_CREDENTIALS_JSON or decoded B64)")
            cred_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        except Exception as e:
            # Fall back to initialize_app() to allow other auth methods, but log error
            print(f"[Firebase] Failed to parse GOOGLE_CREDENTIALS_JSON: {e}")
            print("[Firebase] Falling back to default firebase_admin.initialize_app()")
            firebase_admin.initialize_app()
    else:
        print("[Firebase] No credentials provided; initializing default app (may fail if no default credentials available)")
        firebase_admin.initialize_app()

db = firestore.client()


# ---- Interview questions ----

def load_interview_questions(interview_id: str) -> List[str]:
    """
    Try to load questions from:
    1) interviews/{interviewId}/questions subcollection, ordered by 'order'
    2) global INTERVIEW_QUESTIONS_COLLECTION where interviewId == interview_id
    If none found, return a simple default list.
    """
    questions: List[str] = []

    # Option 1: subcollection under interviews/{interview_id}/questions
    sub_ref = db.collection("interviews").document(interview_id).collection("questions")
    sub_docs = list(sub_ref.stream())
    if sub_docs:
        sub_docs_sorted = sorted(
            sub_docs,
            key=lambda d: (d.to_dict() or {}).get("order", 0)
        )
        for d in sub_docs_sorted:
            data = d.to_dict() or {}
            q_text = data.get("text")
            if q_text:
                questions.append(q_text)

    if questions:
        print(f"[Interview] Loaded {len(questions)} questions from interviews/{interview_id}/questions")
        return questions

    # Option 2: global collection with interviewId field
    col_ref = db.collection(INTERVIEW_QUESTIONS_COLLECTION)
    global_docs = list(col_ref.where("interviewId", "==", interview_id).stream())
    questions = []
    for d in global_docs:
        data = d.to_dict() or {}
        q_text = data.get("text")
        if q_text:
            questions.append(q_text)

    if questions:
        print(f"[Interview] Loaded {len(questions)} questions from '{INTERVIEW_QUESTIONS_COLLECTION}' for interviewId={interview_id}")
        return questions

    # Fallback: default questions
    print("[Interview] No questions found in Firestore. Using default questions.")
    return [
        "Can you briefly introduce yourself?",
        "Why are you interested in this role?",
        "Tell me about a challenging project you worked on and how you handled it.",
    ]


# ---- Candidate responses ----

def save_user_response(
    interview_id: str,
    user_id: str,
    question_index: int,
    question_text: str,
    answer_text: str,
    metrics: Optional[Dict[str, Any]] = None,
    attempt_number: int = 1,
):
    """
    Store every user response in Firestore under:
    interviews/{interviewId}/responses/{autoId}
    """
    try:
        doc_ref = (
            db.collection("interviews")
            .document(interview_id)
            .collection("responses")
            .document()
        )
        server_ts = getattr(firestore, "SERVER_TIMESTAMP", None)
        doc_ref.set(
            {
                "interviewId": interview_id,
                "userId": user_id,
                "questionIndex": question_index,
                "question": question_text,
                "answer": answer_text,
                "metrics": metrics or {},
                "attemptNumber": attempt_number,
                "createdAt": server_ts,
            }
        )
        print(f"[Interview] Saved response for interview={interview_id}, q_index={question_index}")
    except Exception as e:
        print(f"[Interview] Error saving response: {e}")


def save_interview_score(
    interview_id: str,
    user_id: str,
    score: int,
    justification: str,
) -> int:
    """
    Store the final interview score in Firestore.
    Saved to: interviews/{interviewId}
    Also increments the attempt count for this user on this interview.
    Returns the attempt count.
    """
    try:
        doc_ref = db.collection("interviews").document(interview_id)
        server_ts = getattr(firestore, "SERVER_TIMESTAMP", None)
        
        # Get current attempt count (if any)
        existing_doc = doc_ref.get()
        attempt_count = 1
        if existing_doc.exists:
            data = existing_doc.to_dict() or {}
            attempt_count = (data.get("attempt_count", 0) or 0) + 1
        
        doc_ref.set(
            {
                "interviewId": interview_id,
                "userId": user_id,
                "score": score,
                "justification": justification,
                "attempt_count": attempt_count,
                "completedAt": server_ts,
            },
            merge=True  # Merge with existing document if any metadata already exists
        )
        print(f"[Interview] Saved final score {score}/100 for interview={interview_id} (attempt #{attempt_count})")
        return attempt_count
    except Exception as e:
        print(f"[Interview] Error saving score: {e}")
        return 1


def get_session_with_events(user_id: str, session_id: str) -> Dict[str, Any]:
    """
    Fetch a user's session document and its `events` subcollection.

    Returns a dict: {"session": <session_doc_or_None>, "events": [event_dicts...]}
    Events are sorted by `timestamp` (ascending).
    """
    try:
        session_ref = db.collection("users").document(user_id).collection("sessions").document(session_id)
        session_doc = session_ref.get()
        session = session_doc.to_dict() if session_doc.exists else None

        events_ref = session_ref.collection("events")
        events = [ (e.id, e.to_dict() or {}) for e in events_ref.stream() ]
        # Extract dicts and sort by timestamp if present
        events_only = [e for (_id, e) in events]
        events_sorted = sorted(events_only, key=lambda x: x.get("timestamp", 0))

        return {"session": session, "events": events_sorted}
    except Exception as e:
        print(f"[Firebase] Error fetching session/events for {user_id}/{session_id}: {e}")
        return {"session": None, "events": []}
