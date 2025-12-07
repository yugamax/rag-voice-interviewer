import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import SecretStr


from config import (
    INTERVIEW_CONTEXT_COLLECTION,
    NON_EMPTY_GROQ_KEYS,
)
from firebase_client import db

# ---- Embeddings + Vector store ----

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def build_vectorstore_from_firestore(interview_id: Optional[str] = None) -> Optional[FAISS]:
    """
    Pulls documents from Firestore and builds a FAISS vector store.
    If interview_id is provided, filters to that interview's context only.
    Falls back to all context docs if no interview_id.
    Expects each document to have a 'content' (or 'text') field.
    """
    collection_ref = db.collection(INTERVIEW_CONTEXT_COLLECTION)
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    if interview_id:
        # Filter by interview_id
        query = collection_ref.where("interviewId", "==", interview_id)
        docs_to_process = list(query.stream())
        print(f"[RAG] Loading context for interview {interview_id}...")
    else:
        # Load all (fallback)
        docs_to_process = list(collection_ref.stream())
        print(f"[RAG] Loading all context (no interview_id filter)...")

    for doc_snap in docs_to_process:
        data = doc_snap.to_dict() or {}
        content = data.get("content") or data.get("text")
        if not content:
            continue

        meta = {k: v for k, v in data.items() if k not in ("content", "text")}
        meta["doc_id"] = doc_snap.id

        texts.append(content)
        metadatas.append(meta)

    if not texts:
        print(f"[RAG] No documents found for interview {interview_id or 'all'} in collection: {INTERVIEW_CONTEXT_COLLECTION}")
        return None

    print(f"[RAG] Loaded {len(texts)} context docs for interview {interview_id or 'all'}.")
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
    )
    return vectorstore

# Vectorstore will be built per-interview in the websocket endpoint
vectorstore = None
retriever = None

# ---- LLM with key failover (uses NON_EMPTY_GROQ_KEYS only) ----

if not NON_EMPTY_GROQ_KEYS:
    raise RuntimeError("No Groq API keys configured for LLM.")

llm_clients = [
    ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=SecretStr(k),
    )
    for k in NON_EMPTY_GROQ_KEYS
]

# ---- Prompt helpers ----

def format_history(chat_hist: List[Dict[str, str]]) -> str:
    """Convert stored history into a plain-text conversation summary for the prompt."""
    lines: List[str] = []
    for msg in chat_hist:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"Candidate: {content}")
        elif role == "assistant":
            lines.append(f"Interviewer: {content}")
    return "\n".join(lines)


def format_metrics(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return "Audio analysis not available for this response."
    
    parts = []
    speech_dur = metrics.get('speechDuration')
    total_dur = metrics.get('totalDuration')
    silence_ratio = metrics.get('silenceRatio')
    pause_count = metrics.get('pauseCount')
    start_latency = metrics.get('startLatency')
    
    if speech_dur and total_dur:
        parts.append(f"spoke for {speech_dur}s out of {total_dur}s total recording")
    
    if silence_ratio is not None:
        if silence_ratio < 0.2:
            parts.append("very fluent with minimal pauses")
        elif silence_ratio < 0.4:
            parts.append("good fluency with some natural pauses")
        elif silence_ratio < 0.6:
            parts.append("moderate pauses during speech")
        else:
            parts.append("significant pauses or hesitation")
    
    if pause_count is not None:
        if pause_count == 0:
            parts.append("no noticeable breaks")
        elif pause_count <= 2:
            parts.append(f"{pause_count} brief pause(s)")
        else:
            parts.append(f"{pause_count} pauses detected")
    
    if start_latency is not None:
        if start_latency < 1:
            parts.append("responded immediately")
        elif start_latency < 3:
            parts.append(f"started speaking after {start_latency:.1f}s")
        else:
            parts.append(f"took {start_latency:.1f}s to begin response")
    
    return "; ".join(parts) if parts else "Audio captured successfully"


INTERVIEWER_PROMPT_TEMPLATE = """
You are a professional AI interviewer conducting a live job interview. Your tone must be formal, polite, calm, and encouraging — not robotic or harsh.

You must strictly follow these rules:
- important: Don't laugh, or make sounds or say "uhm", "ah", etc.
- You are in the middle of an interview.
- Never say the candidate's name.
- After each candidate answer, provide a VERY short, balanced review (1–3 sentences, max 40 words) with this weighting:
  - 60% focus on DELIVERY METRICS: speaking pace, clarity, confidence, pauses, fluency, and response time
  - 40% focus on CONTENT: relevance and completeness of the answer
  The feedback should be constructive and supportive, highlighting both delivery quality and content.
- Then, if there is a next question, ask it in a natural, conversational way (do NOT label it as Question 1, Question 2, etc.).
  Use phrasing like: "My next question is…", "Let's move on to…", or "I'd like to ask you about…".
- Important: If there is NO next question (this is the last one), instead give a comprehensive overall review of the candidate's performance (50-60 words) with this emphasis:
  - 60% on delivery: Discuss their overall communication clarity, speaking confidence, response timing, pacing consistency, fluency, and how well they maintained engagement throughout the interview. Comment on their vocal presence and presentation style.
  - 40% on content: Address their technical knowledge, problem-solving approach, depth of answers, and relevance to the questions asked.
  Provide specific observations from across all their responses, highlighting strengths and areas for improvement.
  Then say something like:
  "All questions have been asked and the interview is over."

Use the following job-related context if it is relevant:

<context>
{context}
</context>

Conversation so far:
{history}

Current question the candidate just answered:
{current_question}

Candidate's answer:
{user_answer}

Audio delivery metrics (use prominently in your feedback):
{audio_metrics}

Next question (if any):
{next_question}

Is this the last question? {is_last_question}

Now produce your response in plain text, strictly following all the rules above.
"""




def generate_interviewer_reply(
    user_answer: str,
    chat_hist: List[Dict[str, str]],
    current_question: str,
    next_question: Optional[str],
    metrics: Optional[Dict[str, Any]] = None,
    retriever: Optional[Any] = None,
) -> str:
    """Use LangChain + Groq + Firestore-backed RAG to review the answer and ask next question / end interview."""
    history_str = format_history(chat_hist)
    metrics_str = format_metrics(metrics)

    # Build RAG context from current question + answer
    context_text = ""
    if retriever is not None:
        try:
            query = f"{current_question}\n{user_answer}"
            docs = retriever.invoke(query)
            context_text = "\n\n".join(d.page_content for d in docs)
        except Exception as e:
            print(f"[RAG] Error retrieving context: {e}")
            context_text = ""

    prompt = INTERVIEWER_PROMPT_TEMPLATE.format(
        context=context_text,
        history=history_str,
        current_question=current_question,
        user_answer=user_answer,
        audio_metrics=metrics_str,
        next_question=next_question or "",
        is_last_question="yes" if next_question is None else "no",
    )

    last_error: Optional[Exception] = None
    for i, client in enumerate(llm_clients, start=1):
        try:
            ai_msg = client.invoke(prompt)
            return getattr(ai_msg, "content", str(ai_msg)).strip()
        except Exception as exc:  # pragma: no cover - external service
            last_error = exc
            print(f"[RAG] LLM client {i} failed: {exc}")

    raise RuntimeError(f"All Groq LLM keys failed. Last error: {last_error}")


def generate_final_score(
    chat_hist: List[Dict[str, str]],
    questions: List[str],
    metrics_by_question: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Use LLM to evaluate the entire interview and generate a final score (0-100) with justification.
    
    Args:
        chat_hist: Full conversation history including all Q&A pairs
        questions: List of all interview questions that were asked
    
    Returns:
        Dict with 'score' (int 0-100) and 'justification' (str)
    """
    history_str = format_history(chat_hist)
    questions_str = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    
    metrics_lines = []
    if metrics_by_question:
        for idx in sorted(metrics_by_question.keys()):
            m = format_metrics(metrics_by_question.get(idx))
            metrics_lines.append(f"Q{idx+1}: {m}")
    metrics_block = "\n".join(metrics_lines) if metrics_lines else "No audio metrics provided."

    scoring_prompt = f"""
You are an expert interviewer evaluating a candidate's interview performance.

Below are the questions asked and the full conversation history:

<questions>
{questions_str}
</questions>

<conversation>
{history_str}
</conversation>

Audio delivery metrics per question (weight these heavily in scoring):
{metrics_block}

Evaluate the candidate's performance with this weighting:
- 60% weight: Delivery quality - communication clarity, speaking confidence, pacing, fluency, response timing, and overall presentation
- 40% weight: Content quality - technical knowledge, problem-solving approach, depth of understanding, and answer completeness

IMPORTANT INSTRUCTIONS:
- Prioritize delivery metrics in your evaluation, as strong communication skills are critical for this role.
- If some questions lack detailed audio metrics, focus your evaluation on the questions where metrics ARE available and the overall conversation quality.
- DO NOT mention "lack of audio metrics" or "incomplete metrics" in your justification.
- Use natural, conversational language - avoid technical metric terms.

Provide:
1. A numerical score from 0 to 100 (0=very poor, 100=excellent)
2. A detailed justification (5-7 sentences, 100-120 words) explaining the score. Be specific about both delivery quality and content quality. Be candid and bluntly honest while staying professional.

Format your response EXACTLY as:
SCORE: <number>
JUSTIFICATION: <text>
"""
    
    last_error: Optional[Exception] = None
    for i, client in enumerate(llm_clients, start=1):
        try:
            ai_msg = client.invoke(scoring_prompt)
            response = getattr(ai_msg, "content", str(ai_msg)).strip()
            
            # Parse the response
            score_line = ""
            justification_line = ""
            for line in response.split("\n"):
                if line.startswith("SCORE:"):
                    score_line = line.replace("SCORE:", "").strip()
                elif line.startswith("JUSTIFICATION:"):
                    justification_line = line.replace("JUSTIFICATION:", "").strip()
            
            # Extract score number
            try:
                score = int(score_line)
                if not (0 <= score <= 100):
                    score = max(0, min(100, score))  # Clamp to valid range
            except (ValueError, TypeError):
                # If parsing fails, try to extract first number from response
                import re
                numbers = re.findall(r'\b\d+\b', response)
                score = int(numbers[0]) if numbers else 50  # Default to 50 if no number found
            
            justification = justification_line if justification_line else response
            
            return {
                "score": score,
                "justification": justification,
            }
            
        except Exception as exc:  # pragma: no cover - external service
            last_error = exc
            print(f"[RAG] Scoring LLM client {i} failed: {exc}")
    
    raise RuntimeError(f"All Groq LLM keys failed during scoring. Last error: {last_error}")
