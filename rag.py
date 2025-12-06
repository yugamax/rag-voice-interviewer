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


def build_vectorstore_from_firestore() -> Optional[FAISS]:
    """
    Pulls documents from Firestore and builds a FAISS vector store.
    Expects each document to have a 'content' (or 'text') field.
    """
    collection_ref = db.collection(INTERVIEW_CONTEXT_COLLECTION)
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for doc_snap in collection_ref.stream():
        data = doc_snap.to_dict() or {}
        content = data.get("content") or data.get("text")
        if not content:
            continue

        meta = {k: v for k, v in data.items() if k not in ("content", "text")}
        meta["doc_id"] = doc_snap.id

        texts.append(content)
        metadatas.append(meta)

    if not texts:
        print("[RAG] No documents found in Firestore collection:", INTERVIEW_CONTEXT_COLLECTION)
        return None

    print(f"[RAG] Loaded {len(texts)} docs from Firestore.")
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
    )
    return vectorstore


vectorstore = build_vectorstore_from_firestore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) if vectorstore is not None else None

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


INTERVIEWER_PROMPT_TEMPLATE = """
You are a professional AI interviewer conducting a live job interview. Your tone must be formal, polite, calm, and encouraging — not robotic or harsh.

You must strictly follow these rules:
- Don't laugh, or make sounds or say "uhm", "ah", etc.
- You are in the middle of an interview.
- Never say the candidate's name.
- After each candidate answer, first give a VERY short, balanced, and professional review of the answer (1–3 sentences, max 40 words). 
  The feedback should be constructive, supportive, and never overly harsh.
- Then, if there is a next question, ask it in a natural, conversational way (do NOT label it as Question 1, Question 2, etc.).
  Use phrasing like: “My next question is…”, “Let’s move on to…”, or “I’d like to ask you about…”.
- Important: If there is NO next question (this is the last one), instead give a brief overall review of the candidate's performance (max 80 words) and then say something like:
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

Next question (if any):
{next_question}

Is this the last question? {is_last_question}

Now produce your response in plain text, following the rules above.
"""



def generate_interviewer_reply(
    user_answer: str,
    chat_hist: List[Dict[str, str]],
    current_question: str,
    next_question: Optional[str],
) -> str:
    """Use LangChain + Groq + Firestore-backed RAG to review the answer and ask next question / end interview."""
    history_str = format_history(chat_hist)

    # Build RAG context from current question + answer
    context_text = ""
    if retriever is not None:
        query = f"{current_question}\n{user_answer}"
        docs = retriever.invoke(query)
        context_text = "\n\n".join(d.page_content for d in docs)

    prompt = INTERVIEWER_PROMPT_TEMPLATE.format(
        context=context_text,
        history=history_str,
        current_question=current_question,
        user_answer=user_answer,
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
