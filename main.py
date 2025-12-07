# main.py
import os
import tempfile
import json
import uuid
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from groq import Groq

from firebase_admin import auth as firebase_auth

from config import NON_EMPTY_GROQ_KEYS
from firebase_client import load_interview_questions, save_user_response, save_interview_score
from rag import generate_interviewer_reply, generate_final_score
from tts import tts_text_to_base64_wav

# ---- Groq client for STT (Whisper) ----

if not NON_EMPTY_GROQ_KEYS:
    raise RuntimeError("No Groq API keys configured for STT (Whisper).")

stt_api_key = NON_EMPTY_GROQ_KEYS[-1]  # choose one key for STT
client_stt = Groq(api_key=stt_api_key)

# ---- FastAPI app ----

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/groqspeaks")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 1) REQUIRE Firebase login: get idToken from query params
    id_token = websocket.query_params.get("idToken")
    if not id_token:
        await websocket.close(code=4401)
        print("[WS] Missing idToken, closing connection")
        return

    # 2) Verify token with Firebase Admin SDK
    try:
        decoded = firebase_auth.verify_id_token(id_token)
        user_id = decoded["uid"]
        print(f"[WS] Authenticated user: {user_id}")
    except Exception as e:
        await websocket.close(code=4401)
        print(f"[WS] Invalid idToken: {e}")
        return

    # 3) Interview ID (can be provided by frontend or auto-generated)
    interview_id = websocket.query_params.get("interviewId") or str(uuid.uuid4())

    # Get current attempt number for this interview
    from firebase_client import db
    try:
        interview_doc = db.collection("interviews").document(interview_id).get()
        current_attempt = 1
        if interview_doc.exists:
            data = interview_doc.to_dict() or {}
            current_attempt = (data.get("attempt_count", 0) or 0) + 1
    except Exception:
        current_attempt = 1
    print(f"[Interview] Starting attempt #{current_attempt} for interview {interview_id}")

    # Load interview questions
    questions = load_interview_questions(interview_id)
    if not questions:
        await websocket.send_text("No interview questions configured.")
        await websocket.close()
        return

    # Build interview-specific vectorstore for RAG context
    from rag import build_vectorstore_from_firestore
    try:
        print(f"[Interview] Building RAG vectorstore for interview {interview_id}")
        interview_vectorstore = build_vectorstore_from_firestore(interview_id)
        if interview_vectorstore:
            interview_retriever = interview_vectorstore.as_retriever(search_kwargs={"k": 4})
            print(f"[Interview] RAG vectorstore ready for {interview_id}")
        else:
            interview_retriever = None
            print(f"[Interview] No RAG context found for {interview_id}")
    except Exception as e:
        print(f"[Interview] Error building vectorstore: {e}")
        interview_retriever = None

    current_q_index = 0

    # Chat history for LLM context
    chat_hist = [
        {
            "role": "system",
            "content": "Interviewer AI session started.",
        }
    ]

    # ---- Send initial greeting + first question ----
    first_question = questions[current_q_index]
    intro_text = (
        "Hello, I am your AI interviewer. "
        "We will proceed through the questions one by one.\n\n"
        f"Your first question is: {first_question}"
    )
    intro_audio_base64 = tts_text_to_base64_wav(intro_text)

    initial_payload = {
        "text": intro_text,
        "audio_base64": intro_audio_base64,
        "interviewId": interview_id,
        "questionIndex": current_q_index,
    }
    await websocket.send_text(json.dumps(initial_payload))
    chat_hist.append({"role": "assistant", "content": intro_text})

    metrics_buffer = {}
    metrics_by_question = {}

    try:
        while True:
            try:
                message = await websocket.receive()
            except Exception as e:
                print(f"[WS] Error receiving message: {e}")
                break

            if message.get("type") == "websocket.disconnect":
                break

            text_data = message.get("text")
            bytes_data = message.get("bytes")

            if text_data is not None:
                try:
                    payload = json.loads(text_data)
                except Exception:
                    continue

                if payload.get("type") == "metrics":
                    q_idx = int(payload.get("questionIndex", current_q_index))
                    received_metrics = payload.get("metrics", {})
                    metrics_buffer[q_idx] = received_metrics
                    print(f"[Metrics] Received metrics for Q{q_idx}: {received_metrics}")
                continue

            if bytes_data is None:
                continue

            audio_bytes = bytes_data

            # Persist to a temp file for the STT client (some clients accept file path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name

            try:
                with open(temp_audio_path, "rb") as f:
                    t0 = time.monotonic()
                    resp = client_stt.audio.transcriptions.create(
                        file=(temp_audio_path, f.read()),
                        model="whisper-large-v3-turbo",
                        response_format="verbose_json",
                    )
                    t1 = time.monotonic()
                    print(f"[STT] Transcription took {t1 - t0:.2f}s")
                user_text = getattr(resp, "text", "") or ""
            except Exception as e:
                print(f"[STT] Transcription error: {e}")
                user_text = ""
            finally:
                try:
                    os.remove(temp_audio_path)
                except Exception:
                    pass

            # If transcript empty, inform client and continue waiting for next audio
            if not user_text:
                try:
                    await websocket.send_text(json.dumps({"text": "Transcription empty or failed."}))
                except Exception:
                    pass
                continue

            # Prepare metrics (frontend-sent only; filler counts removed)
            frontend_metrics = metrics_buffer.pop(current_q_index, {})
            merged_metrics = {**frontend_metrics}
            metrics_by_question[current_q_index] = merged_metrics
            print(f"[Metrics] Using metrics for Q{current_q_index}: {merged_metrics}")

            # Append to chat history and persist
            chat_hist.append({"role": "user", "content": user_text})
            current_question_text = questions[current_q_index]

            # Send transcript + metrics back to client
            try:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "transcript",
                            "interviewId": interview_id,
                            "questionIndex": current_q_index,
                            "transcript": user_text,
                            "metrics": merged_metrics,
                        }
                    )
                )
            except Exception:
                pass

            save_user_response(
                interview_id=interview_id,
                user_id=user_id,
                question_index=current_q_index,
                question_text=current_question_text,
                answer_text=user_text,
                metrics=merged_metrics,
                attempt_number=current_attempt,
            )

            # Determine next question (if any)
            if current_q_index < len(questions) - 1:
                next_question = questions[current_q_index + 1]
            else:
                next_question = None

            try:
                t0_llm = time.monotonic()
                res = generate_interviewer_reply(
                    user_answer=user_text,
                    chat_hist=chat_hist,
                    current_question=current_question_text,
                    next_question=next_question,
                    metrics=merged_metrics,
                    retriever=interview_retriever,
                )
                t1_llm = time.monotonic()
                print(f"[WS] LLM produced response (took {t1_llm - t0_llm:.2f}s): {res[:200]}")
                chat_hist.append({"role": "assistant", "content": res})

                t0_tts = time.monotonic()
                audio_base64 = tts_text_to_base64_wav(res)
                t1_tts = time.monotonic()
                print(f"[TTS] Synthesis took {t1_tts - t0_tts:.2f}s")
                
                response_payload = {
                    "type": "llm_response",
                    "text": res,
                    "interviewId": interview_id,
                    "questionIndex": current_q_index,
                }
                
                if audio_base64 is None:
                    print("[TTS] All TTS models failed, sending text-only response")
                    response_payload["audio_base64"] = None
                else:
                    response_payload["audio_base64"] = audio_base64
                
                await websocket.send_text(json.dumps(response_payload))

                # If that was the last question, generate and store final score
                if next_question is None:
                    try:
                        print(f"[Interview] Generating final score for interview {interview_id}")
                        score_result = generate_final_score(chat_hist, questions, metrics_by_question)
                        score = score_result["score"]
                        justification = score_result["justification"]

                        attempt_count = save_interview_score(
                            interview_id=interview_id,
                            user_id=user_id,
                            score=score,
                            justification=justification,
                        )

                        final_payload = {
                            "type": "final_score",
                            "score": score,
                            "justification": justification,
                            "interviewId": interview_id,
                            "attempt_count": attempt_count,
                        }
                        await websocket.send_text(json.dumps(final_payload))
                    except Exception as e:
                        print(f"[Interview] Error generating/saving score: {e}")

                    print(f"[Interview] Completed interview {interview_id} for user {user_id}")
                    await websocket.close()
                    break

                # Move to next question for the next user reply
                current_q_index += 1

            except Exception as e:
                err_msg = f"Error: {str(e)}"
                try:
                    await websocket.send_text(json.dumps({"text": err_msg}))
                except Exception:
                    pass

    except WebSocketDisconnect:
        chat_hist = []
        print("WebSocket got disconnected")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="127.0.0.1", port=port)
