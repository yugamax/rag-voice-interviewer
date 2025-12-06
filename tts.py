import base64
from typing import Optional, List, Any

try:
    from groq import Groq
except Exception as e:
    Groq = None
    print(f"[TTS] Warning: could not import Groq client: {e}")

from config import NON_EMPTY_GROQ_KEYS


def _strip_data_url(s: str) -> str:
    # remove leading data:...;base64, if present
    if not isinstance(s, str):
        return s
    if ";base64," in s:
        return s.split(";base64,", 1)[1]
    if s.startswith("data:") and "," in s:
        return s.split(",", 1)[1]
    return s


def _extract_bytes(obj: Any) -> bytes:
    """Attempt to extract raw bytes from many possible response shapes.

    This function is synchronous and defensive: it will try several common
    access patterns and return an empty bytes object on failure.
    """
    try:
        # direct bytes
        if isinstance(obj, (bytes, bytearray)):
            return bytes(obj)

        # if it's a string, assume it's base64 or raw text
        if isinstance(obj, str):
            s = _strip_data_url(obj).strip()
            try:
                return base64.b64decode(s)
            except Exception:
                return s.encode("utf-8")

        # dict-like objects: check common keys
        if isinstance(obj, dict):
            for key in ("audio", "data", "result", "audio_base64", "base64", "content"):
                if key in obj:
                    val = obj[key]
                    b = _extract_bytes(val)
                    if b:
                        return b
            # scan for long base64-like strings
            for v in obj.values():
                if isinstance(v, str) and len(v) > 100:
                    try:
                        return base64.b64decode(_strip_data_url(v))
                    except Exception:
                        continue

        # objects with common attributes
        for attr in ("content", "raw", "data", "audio", "body"):
            if hasattr(obj, attr):
                try:
                    val = getattr(obj, attr)
                    b = _extract_bytes(val)
                    if b:
                        return b
                except Exception:
                    pass

        # file-like objects with read() (avoid treating dicts as file-like)
        if not isinstance(obj, dict) and hasattr(obj, "read") and callable(getattr(obj, "read")):
            try:
                chunk = obj.read()
                return _extract_bytes(chunk)
            except Exception:
                pass

        # iterables of bytes/bytearray
        try:
            if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray, dict)):
                # try to join bytes if possible
                parts = []
                for item in obj:
                    if isinstance(item, (bytes, bytearray)):
                        parts.append(bytes(item))
                    else:
                        # try extracting recursively
                        b = _extract_bytes(item)
                        if b:
                            parts.append(b)
                if parts:
                    return b"".join(parts)
        except Exception:
            pass

    except Exception:
        pass

    return b""


# Instantiate Groq clients for TTS (filter out empty entries)
clients: List[Any] = []
if Groq is not None:
    for k in NON_EMPTY_GROQ_KEYS:
        if k:
            try:
                clients.append(Groq(api_key=k))
            except Exception as e:
                print(f"[TTS] Failed to initialize Groq client for a key: {e}")

print(f"[TTS] Initialized {len(clients)} TTS client(s).")


def tts_text_to_base64_wav(text: str) -> Optional[str]:
    """
    Use Groq TTS clients to convert text → base64 wav.
    Tries each configured API key in order. Returns None if all fail.
    """
    if not clients:
        print("[TTS] No Groq API keys configured for TTS.")
        return None

    last_error = None

    for i, client in enumerate(clients, start=1):
        try:
            print(f"[TTS] Trying client {i}...")
            tts_response = client.audio.speech.create(
                model="playai-tts",
                voice="Nia-PlayAI",
                response_format="wav",
                input=text,
            )

            audio_data = _extract_bytes(tts_response)

            if not audio_data:
                raise RuntimeError(f"TTS returned empty audio payload (type={type(tts_response)})")

            enc_aud = base64.b64encode(audio_data).decode("utf-8")
            print(f"[TTS] Client {i} succeeded (response type: {type(tts_response)})")
            return enc_aud

        except Exception as e:
            last_error = e
            print(f"[TTS] Client {i} failed: {e}")

    print(f"[TTS] All TTS clients failed. Last error: {last_error}")
    return None
