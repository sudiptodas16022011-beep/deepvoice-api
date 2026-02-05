from fastapi import FastAPI, Header, HTTPException, Body
from pydantic import BaseModel
from app.model import detector
from app.utils import decode_base64_audio, generate_explanation

app = FastAPI(
    title="DeepVoice Guard API",
    description="SOTA AI Voice Detection for Tamil, English, Hindi, Malayalam, Telugu",
    version="1.0.0"
)

# --- CONFIGURATION ---
# STRICTLY MATCHING PDF REQUIREMENTS
API_KEY_SECRET = "sk_test_123456789"  # Change this if you want, but keep it simple for testing

# --- REQUEST/RESPONSE MODELS ---
class AudioRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"message": "DeepVoice Guard is Active. Use POST /api/voice-detection"}

@app.post("/api/voice-detection")
async def detect_voice(
    payload: AudioRequest,
    x_api_key: str = Header(None, alias="x-api-key") # Handle header strictly
):
    # 1. AUTHENTICATION (PDF Section 5)
    if x_api_key != API_KEY_SECRET:
        return {
            "status": "error",
            "message": "Invalid API key or malformed request"
        }

    # 2. VALIDATION (PDF Section 2)
    valid_languages = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
    if payload.language not in valid_languages:
        return {
            "status": "error",
            "message": "Unsupported language."
        }
    
    if payload.audioFormat.lower() != "mp3":
        return {
            "status": "error",
            "message": "Only mp3 format is supported."
        }

    # 3. PROCESSING
    audio_array = decode_base64_audio(payload.audioBase64)
    if audio_array is None or len(audio_array) == 0:
        return {
            "status": "error",
            "message": "Invalid Base64 audio content."
        }

    # 4. AI INFERENCE
    classification, confidence = detector.predict(audio_array)
    
    # 5. EXPLAINABILITY (The Winner Factor)
    explanation_text = generate_explanation(confidence, classification, audio_array)

    # 6. JSON RESPONSE (PDF Section 8)
    return {
        "status": "success",
        "language": payload.language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation_text
    }