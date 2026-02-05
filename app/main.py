from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import base64
import os
import uuid

# Import only the new accurate function from utils
from app.utils import detect_voice_authenticity

app = FastAPI(title="DeepVoice Guard API")

# --- Security Configuration ---
API_KEY_NAME = "x-api-key"
# This must match what you submitted to the GUVI portal
VALID_API_KEY = "sk_test_123456789" 

class AudioRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str

async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return x_api_key

@app.get("/")
def read_root():
    return {"status": "online", "message": "DeepVoice Detection API is active"}

@app.post("/api/voice-detection")
async def voice_detection(request: AudioRequest, api_key: str = Depends(verify_api_key)):
    temp_filename = f"temp_{uuid.uuid4()}.mp3"
    
    try:
        # 1. Clean the Base64 string (remove data headers if present)
        b64_str = request.audio_base64
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]

        # 2. Decode and save to a temporary file
        audio_data = base64.b64decode(b64_str)
        with open(temp_filename, "wb") as f:
            f.write(audio_data)

        # 3. Call the highly accurate detection engine from utils.py
        classification, score, explanation = detect_voice_authenticity(temp_filename)

        # 4. Return the official response format
        return {
            "classification": classification,
            "confidence_score": score,
            "explanation": explanation,
            "metadata": {
                "language": request.language,
                "format": request.audio_format
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing failed: {str(e)}")

    finally:
        # 5. Always clean up the temporary file to save Render disk space
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)