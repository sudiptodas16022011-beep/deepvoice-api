from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import base64
import os
import uuid

# Corrected Import: Matches the new function name in utils.py
from app.utils import detect_voice_authenticity

app = FastAPI(title="DeepVoice Guard API")

# --- Security Configuration ---
API_KEY_NAME = "x-api-key"
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
    # Generate a unique temporary filename for this specific request
    temp_filename = f"temp_{uuid.uuid4()}.mp3"
    
    try:
        # 1. Sanitize the Base64 string
        b64_str = request.audio_base64
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]

        # 2. Decode and save to a temporary file for analysis
        audio_data = base64.b64decode(b64_str)
        with open(temp_filename, "wb") as f:
            f.write(audio_data)

        # 3. Call the updated, high-accuracy detection engine
        # This replaces the old decode_base64_audio function
        classification, score, explanation = detect_voice_authenticity(temp_filename)

        # 4. Return the standard JSON response
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
        # Catch errors and provide detail to help with debugging
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")

    finally:
        # 5. Clean up: Delete the temporary file to keep the server storage clean
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    # Render uses port 10000 by default
    uvicorn.run(app, host="0.0.0.0", port=10000)