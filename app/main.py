from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
import base64
import os
import uuid

# Assuming this utility exists in your app folder
from app.utils import detect_voice_authenticity

app = FastAPI(title="DeepVoice Guard API")

# --- Security Configuration ---
API_KEY_NAME = "x-api-key"
VALID_API_KEY = "sk_test_123456789" 

class AudioRequest(BaseModel):
    language: str
    # Aliases ensure the API accepts 'audioFormat' and 'audioBase64' from the JSON
    audio_format: str = Field(..., alias="audioFormat")
    audio_base64: str = Field(..., alias="audioBase64")

    class Config:
        # This allows the model to be populated by the field name OR the alias
        populate_by_name = True

async def verify_api_key(x_api_key: str = Header(None, alias="x-api-key")):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return x_api_key

@app.get("/")
def read_root():
    return {"status": "online", "message": "DeepVoice Detection API is active"}

@app.post("/api/voice-detection")
async def voice_detection(request: AudioRequest, api_key: str = Depends(verify_api_key)):
    # Generate a unique filename to prevent file collisions during concurrent requests 
    temp_filename = f"temp_{uuid.uuid4()}.mp3"
    
    try:
        # Accessing the attribute via the Python name (audio_base64) 
        b64_str = request.audio_base64 
        
        # Strip metadata prefix if present (e.g., "data:audio/mp3;base64,") 
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]

        # 2. Decode and save
        audio_data = base64.b64decode(b64_str)
        with open(temp_filename, "wb") as f:
            f.write(audio_data)

        # 3. Call detection engine
        # classification, score, and explanation are returned as standard Python types
        classification, score, explanation = detect_voice_authenticity(temp_filename)

        # 4. Return response
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
        # Log the error for server-side debugging and return a 400 error to the client 
        raise HTTPException(status_code=400, detail=f"Processing error: {str(e)}")

    finally:
        # Cleanup: Ensure the temporary file is deleted regardless of success or failure 
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    import uvicorn
    # Port 10000 is often required for Render deployments 
    uvicorn.run(app, host="0.0.0.0", port=10000)