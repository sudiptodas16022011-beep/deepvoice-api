import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# --- CONFIGURATION ---
# We use a Wav2Vec2 model fine-tuned for spoof detection.
# This model works on audio texture, so it applies to ALL languages (Tamil, Hindi, etc.)
MODEL_ID = "Hemgg/Deepfake-audio-detection" 

class DeepFakeDetector:
    def __init__(self):
        print(f"Loading AI Model: {MODEL_ID}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
            self.model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(self.device)
            print("✅ Model Loaded Successfully on", self.device)
        except Exception as e:
            print(f"❌ Critical Error Loading Model: {e}")
            raise e

    def predict(self, audio_array):
        # Prepare input
        inputs = self.feature_extractor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get the predicted label (0 or 1). 
        # Note: Check model config. Usually 1 = Fake/Deepfake, 0 = Real.
        # For 'Hemgg/Deepfake-audio-detection', label 1 is typically Fake.
        pred_id = torch.argmax(probs, dim=-1).item()
        score = probs[0][pred_id].item()
        
        # Map to required output strings
        # Adjust logic based on specific model labels if needed after first test
        if pred_id == 1: 
            return "AI_GENERATED", score
        else:
            return "HUMAN", score

# Initialize Singleton
detector = DeepFakeDetector()