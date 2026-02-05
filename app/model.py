import numpy as np
import librosa

class DeepFakeDetector:
    def __init__(self):
        print("✅ Ultra-Lite Signal Engine Loaded!")

    def predict(self, audio_array):
        # Analyze spectral features (Human voices have natural micro-variations)
        # AI voices often have unnatural uniformity in high-frequency bands
        stft = np.abs(librosa.stft(audio_array))
        flatness = np.mean(librosa.feature.spectral_flatness(S=stft))
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_array))
        
        # Heuristic for synthesis detection
        # Higher flatness and specific ZCR ranges are common in digital synthesis
        if flatness > 0.01 or zcr > 0.15:
            classification = "AI_GENERATED"
            score = round(0.85 + (flatness * 2), 2)
        else:
            classification = "HUMAN"
            score = round(0.92 - flatness, 2)
            
        return classification, min(score, 0.99)

# Essential: This object must be created for main.py to find it
detector = DeepFakeDetector()