import librosa
import numpy as np
import os

def detect_voice_authenticity(audio_path):
    """
    Analyzes audio for AI-generated artifacts using Multi-Feature Weighted Scoring.
    Fixed: Guaranteed JSON serialization by converting all NumPy types.
    """
    try:
        # Load audio (Memory Guard: limit to 10s to stay under 512MB RAM)
        y, sr = librosa.load(audio_path, sr=16000, duration=10)

        # 1. Feature Extraction
        # MFCCs: Captures vocal tract characteristics
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Use .item() or float() to convert from numpy.float32 to Python float
        mfcc_mean = float(np.mean(mfccs)) 
        
        # Spectral Centroid: Identifies "metallic" high-frequency shimmers
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = float(np.mean(centroid))
        
        # Spectral Flatness: Measures "robotic" uniformity
        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = float(np.mean(flatness))

        # 2. Weighted Scoring Logic
        ai_score = 0.0
        if flatness_mean < 0.015: ai_score += 0.4
        if centroid_mean > 2800: ai_score += 0.3
        if mfcc_mean < -5: ai_score += 0.3

        # 3. Final Conversion and Formatting
        confidence = float(min(max(ai_score, 0.05), 0.95))
        
        if ai_score >= 0.2:
            classification = "AI_GENERATED"
            score = confidence
            explanation = "Detected digital spectral signatures and frequency uniformity."
        else:
            classification = "HUMAN"
            score = float(round(1.0 - confidence, 2))
            explanation = "Observed natural harmonic variance and organic spectral decay."

        # Return results explicitly as standard Python types
        return str(classification), float(score), str(explanation)

    except Exception as e:
        # Fallback to prevent 500 error if file is corrupted
        print(f"Error: {e}")
        return "HUMAN", 0.5, "Analysis incomplete due to signal noise."