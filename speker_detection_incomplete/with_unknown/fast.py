import numpy as np
import librosa
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load pre-trained model and preprocessing components
MODEL_PATH = "classifier.pkl"
SCALER_PATH = "scaler.pkl"
PCA_PATH = "pca.pkl"

# Load the model, scaler, and PCA transformer
try:
    best_classifier = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    logging.info("Model, scaler, and PCA loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load required files: {e}")
    raise

def extract_features(audio_file):
    """
    Extract features from the given audio file.
    
    Parameters:
    - audio_file (str): Path to the audio file.

    Returns:
    - np.ndarray: Extracted feature vector, or None if extraction fails.
    """
    try:
        y, sr = librosa.load(audio_file)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=42)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        chroma_features = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

        features = np.hstack([
            np.mean(mfccs, axis=1), np.std(mfccs, axis=1),
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.mean(zero_crossing_rate), np.std(zero_crossing_rate),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            np.mean(spectral_contrast, axis=1), np.std(spectral_contrast, axis=1),
            np.mean(chroma_features, axis=1), np.std(chroma_features, axis=1),
            np.mean(rms), np.std(rms),
            np.mean(spectral_rolloff), np.std(spectral_rolloff),
            np.mean(mel_spectrogram, axis=1), np.std(mel_spectrogram, axis=1)
        ])
        return features
    except Exception as e:
        logging.error(f"Error extracting features from {audio_file}: {e}")
        return None

def predict_speaker_from_audio(audio_file_path):
    """
    Predict the speaker from an audio file.

    Parameters:
    - audio_file_path (str): Path to the audio file.

    Returns:
    - str: Predicted speaker's name, or None if prediction fails.
    """
    try:
        # Step 1: Extract features from the audio file
        features = extract_features(audio_file_path)
        if features is None:
            logging.error("No features could be extracted from the audio file.")
            return None
            
        logging.info(f"Extracted feature shape: {features.shape}")
        
        scaler = joblib.load('scaler.pkl')
        pca = joblib.load('pca.pkl')

        # Verify the number of features the scaler expects
        print(scaler.mean_.shape)  # Should match the number of features
        print(pca.components_.shape)  # Should match the number of features



        # Step 2: Preprocess features
        features_scaled = scaler.transform([features])  # Normalize using the fitted scaler
        features_pca = pca.transform(features_scaled)   # Apply PCA transformation using the fitted PCA model

        # Step 3: Predict the speaker using the best classifier
        predicted_label = best_classifier.predict(features_pca)[0]

        return predicted_label
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return None

# Example usage within this script (you can comment this out when importing the function):
if __name__ == "__main__":
    test_audio_file = "test1.wav"  # Replace with your audio file path
    speaker_name = predict_speaker_from_audio(test_audio_file)
    if speaker_name:
        print(f"Predicted Speaker: {speaker_name}")
    else:
        print("Failed to predict the speaker.")
    test_audio_file = "test2.wav"  # Replace with your audio file path
    speaker_name = predict_speaker_from_audio(test_audio_file)
    if speaker_name:
        print(f"Predicted Speaker: {speaker_name}")
    else:
        print("Failed to predict the speaker.")
    test_audio_file = "test3.wav"  # Replace with your audio file path
    speaker_name = predict_speaker_from_audio(test_audio_file)
    if speaker_name:
        print(f"Predicted Speaker: {speaker_name}")
    else:
        print("Failed to predict the speaker.")
    test_audio_file = "p1.wav"  # Replace with your audio file path
    speaker_name = predict_speaker_from_audio(test_audio_file)
    if speaker_name:
        print(f"Predicted Speaker: {speaker_name}")
    else:
        print("Failed to predict the speaker.")
    test_audio_file = "p2.wav"  # Replace with your audio file path
    speaker_name = predict_speaker_from_audio(test_audio_file)
    if speaker_name:
        print(f"Predicted Speaker: {speaker_name}")
    else:
        print("Failed to predict the speaker.")

