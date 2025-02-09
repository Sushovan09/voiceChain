import os
import numpy as np
import librosa
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set logging
logging.basicConfig(level=logging.INFO)

# Paths for data and saved models
DATASET_PATH = "data"  # Directory with speaker folders and WAV files
SCALER_PATH = "scaler.pkl"
PCA_PATH = "pca.pkl"
MODEL_PATH = "classifier.pkl"

# Parameters
UNKNOWN_THRESHOLD = 0.5  # Confidence threshold to classify as "unknown"
PCA_COMPONENTS = 50  # Number of PCA components


# Helper functions
def extract_features(audio_file):
    """
    Extract MFCC features from an audio file.
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


def load_data(dataset_path):
    """
    Load dataset, extract features, and create labels.
    """
    features = []
    labels = []
    speaker_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    
    for speaker in speaker_folders:
        speaker_path = os.path.join(dataset_path, speaker)
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                file_path = os.path.join(speaker_path, file)
                mfcc_features = extract_features(file_path)
                if mfcc_features is not None:
                    features.append(mfcc_features)
                    labels.append(speaker)
    
    return np.array(features), np.array(labels)


# Step 1: Load Data
logging.info("Loading data...")
features, labels = load_data(DATASET_PATH)

# Step 2: Train-Test Split
logging.info("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.02, random_state=42, stratify=labels)

# Step 3: Preprocess Data
logging.info("Scaling features...")
scaler = StandardScaler()

# Fit the scaler on all collected features (X with 388 dimensions)
scaler.fit(X_train)  # X_train is the feature matrix (number of samples x 388 features)

# Normalize features using the fitted scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_test_scaled.shape)

logging.info("Applying PCA...")
# Dynamically adjust n_components based on the dataset
n_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
pca = PCA(n_components=n_components)

# Apply PCA transformation to the scaled features
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Step 4: Train Model
logging.info("Training classifier...")
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_pca, y_train)

# Step 5: Save models for future use
logging.info("Saving models for future use...")
joblib.dump(scaler, SCALER_PATH)
joblib.dump(pca, PCA_PATH)
joblib.dump(clf, MODEL_PATH)

logging.info("Models saved successfully!")

# You can later load these models and use them for prediction

