import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Parameters
SAMPLE_RATE = 16000
NUM_CLASSES = 10  # Update based on number of speakers
INPUT_SHAPE = (128, 128, 1)  # Mel spectrogram dimensions

# Load and preprocess audio file
def extract_features(file_path, sr=SAMPLE_RATE):
    y, sr = librosa.load(file_path, sr=sr)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_resized = np.resize(mel_spec_db, INPUT_SHAPE[:2])
    return np.expand_dims(mel_spec_resized, axis=-1)

# Build CNN Model
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load dataset
DATASET_PATH = 'data/'  # Update with your dataset path
X, y = [], []
labels = []

for speaker in os.listdir(DATASET_PATH):
    speaker_path = os.path.join(DATASET_PATH, speaker)
    if os.path.isdir(speaker_path):
        for file in os.listdir(speaker_path):
            if file.endswith('.wav'):
                file_path = os.path.join(speaker_path, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(speaker)
                labels.append(speaker)

X = np.array(X)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train Model
model = build_model(INPUT_SHAPE, NUM_CLASSES)
model.fit(X, y, epochs=30, batch_size=16, validation_split=0.2)

# Save Model
model.save('speaker_model.h5')

# Prediction Function
def predict_speaker(file_path, model_path='speaker_model.h5'):
    model = tf.keras.models.load_model(model_path)
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction)
    return label_encoder.inverse_transform([predicted_label])[0]

# Example Prediction
file_path = 'test1.wav'  # Replace with your test audio file
predicted_speaker = predict_speaker(file_path)
print(f'Predicted Speaker: {predicted_speaker}')
