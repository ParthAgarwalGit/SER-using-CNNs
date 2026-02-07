import os
import sys
import numpy as np
import librosa
import tensorflow as tf

# --- Configuration ---
# This matches the filename used in your notebook's model.save()
MODEL_PATH = 'final_emotion_model.h5'

# Audio settings must match training EXACTLY
SR = 22050
DURATION = 3
SAMPLES_PER_TRACK = SR * DURATION

# Emotion Labels (RAVDESS mapping)
# 0=Neutral, 1=Calm, 2=Happy, 3=Sad, 4=Angry, 5=Fearful, 6=Disgust, 7=Surprised
EMOTION_LABELS = {
    0: 'Neutral', 
    1: 'Calm', 
    2: 'Happy', 
    3: 'Sad', 
    4: 'Angry', 
    5: 'Fearful', 
    6: 'Disgust', 
    7: 'Surprised'
}

def preprocess_audio(file_path):
    """
    Loads an audio file and processes it into a (1, 128, 130, 1) Log-Mel Spectrogram.
    """
    try:
        # 1. Load Audio
        y, sr = librosa.load(file_path, sr=SR)
        
        # 2. Trim Silence (Top dB 20 is standard for RAVDESS)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # 3. Pad or Truncate to Fixed Length (3 Seconds)
        if len(y_trimmed) > SAMPLES_PER_TRACK:
            # If too long, crop from the beginning
            y_fixed = y_trimmed[:SAMPLES_PER_TRACK]
        else:
            # If too short, center pad with zeros
            padding = SAMPLES_PER_TRACK - len(y_trimmed)
            offset = padding // 2
            y_fixed = np.pad(y_trimmed, (offset, padding - offset), 'constant')
            
        # 4. Generate Log-Mel Spectrogram
        # n_mels=128 (height), hop_length=512 (default) -> results in width ~130
        spectrogram = librosa.feature.melspectrogram(y=y_fixed, sr=sr, n_mels=128)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        
        # 5. Reshape for CNN Input
        # Model expects: (Batch, Height, Width, Channels) -> (1, 128, 130, 1)
        input_data = spectrogram_db[np.newaxis, ..., np.newaxis]
        
        return input_data

    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def predict_emotion(file_path):
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please place it in this directory.")
        return

    # Load Model
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    model = tf.keras.models.load_model(MODEL_PATH)

    # Process Audio
    input_data = preprocess_audio(file_path)
    
    if input_data is not None:
        # Make Prediction
        predictions = model.predict(input_data, verbose=0)
        
        # Get the highest probability class
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        emotion = EMOTION_LABELS.get(predicted_index, "Unknown")
        
        # Print Output
        print("-" * 30)
        print(f"File:       {os.path.basename(file_path)}")
        print(f"Prediction: {emotion.upper()}")
        print(f"Confidence: {confidence:.2f}%")
        print("-" * 30)

if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_file.wav>")
    else:
        audio_file_path = sys.argv[1]
        if os.path.exists(audio_file_path):
            predict_emotion(audio_file_path)
        else:
            print(f"File not found: {audio_file_path}")