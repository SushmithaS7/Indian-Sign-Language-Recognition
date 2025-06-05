import os
import cv2
import time
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array

# ----- Define labels (same order as training) -----
actions = ["Alright","Good afternoon","Good evening","Good Morning","Good night","Hello","How are you","Pleased", "Thank you"]  # Modify if needed

# Load VGG16 model (no top layer)
base_model = VGG16(weights='imagenet', include_top=False)

# Load trained LSTM model
model = load_model('lstm-model/final_trained.keras', compile=False)

# Function to extract features from frame
def extract_features(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = tf.keras.applications.vgg16.preprocess_input(frame)
    features = base_model.predict(frame, verbose=0)
    return features.flatten()

# Create input-video folder if it doesn't exist
os.makedirs("input-video", exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Recording video for 5 seconds...")

out = cv2.VideoWriter('input-video/input.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height))

frame_features = []
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    out.write(frame)

    # Show countdown timer
    elapsed = int(time.time() - start_time)
    cv2.putText(frame, f"Recording... {5 - elapsed}s", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('ISL Word Recognition - Recording', frame)

    # Extract and store features
    features = extract_features(frame)
    frame_features.append(features)

    if time.time() - start_time > 5:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Video recording complete.")

# Process frames for prediction
max_frames = 30

# Pad or trim sequence
if len(frame_features) < max_frames:
    pad_length = max_frames - len(frame_features)
    padding = np.zeros((pad_length, len(frame_features[0])))
    frame_features = np.vstack([frame_features, padding])
else:
    frame_features = frame_features[:max_frames]

sequence = np.array(frame_features).reshape(1, max_frames, -1)

# Predict
prediction = model.predict(sequence, verbose=0)
predicted_index = np.argmax(prediction)
predicted_word = actions[predicted_index]

print(f"Predicted Word: {predicted_word}")
