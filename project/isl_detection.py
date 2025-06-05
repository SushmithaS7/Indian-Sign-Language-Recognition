import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
from gtts import gTTS
import os
import time

# Load the saved model from file
model = keras.models.load_model("model.h5")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet += list(string.ascii_uppercase)

# # Function to convert text to speech
# def text_to_speech(text):
#     tts = gTTS(text=text, lang='en')
#     tts.save("output.mp3")
#     os.system("start output.mp3")  # For Windows, use "start". For macOS, use "afplay", and for Linux, use "mpg123".
#     time.sleep(2)  # Optional pause after playing audio

# Functions to process landmarks
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    # Convert to relative coordinates using the first landmark as the base
    base_x, base_y = temp_landmark_list[0]
    for index, point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    # Flatten the list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    # Normalize
    max_value = max(list(map(abs, temp_landmark_list))) or 1  # Avoid division by zero
    temp_landmark_list = [x / max_value for x in temp_landmark_list]
    return temp_landmark_list

# Variables to track gesture stability
last_label = None       # The previous gesture label detected
gesture_start_time = 0  # When the current gesture was first seen
tts_called = False      # Flag to prevent repeated TTS calls for the same gesture
stable_duration = 1.5   # Seconds that the gesture must be held stable before TTS is called

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        # Flip the image horizontally for a selfie-view display.
        # image = cv2.flip(image, 1)
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the image to RGB for processing.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Convert back to BGR for displaying.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        debug_image = image.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Draw the landmarks on the image
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Prepare data for prediction
                df = pd.DataFrame(pre_processed_landmark_list).transpose()
                predictions = model.predict(df, verbose=0)
                predicted_classes = np.argmax(predictions, axis=1)
                label = alphabet[predicted_classes[0]]

                # Display label on screen
                cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                # print("Predicted Label:", label)
                # print("------------------------")

                # Check if the gesture is stable long enough before calling TTS:
                current_time = time.time()
                if label != last_label:
                    # A new gesture is detected; reset the timer and flag.
                    last_label = label
                    gesture_start_time = current_time
                    tts_called = False
                else:
                    # Same gesture as before; check if it has been held long enough.
                    if not tts_called and (current_time - gesture_start_time >= stable_duration):
                        print(label)
                        # text_to_speech(label)
                        tts_called = True

        # Show the output image
        cv2.imshow('Indian Sign Language Detector', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()