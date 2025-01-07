import pickle
import cv2
import mediapipe as mp
import numpy as np
import boto3
import os
from collections import Counter

# Text-to-Speech function using Amazon Polly
def text_to_speech(text, output_file="output.mp3"):
    polly = boto3.client('polly')
    try:
        response = polly.synthesize_speech(
            Text=text,
            OutputFormat='mp3',
            VoiceId='Joanna'  # You can choose other voices like 'Matthew', 'Ivy', etc.
        )
        with open(output_file, 'wb') as file:
            file.write(response['AudioStream'].read())
        print(f"Speech saved to {output_file}")
    except Exception as e:
        print(f"Error with text-to-speech: {e}")

# Load the trained model, scaler, and label mappings
with open('./models/model.p', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']
int_to_label = model_dict['int_to_label']
scaler = model_dict['scaler']

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Start capturing video from webcam
cap = cv2.VideoCapture(1)  # Change to 1 if the default camera is not used

print("Press SPACE to convert text to speech. Press ESC to exit.")
sentence = ""  # To store the sequence of recognized letters/numbers
recent_predictions = []  # For smoothing predictions
max_window_size = 5  # Sliding window size for majority voting

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error accessing the webcam.")
        break

    # Flip the frame horizontally for a mirror-like effect
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    # Convert to RGB for Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_, y_ = [], []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract and normalize landmarks
            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
                data_aux.append(lm.x)
                data_aux.append(lm.y)

            # Normalize landmarks
            min_x, max_x = min(x_), max(x_)
            min_y, max_y = min(y_), max(y_)
            for i in range(len(data_aux) // 2):
                data_aux[2 * i] = (data_aux[2 * i] - min_x) / (max_x - min_x) if max_x != min_x else 0
                data_aux[2 * i + 1] = (data_aux[2 * i + 1] - min_y) / (max_y - min_y) if max_y != min_y else 0

        # Standardize features
        data_aux = scaler.transform([np.asarray(data_aux)])

        # Predict the character
        try:
            prediction = model.predict(data_aux)
            predicted_character = int_to_label[int(prediction[0])]

            # Smooth predictions using a sliding window
            if len(recent_predictions) >= max_window_size:
                recent_predictions.pop(0)
            recent_predictions.append(predicted_character)
            most_common_prediction = Counter(recent_predictions).most_common(1)[0][0]

            # Display prediction
            cv2.putText(frame, f"Prediction: {most_common_prediction}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            sentence += most_common_prediction
        except Exception as e:
            print(f"Prediction error: {e}")

    # Display the sentence
    cv2.putText(frame, f"Sentence: {sentence}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Sign Language Translator', frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit
        break
    elif key == 32:  # SPACE key to convert sentence to speech
        if sentence:
            output_file = os.path.join('audio', 'output.mp3')
            text_to_speech(sentence, output_file)
            sentence = ""  # Clear the sentence after speech conversion

# Release resources
cap.release()
cv2.destroyAllWindows()
