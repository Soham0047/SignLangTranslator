import os
import numpy as np
import cv2
import mediapipe as mp

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Data directory where images are stored
DATA_DIR = './data'

# Initialize lists to store the data and corresponding labels
data = []
labels = []

def normalize_landmarks(landmarks):
    """
    Normalize hand landmarks to make them scale- and position-invariant.
    :param landmarks: List of landmark x, y coordinates
    :return: Normalized landmark list
    """
    x = [lm[0] for lm in landmarks]
    y = [lm[1] for lm in landmarks]

    # Calculate bounding box
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)

    # Normalize landmarks
    normalized_landmarks = []
    for lm in landmarks:
        norm_x = (lm[0] - min_x) / (max_x - min_x) if max_x != min_x else 0
        norm_y = (lm[1] - min_y) / (max_y - min_y) if max_y != min_y else 0
        normalized_landmarks.append([norm_x, norm_y])

    return normalized_landmarks

# Loop through each class directory
for label in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(class_dir):
        continue

    print(f"Processing images for class: {label}")
    # Process each image in the class directory
    for img_path in os.listdir(class_dir):
        image_path = os.path.join(class_dir, img_path)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read {image_path}")
            continue

        # Convert image to RGB for Mediapipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks (x, y coordinates only)
                raw_landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]

                # Normalize the landmarks
                normalized_landmarks = normalize_landmarks(raw_landmarks)

                # Flatten the normalized landmarks into a 1D list
                flattened_landmarks = np.array(normalized_landmarks).flatten().tolist()

                # Add the processed data and label to the lists
                data.append(flattened_landmarks)
                labels.append(label)

# Save processed data to NumPy files for faster loading
print("Saving dataset...")
data_array = np.array(data)
labels_array = np.array(labels)

# Save as .npy files
np.save('data.npy', data_array)
np.save('labels.npy', labels_array)

print("Dataset created and saved as NumPy arrays.")
