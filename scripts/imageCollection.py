import os
import cv2

# Directory to store the dataset
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Classes for A-Z and 0-9
classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("0123456789")
dataset_size = 300  # Number of images per class (increase for better accuracy)

# Function to augment images
def augment_image(image):
    augmented_images = []
    # Horizontal flip
    augmented_images.append(cv2.flip(image, 1))
    # Rotate by 15 degrees
    rows, cols, _ = image.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
    augmented_images.append(cv2.warpAffine(image, M, (cols, rows)))
    # Rotate by -15 degrees
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -15, 1)
    augmented_images.append(cv2.warpAffine(image, M, (cols, rows)))
    return augmented_images

# Initialize webcam
cap = cv2.VideoCapture(0)  # Change to 1 if your webcam is not detected as the default

for label in classes:
    # Create a directory for each class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, label)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class: {label}')
    print('Press "Q" to start capturing images for this class.')

    # Wait for user to press "Q" to start capturing
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access webcam")
            break

        cv2.putText(frame, f'Prepare for {label}. Press "Q" to start!', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Start capturing images
    print(f'Capturing images for class: {label}')
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access webcam")
            break

        # Display progress
        cv2.putText(frame, f'Capturing: {label} ({counter}/{dataset_size})', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        # Save original image
        image_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)

        # Perform augmentation and save augmented images
        augmented_images = augment_image(frame)
        for aug_idx, aug_image in enumerate(augmented_images):
            aug_image_path = os.path.join(class_dir, f'{counter}_aug{aug_idx}.jpg')
            cv2.imwrite(aug_image_path, aug_image)

        counter += 1
        if cv2.waitKey(1) & 0xFF == 27:  # Press "ESC" to exit early
            print("Exiting...")
            break

print("Data collection completed!")
cap.release()
cv2.destroyAllWindows()
