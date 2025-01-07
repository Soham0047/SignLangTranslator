<<<<<<< HEAD
# Sign2Text2Speech
This is a Python-based project that converts hand signs into text using Mediapipe and a trained machine learning model, and further transforms the text into speech via Amazon Polly. It supports real-time recognition of A-Z and 0-9, leveraging data preprocessing, feature normalization, and advanced classifiers for accuracy.
=======
SignLangTranslator
SignLangTranslator is a Python-based application that translates hand signs into text and converts the text into speech using Amazon Polly. The project supports real-time recognition of letters (A-Z) and numbers (0-9) and employs Mediapipe for landmark detection and machine learning for classification.

Features
Real-time hand sign recognition using a webcam.
Supports letters (A-Z) and numbers (0-9).
Converts recognized text into speech using Amazon Polly.
Data preprocessing with Mediapipe for accurate landmark detection.
Machine learning model trained on normalized hand landmark features.
Robust text smoothing with sliding window predictions.
Technologies Used
Python
OpenCV: For capturing webcam input.
Mediapipe: For hand landmark detection.
Scikit-learn: For machine learning model training.
Amazon Polly: For text-to-speech conversion.
NumPy: For data manipulation and saving.
Matplotlib: For optional data visualization.
Getting Started
Prerequisites
Python 3.7 or above
Install the required Python packages:
bash
Copy code
pip install -r requirements.txt
Setup Amazon Polly
Configure AWS CLI with your credentials:
bash
Copy code
aws configure
Set up an IAM user with access to Amazon Polly.
Project Structure
bash
Copy code
SignLangTranslator/
│
├── data/                  # Collected hand sign images
├── scripts/               # Python scripts
│   ├── imageCollection.py    # Captures hand sign images
│   ├── createDataset.py      # Processes images into landmarks
│   ├── trainClassifier.py    # Trains the machine learning model
│   ├── inferenceClassifier.py# Real-time hand sign recognition
│
├── models/               # Saved machine learning model
│   └── model.p
├── audio/                # Generated speech files
│   └── output.mp3
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
How It Works
Data Collection:

Run imageCollection.py to capture images of hand signs.
Augments data for improved accuracy.
Preprocessing:

Run createDataset.py to extract and normalize hand landmarks from images.
Saves the dataset in data.npy and labels.npy.
Model Training:

Run trainClassifier.py to train a machine learning model on the dataset.
The model is saved in models/model.p.
Real-Time Recognition:

Run inferenceClassifier.py to recognize hand signs in real-time.
Converts recognized text into speech using Amazon Polly.
Usage
Run the application:
bash
Copy code
python scripts/inferenceClassifier.py
Controls:
Perform hand signs in front of the webcam.
Press SPACE to convert the text to speech.
Press ESC to exit.
Future Enhancements
Add support for gestures representing common phrases.
Train on larger, more diverse datasets for improved accuracy.
Implement multilingual speech synthesis via Amazon Polly.
Add support for dynamic gestures (e.g., "hello," "thank you").
Contributing
Contributions are welcome! Feel free to submit pull requests or report issues.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Let me know if you'd like help creating or refining any specific section!
>>>>>>> 5c4f9c3 (Initial commit: Sign Language Translator with Mediapipe and Polly)
