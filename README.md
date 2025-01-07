# Sign Language Translator

A machine vision and deep learning-based solution for translating sign language gestures into text or speech. This project aims to bridge the communication gap between individuals who use sign language and those who do not.

---

## 🚀 Features

- **Image Collection**: Automatically captures and collects images required for gesture recognition.
- **Dataset Creation**: Prepares a comprehensive dataset for training.
- **Model Training**: Includes a script to train a classifier for recognizing sign language gestures.
- **Inference**: Real-time or pre-recorded gesture translation into text/speech.
- **User-Friendly**: Easily extendable and adaptable to other sign languages or datasets.

---

## 🛠️ Technologies Used

- **Programming Language**: Python
- **Libraries and Tools**:
  - OpenCV for real-time image processing
  - TensorFlow/Keras or PyTorch for training the classifier
  - Numpy, Pandas - for data processing.

---

## 📁 Project Structure

```markdown
├── scripts/                     # All scripts related to main functionalities
│   ├── imageCollection.py       # Script to collect sign language images
│   ├── createDataset.py         # Script to preprocess and organize datasets
│   ├── trainClassifier.py       # Script for training the model
│   ├── inferenceClassifier.py   # Script for running predictions
│
├── README.md                    # Project documentation (this file)
├── .gitignore                   # List of ignored files and directories
```

---

## 🧩 Prerequisites

Before running the application, make sure to install the following:

1. **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/).
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Sign-Language-Translator
   ```

2. Set up a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On macOS or Linux
   .venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

1. **Run Image Collection Script**:
   Collect hand gesture images to create a dataset:
   ```bash
   python scripts/imageCollection.py
   ```

2. **Create Dataset**:
   Process the collected images into labeled datasets.
   ```bash
   python scripts/createDataset.py
   ```

3. **Train the Classifier**:
   Train a machine learning model to recognize gestures.
   ```bash
   python scripts/trainClassifier.py
   ```

4. **Run Inference**:
   Translate sign language into text or speech using the trained model.
   ```bash
   python scripts/inferenceClassifier.py
   ```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit changes:
   ```bash
   git commit -m "Add your message"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🌟 Acknowledgements

Special thanks to the developers of libraries like TensorFlow, OpenCV, and others that made this project possible.