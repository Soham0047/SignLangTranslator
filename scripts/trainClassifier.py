import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load preprocessed data
print("Loading dataset...")
data = np.load('data.npy')
labels = np.load('labels.npy')

# Encode labels as integers
label_set = sorted(set(labels))
label_to_int = {label: i for i, label in enumerate(label_set)}
int_to_label = {i: label for i, label in enumerate(label_set)}

# Convert labels to integers
encoded_labels = np.array([label_to_int[label] for label in labels])

# Standardize features
print("Standardizing features...")
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
)

# Train a Random Forest Classifier
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_set))

# Ensure the 'models/' directory exists
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# Save the trained model and scaler
model_file = os.path.join(model_dir, 'model.p')
with open(model_file, 'wb') as f:
    pickle.dump({'model': model, 'int_to_label': int_to_label, 'scaler': scaler}, f)

print(f"Model saved to {model_file}")
