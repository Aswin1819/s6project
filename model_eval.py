import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical  # Import to_categorical function
from sklearn.metrics import classification_report

# Load the saved model
model = load_model('action.h5')

# Define the data path for evaluation
DATA_PATH = os.path.join('dataset')

# Define the number of sequences and sequence length
no_sequences = 30
sequence_length = 30

# Define actions based on your labels
actions = [folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))]
actions = np.array(actions)
# Create a label map
label_map = {label: num for num, label in enumerate(actions)}

# Load evaluation data
sequences_eval, labels_eval = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy"))
            window.append(res)
        sequences_eval.append(window)
        labels_eval.append(label_map[action])

X_eval = np.array(sequences_eval)
y_eval = to_categorical(labels_eval, num_classes=len(actions))  # Use to_categorical function

# Make predictions
predictions = model.predict(X_eval)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_eval, axis=1)

# Generate classification report
report = classification_report(true_labels, predicted_labels, target_names=actions)
print(report)
