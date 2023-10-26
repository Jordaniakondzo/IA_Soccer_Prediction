import pandas as pd
from predict import make_prediction
import pickle
from sklearn.metrics import confusion_matrix

# Load test and train data
X_test = pd.read_csv('data/X_test.csv')
X_train = pd.read_csv('data/X_train.csv')
y_test = pd.read_csv('data/y_test.csv')
y_train = pd.read_csv('data/y_train.csv')

def load_trained_model(model_path="models/LogisticRegression.pkl"):
    """
    Load the trained model from a .pkl file.

    Args:
    - model_path (str): Path to the saved model.

    Returns:
    - model: Loaded model.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Assuming y_test and predictions are already defined
# Load trained model
trained_model = load_trained_model()

# Make predictions
predictions = make_prediction(trained_model, X_test)

# Labels for the confusion matrix
labels = [1, 0, -1]

# Generate the confusion matrix
cm = confusion_matrix(y_test, predictions, labels=labels)

# Display the confusion matrix
print("Confusion Matrix:")
print(cm)

# Mapping labels for display
label_mapping = {1: "W (Home team win)", 0: "D (Draw)", -1: "L (Away team win)"}

# Display the results
for i, prediction in enumerate(predictions, 1):
    print(f"Match {i}: {label_mapping[prediction]}")

    """""
     After running this function:
     In the confusion matrix:
     
    1. The first row shows that 326 games were correctly predicted as home team wins (W).
    2. The second row shows that there was a small error where one match was misclassified, but 189 were correctly predicted as draws (D).
    3. The third row shows that 239 matches were correctly predicted as away team wins (L). 

    """
