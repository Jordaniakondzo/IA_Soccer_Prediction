import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import preprocess
from sklearn.linear_model import LogisticRegression
import pickle
import os
from evaluate_model import evaluate_model


def load_data(filepath):
    """
    Load the data from a CSV file.

    Args:
    - filepath (str): Path to the CSV file.

    Returns:
    - DataFrame: Loaded data.
    """
    return pd.read_csv(filepath)

def split_data(data, target_column, test_size=0.2):
    """
    Split the data into training and test sets.

    Args:
    - data (DataFrame): The data to split.
    - target_column (str): The name of the target variable column.
    - test_size (float): Proportion of the data to be used as a test set.

    Returns:
    - tuple: (X_train, X_test, y_train, y_test)
    """
    # Identifying duplicate rows and printing the count
    print(f"Number of duplicate rows: {data.duplicated().sum()}")

    # Removing duplicate rows
    data = data.drop_duplicates()
    
    # Reset indices
    data.reset_index(drop=True, inplace=True)
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Validate the number of overlapping indices
    overlap_rows = pd.merge(X_train, X_test, how='inner')
    print(f"Number of overlapping rows: {len(overlap_rows)}")
    print(f"Number of duplicate rows: {data.duplicated().sum()}")

    return X_train, X_test, y_train, y_test


def save_to_csv(X_train, X_test, y_train, y_test, path="data/"):
    """
    Save the dataframes to CSV files.

    Args:
    - X_train, X_test, y_train, y_test (DataFrame): Dataframes to be saved.
    - path (str): Directory path to save the files.

    Returns:
    - None
    """
    X_train.to_csv(f"{path}X_train.csv", index=False)
    X_test.to_csv(f"{path}X_test.csv", index=False)
    y_train.to_csv(f"{path}y_train.csv", index=False)
    y_test.to_csv(f"{path}y_test.csv", index=False)

def train_model(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
    - X_train (DataFrame): Training data features.
    - y_train (Series): Training data target.

    Returns:
    - Trained model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def save_model(model, directory="models"):
    """
    Save the trained model to a .pkl file.

    Args:
    - model: The trained model.
    - directory (str): Directory to save the model.

    Returns:
    - None
    """
    # Extract the name of the model
    model_name = type(model).__name__
    
    # Combine the directory, model name, and ".pkl" to get the full path
    file_path = os.path.join(directory, f"{model_name}.pkl")

    # Check if directory exists, if not create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the model to the specified path
    with open(file_path, "wb") as f:
        pickle.dump(model, f)

    return file_path

def process_and_train(filepath):
    """
    Process the data and train the model.

    Args:
    - filepath (str): Path to the raw data file.

    Returns:
    - None
    """
    data = load_data(filepath)
    
    # Log the shape of original data
    print("Original data shape:", data.shape)
    
    preprocessed_data = preprocess(data)
    # Log the shape after preprocessing
    print("Preprocessed data shape:", preprocessed_data.shape)
    
    target_column = "home_result"
    X_train, X_test, y_train, y_test = split_data(preprocessed_data, target_column)

    # Save the split data to CSV files
    save_to_csv(X_train, X_test, y_train, y_test)
    
    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model on test data
    evaluate_model(model, X_test, y_test)


    # Save the trained model with a unique name
    save_model(model)

if __name__ == "__main__":
    filepath = "data/ligadata.csv"
    process_and_train(filepath)
