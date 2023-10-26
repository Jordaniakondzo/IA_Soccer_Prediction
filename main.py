import pandas as pd
import model_training

# Load data
def load_data(filepath):
    """
    Load the data from the given filepath.

    Args:
    - filepath (str): Path to the CSV data file.

    Returns:
    - DataFrame: Loaded data.
    """
    return pd.read_csv(filepath)

if __name__ == "__main__":
    data_path = 'data/ligadata.csv'

    # Passing the data for further processing and model training
    model_training.process_and_train(data_path)
    