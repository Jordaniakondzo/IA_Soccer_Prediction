import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def drop_unnecessary_columns(data):
    """
    Drop columns that are not required for the model.

    Args:
    - data (DataFrame): Raw data.

    Returns:
    - DataFrame: Data without unnecessary columns.
    """
    columns_to_drop = ['match_date', 'match_time', 'stadium', 'public_numb', 'away_team_home', 'away_team_away']
    return data.drop(columns=columns_to_drop)

def encode_categorical(data):
    """
    One-Hot encode categorical columns.

    Args:
    - data (DataFrame): Data with categorical columns.

    Returns:
    - DataFrame: Data with one-hot encoded columns.
    """
    categorical_columns = ['home_team', 'away_team']
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    
    encoded_data = encoder.fit_transform(data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

    data = pd.concat([data, encoded_df], axis=1).reset_index(drop=True)
    data = data.drop(columns=categorical_columns)

    return data

def encode_results(data):
    """
    Encode the 'home_result' and 'away_result' columns to numerical categories.

    Args:
    - data (DataFrame): Data with the 'home_result' and 'away_result' columns.

    Returns:
    - DataFrame: Data with the 'home_result' and 'away_result' columns encoded.
    """
    result_mapping = {'W': 1, 'L': -1, 'D': 0}
    data['home_result'] = data['home_result'].map(result_mapping).astype(int)
    data['away_result'] = data['away_result'].map(result_mapping).astype(int)
    
    return data

def scale_data(data):
    """
    Scale the data using standard scaling.

    Args:
    - data (DataFrame): Un-scaled data.

    Returns:
    - DataFrame: Scaled data.
    """
    columns_to_exclude = ['home_score', 'away_score', 'home_result', 'away_result']
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.difference(columns_to_exclude)
    
    scaler = StandardScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data

def preprocess(data):
    """
    Preprocess the raw data.

    Args:
    - data (DataFrame): Raw data.

    Returns:
    - DataFrame: Preprocessed data.
    """
    data = drop_unnecessary_columns(data)
    data = encode_categorical(data)
    data = encode_results(data)
    data = scale_data(data)
    
    return data
