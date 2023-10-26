import pandas as pd
import pickle

# Load the trained model from the saved location
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

def make_prediction(model, input_data):
    """
    Make a prediction using the trained model.

    Args:
    - model (model object): Trained machine learning model.
    - input_data (DataFrame): Data for which predictions are to be made.

    Returns:
    - array: Predictions.
    """
    return model.predict(input_data)

def map_predictions_to_results(predictions):
    """
    Map numerical predictions to match outcomes, specifically from the perspective of the home team.

    This mapping takes into consideration that the target column "home_result" is based on the outcome for 
    the home team. Therefore:
    - "W (Home team win)" means the home team won the match.
    - "D (Draw)" indicates the match ended in a draw.
    - "L (Away team win)" signifies that the home team lost, and the away team won.

    It's essential to understand that these predictions are always relative to the home team's performance.

    Args:
    - predictions (array): Predictions made by the model.

    Returns:
    - list: List of match outcomes with respect to the home team's performance.
    """
    result_mapping = {
        1: "W (Home team win)",
        0: "D (Draw)",
        -1: "L (Away team win)"
    }
    return [result_mapping[pred] for pred in predictions]

def recommendation_based_on_data(row):
    """
    Provide a recommendation based on the performance of the home team.

    This function evaluates the match data to categorize the recommendation 
    into one of three categories: 'Recommended', 'Dangerous', or 'Suspicious'.

    Recommendations are primarily made based on the home team's performance metrics 
    such as possession, accuracy, total shots, and score difference. 

    The thresholds used in this function are adjustable. They can be modified based 
    on the user's judgment regarding betting or a deeper analysis of the match data.

    Parameters:
    - row (dict): A dictionary containing match data for both home and away teams.

    Returns:
    - str: A string representing the recommendation ('Recommended', 'Dangerous', or 'Suspicious').

    """
    # Recommend if home team has strong performance metrics
    if (row['home_possesion'] > row['away_possession']) and \
       (row['home_accuaracy'] > 0.2) and (row['home_total_Shots'] > 0.3) and \
       (row['home_score_home'] - row['away_score_home'] > 0.2):
        return "Recommended"
    
    # Mark as dangerous if away team's performance is strong or home team's performance is weak
    elif (row['home_possesion'] > row['away_possession'] - 0.1) or \
         (row['away_accuaracy'] > 0.5) and (row['away_total_Shots'] > 0.5) and \
         (row['away_score_away'] - row['home_score_away'] > 0):
        return "Dangerous"
    
    # Mark as suspicious for all other cases
    else:
        return "Suspicious"


if __name__ == "__main__":
    # Load preprocessed test data
    X_test = pd.read_csv("data/X_test.csv")
    
    # Load trained model
    trained_model = load_trained_model()

    # Make predictions on test data
    predictions = make_prediction(trained_model, X_test)
    probs = trained_model.predict_proba(X_test)

    # Map predictions to match outcomes
    match_outcomes = map_predictions_to_results(predictions)
    recommendations = X_test.apply(recommendation_based_on_data, axis=1)

    # Print out match outcomes and recommendations
    for i, (outcome, rec) in enumerate(zip(match_outcomes, recommendations), 1):
        print(f"Match {i}: {outcome} -> {rec}")
