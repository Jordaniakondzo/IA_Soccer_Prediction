# IA Soccer Prediction

## Overview
This project aims to use artificial intelligence to predict soccer match outcomes based on historical data. It not only predicts the result but also gives recommendations that can be used for betting or match analysis.

## Project Structure/Architecture

- `data`: Contains all datasets, including the main dataset `ligadata.csv` which is used for training, testing, and evaluating the model.
  
- `models`: Storage location for the trained models.
  
- `evaluate_model.py`: Evaluates the model's performance against test data and generates performance metrics.
  
- `exploratory_analysis.py`: Contains code related to initial data analysis and understanding of the dataset.
  
- `main.py`: The main execution script that ties everything together.
  
- `model_training.py`: Contains all the logic for training models based on the dataset.
  
- `predict.py`: Uses the trained model to make predictions on new data.
  
- `preprocessing.py`: Handles data preprocessing steps like normalization, encoding, and splitting.
  
- `requirements.txt`: Lists all dependencies necessary to run this project.

## Workflow

1. **Data Preprocessing**: The raw data from `ligadata.csv` undergoes preprocessing where it is cleaned, normalized, and split into training and test datasets.

2. **Exploratory Data Analysis (EDA)**: A deep dive into the data is performed to understand patterns, relationships, and anomalies, if any.

3. **Model Training**: Using the training data, various models are trained to understand and learn from the historical data.

4. **Model Evaluation**: The trained models are then evaluated against the test data to understand their accuracy and predictive capability.

5. **Prediction & Recommendation**: Based on the trained model, predictions are made on new data, and subsequent betting recommendations are generated.

## Setup and Execution

### Key Libraries

To install the required libraries:

*bash:*
pip install -r requirements.txt.

- `pandas`: For data manipulation and analysis.
  
- `scikit-learn`: For implementing machine learning algorithms.
  
- `matplotlib` & `seaborn`: For data visualization.

### Running the Application

1. **Navigate to the project directory**: cd path/to/IA_SOCCER_PREDICTION

2. **Run the main script**: python main.py

**Note**: *The trained models are stored in serialized files, making it easier to reuse them without retraining. This ensures both efficiency and consistency in predictions. They are saved in the models folder.*

3. **For model evaluation**: python evaluate_model.py

4. **For exploratory data analysis**: python exploratory_analysis.py

5. **For predictions and recommendations**: python predict.py

## Data

The primary dataset used is `ligadata.csv`:

- **Training Data**: This is the data that the model learns from. It comprises features (`X_train`) and the target outcome (`y_train`).
  
- **Test Data**: After training, the model is evaluated on the test data to understand its performance in real-world scenarios. This data consists of features (`X_test`) which the model hasn't seen during training.

## Recommendations

Based on the prediction results from the model, three types of recommendations are provided:

- **Recommended**: Indicates a high probability of the home team winning.
  
- **Dangerous**: Indicates a high probability of the away team winning or a weak performance from the home team.
  
- **Suspicious**: Cases where the outcome isn't very clear.

## Future Enhancements

- Incorporate more features like player statistics, weather conditions, etc.
  
- Use deep learning techniques for better accuracy.
  
- Develop a user-friendly frontend for users to interact with the model.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure you have tested your changes thoroughly before submitting.

## License

This project is open-source, licensed under [MIT License](https://opensource.org/licenses/MIT).
