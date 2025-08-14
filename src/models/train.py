import pandas as pd
import joblib # Recommended library for saving models [cite: 48]
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor # Example model
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Import your preprocessor
from src.data.preprocess import preprocess_data

def train_model(X_train, y_train, preprocessor):
    """
    Trains a machine learning model on the training data.
    """
    # Create the full pipeline with preprocessing and the model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', CatBoostRegressor({'depth': 4, 'iterations': 100, 'l2_leaf_reg': 5, 'learning_rate': 0.01})) # Or another model
    ])
    
    # Train the model
    model_pipeline.fit(X_train, y_train)
    
    return model_pipeline

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('data/raw/train.csv')

    # Define features and target variable
    features = data.drop('Recycling Rate (%)', axis=1) # All other columns 
    target = data['Recycling Rate (%)']
    
    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

    # Preprocess and train
    preprocessor = preprocess_data(data)
    trained_model = train_model(X_train, y_train, preprocessor)
    
    # Evaluate the model (e.g., calculate RMSE on the validation set)
    predictions = trained_model.predict(X_val)
    rmse = mean_squared_error(y_val, predictions, squared=False)
    print(f"Validation RMSE: {rmse}")

    # Save the trained model [cite: 95]
    joblib.dump(trained_model, 'models/trained_model.pkl')