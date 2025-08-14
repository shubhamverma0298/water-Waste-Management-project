from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model once when the app starts
model = joblib.load('models/trained_model.pkl')

@app.route('/')
def home():
    return render_template('index.html') # A basic HTML template

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # The input data should match the features the model was trained on
    input_df = pd.DataFrame([data])
    
    # Make a prediction
    prediction = model.predict(input_df)
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    # Run the app
    # For local testing:
    app.run(debug=True)
    # For production: use a WSGI server like Gunicorn [cite: 48]
    # Example command (run in terminal, not in Python):