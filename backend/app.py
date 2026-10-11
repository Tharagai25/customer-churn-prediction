from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model
model_path = os.path.join(BASE_DIR, "model", "customer_churn_model.keras")
preprocessor_path = os.path.join(BASE_DIR, "model", "preprocessor.pkl")

model = tf.keras.models.load_model(model_path)
preprocessor = joblib.load(preprocessor_path)

print("Customer Churn Model loaded successfully!")
print("Preprocessor loaded successfully!")


@app.route("/")
def home():
    return jsonify({
        "message": "Customer Churn Prediction Backend is running!"
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    try:
        df = pd.DataFrame([data])

        # Preprocess input
        processed_data = preprocessor.transform(df)

        # Prediction
        prediction = model.predict(processed_data, verbose=0)

        probability = float(prediction[0][0])

        if probability >= 0.5:
            result = "Customer will churn"
        else:
            result = "Customer will not churn"

        return jsonify({
            "prediction": result,
            "churn_probability": round(probability * 100, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)