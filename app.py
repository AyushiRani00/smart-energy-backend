from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Load trained ML model
model = joblib.load("energy_prediction_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    household_size = data["household_size"]
    temperature = data["temperature"]
    has_ac = data["has_ac"]
    peak_usage = data["peak_usage"]

    input_data = np.array([[household_size, temperature, has_ac, peak_usage]])
    prediction = model.predict(input_data)[0]

    return jsonify({
        "predicted_energy": round(float(prediction), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)