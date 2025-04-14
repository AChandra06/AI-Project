
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

stock_model = joblib.load("improved_stock_model.pkl")
staff_model = joblib.load("api_ready_doctor_model.pkl")

@app.route("/")
def index():
    return jsonify({"message": "✅ Hospital Resource Forecasting API is running!"})

@app.route("/predict_stock", methods=["POST"])
def predict_stock():
    try:
        data = request.get_json()
        features = np.array([
            data["med_id"],
            data["day"],
            data["prev_day_demand"],
            data["rolling_3day_avg"],
            data["rolling_7day_avg"],
            data["cumulative_demand"]
        ]).reshape(1, -1)
        prediction = stock_model.predict(features)
        return jsonify({"predicted_stock_level": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict_staff", methods=["POST"])
def predict_staff():
    try:
        data = request.get_json()
        features = np.array([
            data["doctor_id"],
            data["day"],
            data["experience"],
            data["shift_type"],
            data["is_weekend"],
            data["week_number"],
            data["prev_day_seen"],
            data["rolling_3day_seen"],
            data["rolling_7day_seen"],
            data["rolling_14day_seen"],
            data["cumulative_seen"],
            data["patients_exp_interaction"]
        ]).reshape(1, -1)
        prediction = staff_model.predict(features)
        return jsonify({"predicted_available_staff": float(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
