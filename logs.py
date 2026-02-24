from flask import Blueprint, request, jsonify
import joblib
import numpy as np
from similarity import calculate_similarity

logs_bp = Blueprint("logs", __name__)

model = joblib.load("isolation_model.pkl")

# TEMP storage for submitted codes
submitted_codes = []

@logs_bp.route("/api/logs", methods=["POST"])
def collect_logs():
    data = request.get_json()
    code = data.get("code", "")

    # STRICT POLICY: No paste
    if data["paste_events"] > 0:
        return jsonify({
            "status": "success",
            "risk_level": "High",
            "reason": "Copy-Paste detected"
        }), 200

    # SIMILARITY CHECK
    for previous_code in submitted_codes:
        similarity = calculate_similarity(code, previous_code)
        if similarity > 0.85:
            return jsonify({
                "status": "success",
                "risk_level": "High",
                "reason": f"Code similarity detected ({similarity*100:.2f}%)"
            }), 200

    # Store current code for future comparisons
    submitted_codes.append(code)

    # ML Check
    feature_vector = np.array([[
        data["typing_time"],
        data["paste_events"],
        data["paste_length"],
        data["attempts"],
        data["similarity_score"],
        data["avg_typing_speed"],
        data["time_to_code_ratio"]
    ]])

    prediction = model.predict(feature_vector)

    if prediction[0] == -1:
        risk = "High"
        reason = "Anomalous behavior detected"
    else:
        risk = "Normal"
        reason = "Behavior within normal range"

    return jsonify({
        "status": "success",
        "risk_level": risk,
        "reason": reason
    }), 200