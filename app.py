from flask import Flask
from routes.logs import logs_bp
import joblib

app = Flask(__name__)

# Load ML model once when server starts
model = joblib.load("isolation_model.pkl")

app.register_blueprint(logs_bp)

if __name__ == "__main__":
    app.run(debug=True)