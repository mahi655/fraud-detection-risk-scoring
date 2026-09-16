import os
import joblib
import pandas as pd

from api.preprocessing import engineer_features


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_fraud_model.pkl"
)


FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_names.pkl"
)


model = joblib.load(MODEL_PATH)

feature_names = joblib.load(FEATURE_PATH)


def predict_transaction(transaction_data):

    transaction_df = pd.DataFrame([transaction_data])

    transaction_df = engineer_features(transaction_df)

    transaction_df = transaction_df[feature_names]

    prediction = model.predict(transaction_df)[0]

    fraud_probability = model.predict_proba(
        transaction_df
    )[0][1]

    risk_score = fraud_probability * 100


    if risk_score < 30:
        risk_level = "Low Risk"
        decision = "Allow Transaction"

    elif risk_score < 70:
        risk_level = "Medium Risk"
        decision = "Review Transaction"

    else:
        risk_level = "High Risk"
        decision = "Block Transaction"


    return {
        "prediction": int(prediction),
        "fraud_probability": round(
            float(fraud_probability), 4
        ),
        "risk_score": round(
            float(risk_score), 2
        ),
        "risk_level": risk_level,
        "decision": decision
    }