import joblib
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load('best-model-0.96roc-41.pkl')
scaler = joblib.load('scaler1.pkl')

@app.get('/')
def get_home():
    return {'message': 'API top level working'}


@app.get('/predict/')
def predict_stroke(
        gender: str = Query(..., description="Gender: 0 for female, 1 for male"),
        age: str = Query(..., description="Age of the person"),
        hypertension: str = Query(..., description="1 if the person has hypertension, 0 otherwise"),
        heart_disease: str = Query(..., description="1 if the person has heart disease, 0 otherwise"),
        ever_married: str = Query(..., description="1 if the person is ever married, 0 otherwise"),
        work_type: str = Query(..., description="Work type: 0 for private, 1 for self-employed, 2 for gov job, 3 for children, 4 for never worked"),
        Residence_type: str = Query(..., description="Residence type: 0 for rural, 1 for urban"),
        avg_glucose_level: str = Query(..., description="Average glucose level"),
        bmi: str = Query(..., description="Body mass index"),
        smoking_status: str = Query( ..., description="Smoking status: 0 for unknown, 1 for never smoked, 2 for formerly smoked, 3 for smokes")
):
    features, feature_set = None, None

    features = [int(gender), int(age), int(hypertension), int(heart_disease), int(ever_married), int(
        work_type), int(Residence_type), float(avg_glucose_level), float(bmi), int(smoking_status)]
    feature_set = np.array([features]).reshape(1, -1)

    feature_set = pd.DataFrame(feature_set, columns=['gender', 'age', 'hypertension', 'heart_disease',
                               'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])

    prediction, probability = None, None

    feature_set_scaled = scaler.transform(feature_set)

    print(features)
    print(feature_set_scaled)

    prediction = model.predict(feature_set_scaled)[0]
    probability = model.predict_proba(feature_set_scaled)[0][1]
    probability = round(probability * 100, 2)

    return {"stroke_prediction": bool(prediction), "probability": float(probability)}
