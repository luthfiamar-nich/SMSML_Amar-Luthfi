import os
import pydantic
import pandas as pd
import numpy as np
from fastapi import FastAPI, Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Inisialisasi FastAPI
app = FastAPI(title="Credit Risk Inference Service with Prometheus Metrics")

# Registrasi Metrik Eksplisit untuk Scrape Prometheus
REQUEST_COUNT = Counter(
    'api_requests_total', 
    'Total request yang masuk ke API inferensi', 
    ['method', 'endpoint', 'http_status']
)
REQUEST_LATENCY = Histogram(
    'api_request_duration_seconds', 
    'Waktu respons API inferensi dalam detik', 
    ['endpoint']
)
PREDICTION_COUNT = Counter(
    'model_predictions_total', 
    'Total prediksi yang dihasilkan oleh model', 
    ['prediction_class']
)

# Model Validasi Input Data Nasabah
class LoanApplicant(pydantic.BaseModel):
    person_age: int
    person_income: int
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

def mock_predict(data):
    # Simulasi logika prediksi risiko gagal bayar (0 = Aman/Lancar, 1 = Gagal Bayar)
    if data['person_income'] > 50000 and data['loan_percent_income'] < 0.3:
        return 0
    return 1

@app.get("/")
def home():
    REQUEST_COUNT.labels(method='GET', endpoint='/', http_status='200').inc()
    return {"status": "Server Inferensi Credit Risk Aktif!"}

@app.post("/predict")
@REQUEST_LATENCY.labels(endpoint='/predict').time()
def predict(applicant: LoanApplicant):
    try:
        input_data = applicant.dict()
        pred = mock_predict(input_data)
        
        # Rekam metrik ke kolektor Prometheus
        PREDICTION_COUNT.labels(prediction_class=str(pred)).inc()
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', http_status='200').inc()
        
        status_label = "Lancar" if pred == 0 else "Risiko Gagal Bayar"
        return {"prediction": pred, "status_credit": status_label}
    except Exception as e:
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', http_status='500').inc()
        return {"error": str(e)}

@app.get("/metrics")
def metrics():
    # Mengembalikan data metrik mentah berformat standar Prometheus
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
