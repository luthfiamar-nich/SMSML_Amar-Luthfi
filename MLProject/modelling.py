import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import dagshub

# Inisialisasi Remote Tracking DagsHub
dagshub.init(repo_owner="amar.luthfi18154", repo_name="SMSML_Amar-Luthfi", mlflow=True)
mlflow.set_experiment("Credit_Risk_Baseline")

def main():
    mlflow.sklearn.autolog()

    data_dir = 'dataset_preprocessing'
    if not os.path.exists(data_dir):
        data_dir = os.path.join('MLProject', 'dataset_preprocessing')

    print("[INFO] Membaca data hasil preprocessing (.csv)...")
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train_processed.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test_processed.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train_processed.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test_processed.csv')).values.ravel()

    with mlflow.start_run(run_name="Baseline_Random_Forest_Autolog"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"[Baseline] Model selesai dilatih. Akurasi: {acc:.4f}")

if __name__ == "__main__":
    main()
