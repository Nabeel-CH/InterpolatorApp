# test all main.py API endpoints

import io
import pickle
import numpy as np
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def generate_test_pkl():
    """
    Create a simple .pkl file for testing with 100 samples and 5 features
    """
    rng = np.random.default_rng(0)
    # generate random data
    X = rng.normal(size=(100, 5))
    y = X.sum(axis=1) 

    # save to .pkl file
    test_data = io.BytesIO()
    pickle.dump({"X": X, "y": y}, test_data)
    test_data.seek(0)

    return test_data


def test_health_endpoint():
    """
    test /health endpoint
    """
    resp = client.get("/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"

def test_train_fails():
    """
    Calling /train before /upload should fail
    """
    train_payload = {
        "hidden_layers": [16, 8],
        "learning_rate": 1e-3,
        "max_epochs": 5,
        "batch_size": 16,
    }
    resp = client.post("/train", json=train_payload)
    assert resp.status_code == 400
    assert "No dataset uploaded" in resp.json()["detail"]


def test_predict_fails():
    """
    Calling /predict before /train should fail
    """
    payload = {"v1": 0.1, "v2": 0.2, "v3": 0.3, "v4": 0.4, "v5": 0.5}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400
    assert "Model is not trained yet" in resp.json()["detail"]


def test_full_flow():
    """
    Full end-to-end test:
    1) upload dataset
    2) train model
    3) predict for a 5D input
    """
    # 1) Upload
    test_data = generate_test_pkl()
    files = {
      "file": ("test_dataset.pkl", test_data.read(), "application/octet-stream")
    }

    # check upload endpoint works
    upload_resp = client.post("/upload", files=files)
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()

    # check dataset was uploaded correctly
    assert upload_data["n_samples"] == 100
    assert upload_data["n_features"] == 5

    # 2) Train
    train_payload = {
        "hidden_layers": [16, 8],
        "learning_rate": 1e-3,
        "max_epochs": 5,
        "batch_size": 16,
    }

    # check train endpoint works
    train_resp = client.post("/train", json=train_payload)
    assert train_resp.status_code == 200

    # check training summary was returned
    train_data = train_resp.json()
    assert train_data["train_samples"] > 0
    assert train_data["val_samples"] > 0
    assert train_data["test_samples"] > 0

    # 3) Predict
    predict_payload = {
        "v1": 0.1,
        "v2": 0.2,
        "v3": 0.3,
        "v4": 0.4,
        "v5": 0.5,
    }

    # check predict endpoint works
    predict_resp = client.post("/predict", json=predict_payload)
    assert predict_resp.status_code == 200

    # check prediction was returned
    pred_data = predict_resp.json()
    assert len(pred_data["input"]) == 5
    assert isinstance(pred_data["prediction"], (float, int))