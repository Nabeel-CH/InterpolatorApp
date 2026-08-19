"""FastAPI backend for the interpolator.

Three main endpoints:

- POST /upload  : upload a 5D dataset in .pkl file
- POST /train   : train the neural network on the uploaded data
- POST /predict : prediction value for given 5 inputs 
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import numpy as np

from fivedreg.data import handle_missing_values, split_and_standardise, load_from_bytes
from fivedreg.model import NeuralNetwork

# Global variables for uploaded data and model
CURRENT_X = None
CURRENT_Y = None
TRAINED_MODEL = None
TRAIN_STATS = None  #mean and std for standardisation


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InterpolatorInput(BaseModel):
    """
    5D feature input for prediction.

    Attributes:
        v1 (float): First feature.
        v2 (float): Second feature.
        v3 (float): Third feature.
        v4 (float): Fourth feature.
        v5 (float): Fifth feature.
    """
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float

class TrainConfig(BaseModel):
    """
    Configuration for training.

    Attributes:
        hidden_layers (list[int]): number of units in each hidden layer.
        learning_rate (float): learning rate.
        max_epochs (int): number of training epochs.
        batch_size (int): batch size.
    """
    hidden_layers: list[int] = [64, 32, 16]
    learning_rate: float = 1e-3
    max_epochs: int = 200
    batch_size: int = 64

@app.get("/")
def hello():
    """
    Welcome endpoint
    """
    return {"message": "Interpolator backend is running :)"}

@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    return {"status": "ok"}


# 1) Upload dataset 
@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a 5D dataset in .pkl file.
    The uploaded file must be a .pkl file containing X and y
    The data is validated and stored in memory for training.

    Returns:
        dict: Summary of the uploaded dataset:
            - message (str): Status message.
            - n_samples (int): Number of samples.
            - n_features (int): Number of features (should be 5).
            - feature_mins (list[float]): feature min for sliders
            - feature_maxs (list[float]): feature max for sliders
            - y_min (float): Min target value for UI
            - y_max (float): Max target value for UI
            - preview_X (list[list[float]]): First 5 rows of X for UI
            - preview_y (list[float]): First 5 values of y for UI

    Raises:
        HTTPException: If the file is not a .pkl or data is invalid.
    """
    global CURRENT_X, CURRENT_Y, TRAINED_MODEL, TRAIN_STATS

    # Only allow .pkl files
    if not file.filename.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Please upload a .pkl file.")

    # Read content from the upload
    content = await file.read()

    # Load and validate dataset
    try:
        X, y = load_from_bytes(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Store in global variables for training
    CURRENT_X = X
    CURRENT_Y = y
    TRAINED_MODEL = None
    TRAIN_STATS = None

    # feature ranges for sliders
    feature_mins = X.min(axis=0).tolist()
    feature_maxs = X.max(axis=0).tolist()

    # target range
    y_min = float(y.min())
    y_max = float(y.max())

    # first 5 rows for preview
    preview_size = min(5, X.shape[0])
    preview_X = X[:preview_size].tolist()
    preview_y = y[:preview_size].tolist()

    return {
        "message": "Dataset uploaded successfully.",
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_mins": feature_mins,
        "feature_maxs": feature_maxs,
        "y_min": y_min,
        "y_max": y_max,
        "preview_X": preview_X,
        "preview_y": preview_y,
    }

# 2) Train model
@app.post("/train")
def train_model(config: TrainConfig):
    """
    Train the neural network.

    This endpoint:
      1. Cleans the data.
      2. Splits data.
      3. Trains the NeuralNetwork model.
      4. Stores the trained model and standardisation statistics

    Parameters:
        config (TrainConfig): Training hyperparameters.

    Returns:
        dict: Training summary including:
            - message (str): Status message.
            - train_samples (int): Number of training samples.
            - val_samples (int): Number of validation samples.
            - test_samples (int): Number of test samples.
            - val_mse (float): Validation MSE.

    Raises:
        HTTPException: When nodataset has been uploaded.
    """
    global CURRENT_X, CURRENT_Y, TRAINED_MODEL, TRAIN_STATS

    if CURRENT_X is None or CURRENT_Y is None:
        raise HTTPException(
            status_code=400,
            detail="No dataset uploaded. Use /upload first.",
        )

    # Handle missing values
    X_clean, y_clean = handle_missing_values(CURRENT_X, CURRENT_Y)

    # Split data and standardise features
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_mean,
        feature_std,
    ) = split_and_standardise(X_clean, y_clean, random_state=0)

    # Create and train the model
    model = NeuralNetwork(
        input_dim=X_train.shape[1],
        hidden_layers=config.hidden_layers,
        learning_rate=config.learning_rate,
        max_epochs=config.max_epochs,
        batch_size=config.batch_size,
    )

    model.fit(X_train, y_train)

    # validation MSE for UI
    val_preds = model.predict(X_val)
    val_mse = float(np.mean((val_preds - y_val) ** 2))

    # Store trained model and stats
    TRAINED_MODEL = model
    TRAIN_STATS = {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }

    return {
        "message": "Model trained successfully.",
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "val_mse": val_mse,
    }



# 3) Predict
@app.post("/predict")
def predict(input_data: InterpolatorInput):
    """
    Make prediction for a 5D input.

    The input is standardised using the training set statistics and 
    then passed through the trained model.

    Parameters:
        input_data (InterpolatorInput): 5D input.

    Returns:
        dict: Prediction result:
            - input (list[float]): Input values.
            - prediction (float): Prediction.

    Raises:
        HTTPException: When model has not been trained yet.
    """
    if TRAINED_MODEL is None or TRAIN_STATS is None:
        raise HTTPException(
            status_code=400,
            detail="Model is not trained yet. Call /train first.",
        )

    # Build array of input values
    x = np.array(
        [[input_data.v1, input_data.v2, input_data.v3, input_data.v4, input_data.v5]],
        dtype=float,
    )

    mean = TRAIN_STATS["feature_mean"]
    std = TRAIN_STATS["feature_std"]

    # Standardise input
    x_std = (x - mean) / std

    # Predict using model
    y_pred = TRAINED_MODEL.predict(x_std)[0]

    return {
        "input": x.tolist()[0],
        "prediction": float(y_pred),
    }