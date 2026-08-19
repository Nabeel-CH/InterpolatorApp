"""Functions for handling 5D regression datasets:
    generating synthetic dataset, loading, cleaning (handling missing values), 
    splitting, standardising
"""

import os
import pickle
import math
import json
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_dataset(n_samples, seed=90):
    """
    Generate a synthetic 5D dataset of size n_samples with some low-noise.

    Parameters: 
        n_samples (int): number of samples to generate
        seed (int): random seed for reproducibility
    
    Returns:
        X (np.ndarray): generated features 
        y (np.ndarray): generated targets 
    """
    rng = np.random.default_rng(seed)
    # X feature
    X = rng.random((n_samples, 5))

    # Set feature variables
    x1, x2, x3, x4, x5 = X.T

    # Define y
    y = (
        np.sin(2 * math.pi * x1) * np.cos(2 * math.pi * x2)
        + 0.3 * np.exp(-(((x3 - 0.5) ** 2 + (x4 - 0.5) ** 2) / 0.02))
        + 0.5 * x5**2
        - 0.2 * x1 * x4
    )

    # add small Gaussian noise
    y += rng.normal(0.0, 0.01, size=n_samples)

    return X.astype(np.float32), y.astype(np.float32)

def generate_pkl(n=5000, seed=111, workdir='.', filename="synthetic_5d_data_test.pkl"):
    """
    Generate synthetic 5D dataset and save to a pickle file.

    Parameters:
        n (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
        workdir (str): Directory where the pickle file will be saved.
        filename (str): Name of the pickle file.

    Returns:
        filepath (str): Path to the saved pickle file.
    """
    os.makedirs(workdir, exist_ok=True)
    filepath = os.path.join(workdir, filename)

    print(f"Generating {n} synthetic 5D data points...")
    X, y = generate_dataset(n_samples=n, seed=seed)

    # Save as pkl file
    data_dict = {
        "X": X,
        "y": y,
        "metadata": {
            "n_samples": n,
            "n_features": 5,
            "seed": seed,
            "feature_names": ["x1", "x2", "x3", "x4", "x5"],
            "target_name": "y",
            "generated_at": json.dumps({"timestamp": str(np.datetime64("now"))}),
        },
    }

    with open(filepath, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"Saved synthetic data to: {filepath}")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(
        f"Feature ranges: x1-x5 ∈ [0,1], "
        f"y ∈ [{y.min():.3f}, {y.max():.3f}]"
    )

    return filepath

def handle_missing_values(X, y):
    """
    Remove rows where y is missing.
    Replace missing values in X with the mean of that feature

    Parameters: 
        X (np.ndarray): features of shape (n_samples, 5)
        y (np.ndarray): targets of shape (n_samples,)

    Returns:
        X_clean (np.ndarray): cleaned features
        y_clean (np.ndarray): cleaned targets
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # 1) Drop rows where y is NaN
    n_rows_before = X.shape[0]
    keep_rows = ~np.isnan(y)
    X_clean = X[keep_rows]
    y_clean = y[keep_rows]
    n_rows_after = X_clean.shape[0]
    rows_removed = n_rows_before - n_rows_after
    print(f"Removed {rows_removed} rows where y was missing.")

    # 2) Replace missing X values with column means
    n_missing_before = np.isnan(X_clean).sum()
    if n_missing_before > 0:
        imputer = SimpleImputer(strategy="mean")
        X_clean = imputer.fit_transform(X_clean)
        print(f"Replaced {n_missing_before} missing X values with column means.")

    return X_clean, y_clean


def split_and_standardise(X, y, random_state=20):
    """
    Split X and y into train / val / test then standardize features.
    
    Split ratio:

        - 60% train
        - 20% validation
        - 20% test

    e.g. For 5000 samples: 3000 / 1000 / 1000.

    Parameters: 
        X (np.ndarray): features of shape (n_samples, 5)
        y (np.ndarray): targets of shape (n_samples,)
        random_state (int): random state for reproducibility


    Returns:
        X_train_std (np.ndarray): standardised training features
        y_train (np.ndarray): training targets
        X_val_std (np.ndarray): standardised validation features
        y_val (np.ndarray): validation targets
        X_test_std (np.ndarray): standardised test features
        y_test (np.ndarray): test targets
        feature_mean (np.ndarray): feature means
        feature_std (np.ndarray): feature standard deviations
    """

    # Split into train and temporary (split later into validation and test) - 60% / 40%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, shuffle=True
    )

    # From temporary: split into val and test (50% / 50%) of the remaining 40%
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, shuffle=True
    )

    print("Train size:", X_train.shape[0])
    print("Validation size:  ", X_val.shape[0])
    print("Test size: ", X_test.shape[0])

    # Standardise using train stats
    scaler = StandardScaler()
    # fit on train and transform train
    X_train_std = scaler.fit_transform(X_train)   
    # val and test transform only
    X_val_std   = scaler.transform(X_val)        
    X_test_std  = scaler.transform(X_test)

    # Store feature means and standard deviations
    feature_mean = scaler.mean_
    feature_std  = scaler.scale_

    return X_train_std, y_train, X_val_std, y_val, X_test_std, y_test, feature_mean, feature_std

def load_from_bytes(content: bytes):
    """Load and validate dataset from pickled bytes.

    Parameters:
        content (bytes): Raw bytes of a pickled object containing features X and targets y.

    Returns:
        X (np.ndarray): features of shape (n_samples, 5)
        y (np.ndarray): targets of shape (n_samples,)

    Raises:
        ValueError: If the bytes cannot be unpickled, 
        or if dimensions are invalid.
    """
    try:
        obj = pickle.loads(content)
    except Exception as e:
        raise ValueError(f"Could not unpickle dataset: {e}")

    # Accept dictionary or tuple/list
    if isinstance(obj, dict) and "X" in obj and "y" in obj:
        X = np.asarray(obj["X"], dtype=float)
        y = np.asarray(obj["y"], dtype=float)
    elif isinstance(obj, (list, tuple)) and len(obj) >= 2:
        X = np.asarray(obj[0], dtype=float)
        y = np.asarray(obj[1], dtype=float)
    else:
        raise ValueError("Pickle must contain X and y (dict or tuple/list).")

    # validate dimensions
    if X.ndim != 2:
        raise ValueError(f"X must be 2D with shape (n_samples, 5), got {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D with shape (n_samples,), got {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples, got {X.shape[0]} and {y.shape[0]}."
        )
    if X.shape[1] != 5:
        raise ValueError(
            f"X must have exactly 5 features for 5D regression, but got {X.shape[1]}."
        )

    return X, y

def load_dataset(filename):
    """
    Load dataset from pickle file then split and standardise.

    Parameters:
        filename (str): Path to the pickle file

    Returns:
        X_train (np.ndarray): training features
        y_train (np.ndarray): Training targets
        X_val (np.ndarray): validation features
        y_val (np.ndarray): validation targets
        X_test (np.ndarray): test features
        y_test (np.ndarray): test targets
        feature_mean (np.ndarray): feature means
        feature_std (np.ndarray): feature standard deviations
    """

    with open(filename, "rb") as f:
        content = f.read()

    # Load dataset and validate dimensions
    X, y = load_from_bytes(content)
    
    n_samples = X.shape[0]
    print(f"Loaded dataset with {n_samples} samples and 5 features.")

    #fix missing values 
    X, y = handle_missing_values(X, y)

    #split and standardise data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_mean, feature_std = split_and_standardise(
    X, y, random_state=20)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_mean, feature_std