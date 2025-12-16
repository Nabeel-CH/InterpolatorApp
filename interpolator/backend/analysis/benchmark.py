"""Benchmark script to measure the performance of the model

This script measures:
- Training time as a function of dataset size
- Peak memory usage during training and prediction
- MSE and R² on the test set

Results are saved to a CSV file in backend/analysis/.
"""

import time
import tracemalloc
import csv
from pathlib import Path

from sklearn.metrics import mean_squared_error, r2_score

from fivedreg.model import NeuralNetwork
from fivedreg.data import generate_dataset, split_and_standardise



DATASET_SIZES = [1_000, 5_000, 10_000]


def run_with_time_and_mem(func, *args, **kwargs):
    """
    Run a function while measuring time and peak memory usage.

    Parameters:
        func (callable): Function to run.
        *args: Positional arguments passed to func.
        **kwargs: Keyword arguments passed to func.

    Returns:
        tuple:
            result: The return value from func(*args, **kwargs).
            elapsed_s (float): Time in seconds.
            peak_mb (float): Peak memory usage in mb.
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    result = func(*args, **kwargs)

    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed_s = t1 - t0
    peak_mb = peak / (1024**2)
    return result, elapsed_s, peak_mb


def train_model(X_train, y_train):
    """
    Create and train a NeuralNetwork on the given training data.

    Parameters:
        X_train (np.ndarray): Training features of shape (n_train, 5).
        y_train (np.ndarray): Training targets of shape (n_train,).

    Returns:
        NeuralNetwork: The trained model.
    """
    model = NeuralNetwork(
        input_dim=5,
        hidden_layers=[64, 32, 16],
        learning_rate=1e-3,
        max_epochs=200,
        batch_size=64,
    )
    model.fit(X_train, y_train)
    return model


def predict_batch(model, X):
    """
    Run predictions for a trained model.

    Parameters:
        model (NeuralNetwork): Trained model.
        X (np.ndarray): Features for prediction of shape (n_samples, 5).

    Returns:
        np.ndarray: Predicted targets of shape (n_samples,).
    """
    return model.predict(X)


def run_benchmarks():
    """
    Run benchmarks for all dataset sizes in DATASET_SIZES.

    For each size:
        1. Generate synthetic data.
        2. Split and standardise 
        3. Measure training time and peak memory.
        4. Measure prediction time and peak memory.
        5. Compute test MSE and R².

    Returns:
        list[dict]: contains the results for each dataset size
    """
    results = []

    for n in DATASET_SIZES:
        print(f"\n=== Benchmark for n_samples = {n} ===")

        # Generate synthetic data
        X, y = generate_dataset(n_samples=n, seed=111)

        # Split and standardise
        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            feature_mean,
            feature_std,
        ) = split_and_standardise(X, y, random_state=111)

        # Measure training time and peak memory
        model, train_time_s, train_peak_mb = run_with_time_and_mem(
            train_model, X_train, y_train
        )

        # Measure prediction time and peak memory on test set
        y_pred, pred_time_s, pred_peak_mb = run_with_time_and_mem(
            predict_batch, model, X_test
        )

        # Compute test MSE and R²
        test_mse = float(mean_squared_error(y_test, y_pred))
        test_r2 = float(r2_score(y_test, y_pred))

        result_row = {
            "n_samples": n,
            "train_time_s": train_time_s,
            "train_peak_mb": train_peak_mb,
            "predict_time_s": pred_time_s,
            "predict_peak_mb": pred_peak_mb,
            "test_mse": test_mse,
            "test_r2": test_r2,
        }
        results.append(result_row)

        print(f"train_time_s     = {train_time_s:.4f}")
        print(f"train_peak_mb    = {train_peak_mb:.2f}")
        print(f"predict_time_s   = {pred_time_s:.4f}")
        print(f"predict_peak_mb  = {pred_peak_mb:.2f}")
        print(f"test_mse         = {test_mse:.4f}")
        print(f"test_r2          = {test_r2:.4f}")

    return results


def save_results_csv(results, path="performance_results.csv"):
    """
    Save benchmark results to a CSV file in the same folder as this script.

    Parameters:
        results (list[dict]): Contains the results for each dataset size
        path (str): File name for the CSV output.
    """
    path = Path(__file__).parent / path
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "n_samples",
        "train_time_s",
        "train_peak_mb",
        "predict_time_s",
        "predict_peak_mb",
        "test_mse",
        "test_r2",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Round values 
            rounded_row = {
                "n_samples": row["n_samples"],
                "train_time_s": round(row["train_time_s"], 2),
                "train_peak_mb": round(row["train_peak_mb"], 2),
                "predict_time_s": round(row["predict_time_s"], 4),
                "predict_peak_mb": round(row["predict_peak_mb"], 4),
                "test_mse": round(row["test_mse"], 4),
                "test_r2": round(row["test_r2"], 4),
            }
            writer.writerow(rounded_row)

    print(f"\nSaved results to: {path}")

# allows the script to be run from the command line
if __name__ == "__main__":
    results = run_benchmarks()
    save_results_csv(results)
