"""Script to plot and compare the performance of the model for different dataset sizes

This script plots and saves the following:
- Training and prediction time vs dataset size
- Peak memory usage during training and prediction vs dataset size
- Test MSE and R² vs dataset size

Results are saved to PNG files in backend/analysis/plots/.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

# set path to the csv file 
CURRENT_DIR = Path(__file__).parent
CSV_PATH = CURRENT_DIR / "performance_results.csv"

def load_results(csv_path=CSV_PATH):
    """
    Loads results from CSV file 

    The CSV file should have the following columns:
    n_samples, train_time_s, predict_time_s, train_peak_mb, predict_peak_mb, test_mse, test_r2.

    Parameters:
        csv_path (pathlib.Path, optional): Path to the CSV file containing results.

    Returns:
        dict: A dictionary containing:
            - n_samples (list[int]): Number of samples.
            - train_time (list[float]): Training times in seconds.
            - pred_time (list[float]): Prediction times in seconds.
            - train_peak (list[float]): Peak memory usage during training in MB.
            - pred_peak (list[float]): Peak memory usage during prediction in MB.
            - test_mse (list[float]): Test mean squared error values.
            - test_r2 (list[float]): Test R squared values.
    """
    n_samples = []
    train_time = []
    pred_time = []
    train_peak = []
    pred_peak = []
    test_mse = []
    test_r2 = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_samples.append(int(row["n_samples"]))
            train_time.append(float(row["train_time_s"]))
            pred_time.append(float(row["predict_time_s"]))
            train_peak.append(float(row["train_peak_mb"]))
            pred_peak.append(float(row["predict_peak_mb"]))
            test_mse.append(float(row["test_mse"]))
            test_r2.append(float(row["test_r2"]))

    return {
        "n_samples": n_samples,
        "train_time": train_time,
        "pred_time": pred_time,
        "train_peak": train_peak,
        "pred_peak": pred_peak,
        "test_mse": test_mse,
        "test_r2": test_r2,
    }

def main():
    """
    Loads results data and generates plots, which are then saved as PNG files in the "plots" folder.

    Three plots:
    1) Training and prediction time vs dataset size
    2) Peak memory usage vs dataset size
    3) test MSE and test R squared vs dataset size
    
    Returns:
        None
    """
    data = load_results()

    # Time vs dataset size
    plt.figure()
    plt.plot(data["n_samples"], data["train_time"], marker="o", label="Train time (s)")
    plt.plot(data["n_samples"], data["pred_time"], marker="o", label="Predict time (s)")
    plt.xlabel("Number of samples")
    plt.ylabel("Time (seconds)")
    plt.title("Training and prediction time vs dataset size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURRENT_DIR / "plots/time_vs_size.png", dpi=200)

    # Peak memory vs dataset size
    plt.figure()
    plt.plot(data["n_samples"], data["train_peak"], marker="o", label="Train peak (MB)")
    plt.plot(data["n_samples"], data["pred_peak"], marker="o", label="Predict peak (MB)")
    plt.xlabel("Number of samples")
    plt.ylabel("Peak memory (MB)")
    plt.title("Peak memory usage vs dataset size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CURRENT_DIR / "plots/memory_vs_size.png", dpi=200)
    # Accuracy (MSE + R²) vs dataset size
    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Number of samples")
    ax1.set_ylabel("Test MSE", color="tab:blue")
    ax1.plot(data["n_samples"], data["test_mse"], marker="o", color="tab:blue", label="Test MSE")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Test R²", color="tab:orange")
    ax2.plot(data["n_samples"], data["test_r2"], marker="s", color="tab:orange", label="Test R²")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    plt.title("Accuracy vs dataset size")
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.savefig(CURRENT_DIR / "plots/accuracy_vs_size.png", dpi=200)

    print("Saved plots to: ", CURRENT_DIR / "plots/")

# allows the script to be run from the command line
if __name__ == "__main__":
    main()
