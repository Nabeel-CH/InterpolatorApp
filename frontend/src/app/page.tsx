"use client";

import { FormEvent, useState, useEffect } from "react";

type PredictResponse = {
  input: number[];
  prediction: number;
};

type UploadResponse = {
  message: string;
  n_samples: number;
  n_features: number;
  feature_mins?: number[];
  feature_maxs?: number[];
  y_min?: number;
  y_max?: number;
  preview_X?: number[][];
  preview_y?: number[];
};

type TrainResponse = {
  message: string;
  train_samples: number;
  val_samples: number;
  test_samples: number;
  val_mse: number;
};

type StepStatus = "pending" | "active" | "completed" | "failed";
type BackendStatus = "checking" | "up" | "down";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const baseButton =
  "inline-flex items-center justify-center rounded-md px-4 py-2 text-sm border transition";
const primaryButton =
  baseButton +
  " border-black bg-black text-white hover:bg-white hover:text-black disabled:opacity-50 disabled:hover:bg-black disabled:hover:text-white";

export default function Home() {
  const [step, setStep] = useState<number>(1);
  const [stepStatuses, setStepStatuses] = useState<{ [key: number]: StepStatus }>({
    1: "active",
    2: "pending",
    3: "pending",
  });

  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");

  const [values, setValues] = useState({
    v1: "",
    v2: "",
    v3: "",
    v4: "",
    v5: "",
  });

  const [learningRate, setLearningRate] = useState<string>("0.001");
  const [maxEpochs, setMaxEpochs] = useState<string>("200");
  const [batchSize, setBatchSize] = useState<string>("64");
  const [hiddenLayersText, setHiddenLayersText] = useState<string>("64,32,16");

  const [result, setResult] = useState<PredictResponse | null>(null);
  const [uploadInfo, setUploadInfo] = useState<UploadResponse | null>(null);
  const [trainInfo, setTrainInfo] = useState<TrainResponse | null>(null);

  const [loading, setLoading] = useState(false);
  const [datasetUploaded, setDatasetUploaded] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [training, setTraining] = useState(false);
  const [trained, setTrained] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [featureMins, setFeatureMins] = useState<number[] | null>(null);
  const [featureMaxs, setFeatureMaxs] = useState<number[] | null>(null);

  // --- Backend health check for "Backend connected" indicator ---
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/health`, { method: "GET" });
        if (res.ok) {
          setBackendStatus("up");
        } else {
          setBackendStatus("down");
        }
      } catch {
        setBackendStatus("down");
      }
    };

    checkHealth();
  }, []);

  const handleChange = (field: keyof typeof values, value: string) => {
    setValues((prev) => ({ ...prev, [field]: value }));
  };

  const updateStepStatus = (stepNum: number, status: StepStatus) => {
    setStepStatuses((prev) => ({ ...prev, [stepNum]: status }));
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    const payload = {
      v1: parseFloat(values.v1),
      v2: parseFloat(values.v2),
      v3: parseFloat(values.v3),
      v4: parseFloat(values.v4),
      v5: parseFloat(values.v5),
    };

    if (Object.values(payload).some((v) => Number.isNaN(v))) {
      setError("Please select valid values for all 5 features.");
      updateStepStatus(3, "failed");
      return;
    }

    try {
      setLoading(true);

      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        let msg = `Request failed (status ${res.status})`;
        try {
          const data = await res.json();
          msg += data.detail ? `: ${data.detail}` : `: ${JSON.stringify(data)}`;
        } catch {
          const text = await res.text();
          if (text) msg += `: ${text}`;
        }
        throw new Error(msg);
      }

      const data: PredictResponse = await res.json();
      setResult(data);
      updateStepStatus(3, "completed");
    } catch (err: any) {
      setError(err?.message || "Something went wrong while running the model.");
      updateStepStatus(3, "failed");
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file?: File) => {
    setError(null);
    setUploadInfo(null);
    setTrainInfo(null);
    setTrained(false);

    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pkl")) {
      setError("Please upload a .pkl file.");
      updateStepStatus(1, "failed");
      return;
    }

    setUploadedFileName(file.name);

    try {
      const form = new FormData();
      form.append("file", file);

      const res = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        let msg = `Upload failed (status ${res.status})`;
        try {
          const data = await res.json();
          msg += data.detail ? `: ${data.detail}` : `: ${JSON.stringify(data)}`;
        } catch {
          const text = await res.text();
          if (text) msg += `: ${text}`;
        }
        throw new Error(msg);
      }

      const data: UploadResponse = await res.json();
      setUploadInfo(data);
      setDatasetUploaded(true);
      updateStepStatus(1, "completed");
      updateStepStatus(2, "pending");

      if (
        data.feature_mins &&
        data.feature_maxs &&
        data.feature_mins.length === 5 &&
        data.feature_maxs.length === 5
      ) {
        setFeatureMins(data.feature_mins);
        setFeatureMaxs(data.feature_maxs);

        const defaults = data.feature_mins.map(
          (min, i) => (min + data.feature_maxs![i]) / 2
        );

        setValues({
          v1: defaults[0].toString(),
          v2: defaults[1].toString(),
          v3: defaults[2].toString(),
          v4: defaults[3].toString(),
          v5: defaults[4].toString(),
        });
      } else {
        setFeatureMins(null);
        setFeatureMaxs(null);
      }
    } catch (err: any) {
      setError(err?.message || "Failed to upload dataset.");
      setDatasetUploaded(false);
      updateStepStatus(1, "failed");
    }
  };

  const handleTrain = async () => {
    setError(null);
    setTrainInfo(null);

    if (!datasetUploaded) {
      setError("Please upload a dataset first.");
      updateStepStatus(2, "failed");
      return;
    }

    const lr = parseFloat(learningRate);
    const epochs = parseInt(maxEpochs, 10);
    const batch = parseInt(batchSize, 10);
    const hiddenLayers = hiddenLayersText
      .split(",")
      .map((part) => parseInt(part.trim(), 10))
      .filter((n) => !Number.isNaN(n) && n > 0);

    if (Number.isNaN(lr) || lr <= 0) {
      setError("Learning rate must be a positive number.");
      updateStepStatus(2, "failed");
      return;
    }
    if (Number.isNaN(epochs) || epochs <= 0) {
      setError("Max epochs must be a positive integer.");
      updateStepStatus(2, "failed");
      return;
    }
    if (Number.isNaN(batch) || batch <= 0) {
      setError("Batch size must be a positive integer.");
      updateStepStatus(2, "failed");
      return;
    }
    if (hiddenLayers.length === 0) {
      setError(
        "Hidden layers must be a comma-separated list of positive integers (e.g. 64,32,16)."
      );
      updateStepStatus(2, "failed");
      return;
    }

    try {
      setTraining(true);

      const res = await fetch(`${API_BASE_URL}/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          hidden_layers: hiddenLayers,
          learning_rate: lr,
          max_epochs: epochs,
          batch_size: batch,
        }),
      });

      if (!res.ok) {
        let msg = `Train failed (status ${res.status})`;
        try {
          const data = await res.json();
          msg += data.detail ? `: ${data.detail}` : `: ${JSON.stringify(data)}`;
        } catch {
          const text = await res.text();
          if (text) msg += `: ${text}`;
        }
        throw new Error(msg);
      }

      const data: TrainResponse = await res.json();
      setTrainInfo(data);
      setTrained(true);
      updateStepStatus(2, "completed");
      updateStepStatus(3, "pending");
    } catch (err: any) {
      setError(err?.message || "Failed to train model.");
      setTrained(false);
      updateStepStatus(2, "failed");
    } finally {
      setTraining(false);
    }
  };

  const getSliderBounds = (idx: number): { min: number; max: number } => {
    if (
      featureMins &&
      featureMaxs &&
      featureMins.length === 5 &&
      featureMaxs.length === 5
    ) {
      return { min: featureMins[idx], max: featureMaxs[idx] };
    }
    return { min: -1, max: 1 };
  };

  const getStepColorClasses = (stepId: number) => {
    const status = stepStatuses[stepId];
    const isActive = step === stepId;

    if (status === "completed") {
      return "border-green-500 bg-green-50";
    } else if (status === "failed") {
      return "border-red-500 bg-red-50";
    } else if (isActive) {
      return "border-blue-500 bg-blue-50 font-semibold";
    } else {
      return "border-gray-300 bg-white";
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-black font-sans">
      <main className="w-full max-w-4xl mx-auto p-6 md:p-8">
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm p-6 md:p-8">
          <header className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="text-center sm:text-left">
              <h1 className="text-3xl font-semibold tracking-tight">
                5D Interpolator
              </h1>
              <p className="mt-2 text-sm text-gray-600">
                Upload a dataset, train a model, then explore predictions.
              </p>
            </div>

            {/* Backend connection status pill */}
            <div className="flex justify-center sm:justify-end">
              <div className="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs bg-gray-50">
                <span
                  className={`h-2.5 w-2.5 rounded-full ${
                    backendStatus === "up"
                      ? "bg-green-500"
                      : backendStatus === "down"
                      ? "bg-red-500"
                      : "bg-gray-400 animate-pulse"
                  }`}
                />
                <span className="font-medium">
                  {backendStatus === "up"
                    ? "Backend: connected"
                    : backendStatus === "down"
                    ? "Backend: unreachable"
                    : "Checking backend..."}
                </span>
              </div>
            </div>
          </header>

          <div className="flex gap-3 mb-6">
            {[
              { id: 1, title: "Upload" },
              { id: 2, title: "Train" },
              { id: 3, title: "Predict" },
            ].map((box) => (
              <button
                key={box.id}
                type="button"
                onClick={() => {
                  setStep(box.id);
                  if (
                    stepStatuses[box.id] === "pending" ||
                    stepStatuses[box.id] === "failed"
                  ) {
                    updateStepStatus(box.id, "active");
                  }
                }}
                className={`flex-1 rounded-lg border px-4 py-3 text-sm text-left transition ${getStepColorClasses(
                  box.id
                )}`}
              >
                <div className="text-[11px] uppercase tracking-wide text-gray-500">
                  Step {box.id}
                </div>
                <div className="mt-1 text-sm">{box.title}</div>
              </button>
            ))}
          </div>

          {error && (
            <div className="mb-4 text-sm border border-red-200 bg-red-50 text-red-700 px-3 py-2 rounded-md">
              {error}
            </div>
          )}

          <div className="flex justify-center">
            <div className="w-full max-w-xl">
              {step === 1 && (
                <div className="space-y-3">
                  <p className="text-sm text-gray-700">
                    Upload dataset (<code>.pkl</code> file containing{" "}
                    <code>X</code> and <code>y</code>).
                  </p>

                  <input
                    id="dataset-file"
                    type="file"
                    accept=".pkl"
                    className="hidden"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) handleFileUpload(f);
                    }}
                  />

                  <div className="border border-dashed border-gray-300 rounded-lg p-4 flex items-center justify-between gap-4">
                    <div className="text-xs text-gray-600">
                      {uploadedFileName ? (
                        <>
                          <div className="font-medium text-gray-800">
                            Selected file
                          </div>
                          <div>{uploadedFileName}</div>
                        </>
                      ) : (
                        <div>No file selected</div>
                      )}
                    </div>
                    <label htmlFor="dataset-file" className={primaryButton}>
                      Choose file
                    </label>
                  </div>

                  {uploadInfo && (
                    <div className="text-xs text-gray-700 border border-gray-200 rounded-md p-3 bg-gray-50">
                      <div>{uploadInfo.message}</div>
                      <div className="mt-1">
                        Samples: <strong>{uploadInfo.n_samples}</strong> ·
                        Features: <strong>{uploadInfo.n_features}</strong>
                      </div>
                      {uploadInfo.y_min !== undefined &&
                        uploadInfo.y_max !== undefined && (
                          <div className="mt-1">
                            Target range:{" "}
                            <strong>{uploadInfo.y_min.toFixed(3)}</strong> to{" "}
                            <strong>{uploadInfo.y_max.toFixed(3)}</strong>
                          </div>
                        )}

                      {uploadInfo.preview_X &&
                        uploadInfo.preview_y &&
                        uploadInfo.preview_X.length > 0 && (
                          <div className="mt-3">
                            <div className="mb-1 font-medium">
                              Preview (first 5 rows):
                            </div>
                            <div className="overflow-auto">
                              <table className="text-xs border-collapse border border-gray-300">
                                <thead>
                                  <tr>
                                    <th className="border px-2 py-1 bg-gray-50">
                                      x1
                                    </th>
                                    <th className="border px-2 py-1 bg-gray-50">
                                      x2
                                    </th>
                                    <th className="border px-2 py-1 bg-gray-50">
                                      x3
                                    </th>
                                    <th className="border px-2 py-1 bg-gray-50">
                                      x4
                                    </th>
                                    <th className="border px-2 py-1 bg-gray-50">
                                      x5
                                    </th>
                                    <th className="border px-2 py-1 bg-gray-50">
                                      y
                                    </th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {uploadInfo.preview_X
                                    .slice(0, 5)
                                    .map((row, i) => (
                                      <tr key={i}>
                                        {row.map((val, j) => (
                                          <td
                                            key={j}
                                            className="border px-2 py-1"
                                          >
                                            {val.toFixed(3)}
                                          </td>
                                        ))}
                                        <td className="border px-2 py-1">
                                          {uploadInfo.preview_y![i]?.toFixed(
                                            3
                                          )}
                                        </td>
                                      </tr>
                                    ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}
                    </div>
                  )}
                </div>
              )}

              {step === 2 && (
                <div className="space-y-3">
                  <p className="text-sm text-gray-700">
                    Configure hyperparameters and train the model on the
                    uploaded dataset.
                  </p>

                  <div className="grid grid-cols-2 gap-3 mb-4">
                    <div className="space-y-1">
                      <label className="text-xs text-gray-600">
                        Learning rate
                      </label>
                      <input
                        type="number"
                        step="any"
                        value={learningRate}
                        onChange={(e) => setLearningRate(e.target.value)}
                        className="w-full border border-gray-300 rounded-md px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-black"
                        placeholder="0.001"
                      />
                    </div>

                    <div className="space-y-1">
                      <label className="text-xs text-gray-600">
                        Max epochs
                      </label>
                      <input
                        type="number"
                        value={maxEpochs}
                        onChange={(e) => setMaxEpochs(e.target.value)}
                        className="w-full border border-gray-300 rounded-md px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-black"
                        placeholder="200"
                      />
                    </div>

                    <div className="space-y-1">
                      <label className="text-xs text-gray-600">
                        Batch size
                      </label>
                      <input
                        type="number"
                        value={batchSize}
                        onChange={(e) => setBatchSize(e.target.value)}
                        className="w-full border border-gray-300 rounded-md px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-black"
                        placeholder="64"
                      />
                    </div>

                    <div className="space-y-1">
                      <label className="text-xs text-gray-600">
                        Hidden layers
                      </label>
                      <input
                        type="text"
                        value={hiddenLayersText}
                        onChange={(e) => setHiddenLayersText(e.target.value)}
                        className="w-full border border-gray-300 rounded-md px-2 py-1 text-sm focus:outline-none focus:ring-1 focus:ring-black"
                        placeholder="64,32,16"
                      />
                    </div>
                  </div>

                  <button
                    onClick={handleTrain}
                    disabled={!datasetUploaded || training}
                    className={primaryButton}
                  >
                    {training
                      ? "Training..."
                      : trainInfo
                      ? "Train again"
                      : "Train model"}
                  </button>

                  {trainInfo && (
                    <div className="mt-3 text-xs border border-gray-200 rounded-md p-3 bg-gray-50 text-gray-800">
                      <div className="font-medium mb-1">
                        {trainInfo.message}
                      </div>
                      <div>
                        Train: {trainInfo.train_samples} · Val:{" "}
                        {trainInfo.val_samples} · Test:{" "}
                        {trainInfo.test_samples}
                      </div>
                      <div>
                        Validation RMSE:{" "}
                        {Math.sqrt(trainInfo.val_mse).toFixed(4)}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {step === 3 && (
                <div className="space-y-4">
                  <p className="text-sm text-gray-700">
                    Use the sliders to select feature values, then run a
                    prediction.
                  </p>

                  <form onSubmit={handleSubmit} className="space-y-4">
                    <div className="space-y-3">
                      {(["v1", "v2", "v3", "v4", "v5"] as const).map(
                        (key, idx) => {
                          const { min, max } = getSliderBounds(idx);
                          const valueStr = values[key];
                          const valueNum =
                            valueStr !== ""
                              ? parseFloat(valueStr)
                              : (min + max) / 2;

                          return (
                            <div key={key} className="space-y-1">
                              <div className="flex justify-between text-xs">
                                <span className="text-gray-600">
                                  Feature {idx + 1}
                                </span>
                                <span className="text-gray-800 font-medium">
                                  {valueNum.toFixed(2)}
                                </span>
                              </div>
                              <input
                                type="range"
                                min={min}
                                max={max}
                                step="any"
                                value={valueNum}
                                onChange={(e) =>
                                  handleChange(key, parseFloat(e.target.value).toFixed(2))
                                }
                                className="w-full"
                              />
                              <div className="flex justify-between text-[10px] text-gray-500">
                                <span>min: {min.toFixed(2)}</span>
                                <span>max: {max.toFixed(2)}</span>
                              </div>
                            </div>
                          );
                        }
                      )}
                    </div>

                    <button
                      type="submit"
                      disabled={loading || !trained}
                      className={primaryButton}
                    >
                      {loading ? "Running..." : "Run prediction"}
                    </button>
                  </form>

                  {result && (
                    <div className="border-2 border-green-500 rounded-lg p-4 bg-green-50">
                      <div className="text-xs text-gray-600 mb-2">Inputs</div>
                      <div className="mb-4 text-xs font-mono text-gray-700">
                        [{result.input.map((v) => v.toFixed(2)).join(", ")}]
                      </div>
                      <div className="text-xs text-gray-600 mb-1">
                        Prediction
                      </div>
                      <div className="text-3xl font-bold text-green-700">
                        {result.prediction.toFixed(4)}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
