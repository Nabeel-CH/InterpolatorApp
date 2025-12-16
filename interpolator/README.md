# InterpolatorApp

A 5D Interpolator app that takes in a 5D dataset, trains a neural network, and then allows the user to predict values for given 5 features. Uses FastAPI backend + Next.js frontend.

## Prerequisites

- Python: 3.12 or higher
- Node.js: 20.x or higher 
- Docker


## 1. Running the app

### 1.1 Recommended: Docker + scripts

Make sure docker is running. 

From the repository root:

```bash
./scripts/docker-start.sh
```

Then open:

- Frontend UI: http://localhost:3000/
- Backend API docs: http://localhost:8000/docs

To stop:

```bash
./scripts/docker-stop.sh
```


### 1.2 Manual (no Docker)

#### Backend

```bash
cd backend

# Create & activate virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install package
pip install -e .

# Run API
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Faster Installation (Recommended):**

The Docker build automatically installs the CPU-only version of PyTorch instead of the full GPU version for faster builds. You can do the same for manual installation:

```bash
# Install PyTorch CPU-only first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install the rest of the dependencies
pip install -e .
```

#### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev   
```

**Note:** Make sure the backend is running before starting the frontend.

---

## 2. Environment variables

### Backend

No custom environment variables are required for local development. The backend listens on port **8000**.

**Docker environment variables** (set automatically):
- `PYTHONPATH=/app` - Python path configuration
- `PYTHONUNBUFFERED=1` - Ensures Python output is not buffered

### Frontend

- `NEXT_PUBLIC_API_URL`
  - URL of the FastAPI backend
  - **Default:** `http://localhost:8000`
  - **Docker:** Set to `http://localhost:8000` in `docker-compose.yml` (note: uses `localhost` because frontend runs in browser, which connects to host machine's port 8000)
  - **Custom:** Set this variable if your backend runs on a different host/port
  - **Usage:** Create a `.env.local` file in `frontend/` directory:
    ```bash
    NEXT_PUBLIC_API_URL=http://your-backend-url:8000
    ```

---

## 3. Usage (UI steps)

### Step 1 – Upload

Dataset format (for upload)

The app expects a `.pkl` file containing:

```python
{"X": X_array, "y": y_array}
# or
(X_array, y_array, ...)
```

where:

- `X_array` is a 2D array with shape `(n_samples, 5)`
- `y_array` is a 1D array with shape `(n_samples,)`
- `n_samples` must match between `X` and `y`.


- Upload your `.pkl` file 
- The app will:
  - Validates shapes and types
  - Shows number of samples and features
  - Shows first 5 rows of `X` and `y`
  - Update min/max of sliders (based on features)
  - Shows target min/max

### Step 2 – Train

- Configure hyperparameters in the UI:
  - Learning rate
  - Max epochs
  - Batch size
  - Hidden layers (comma-separated, e.g. `64,32,16`)
- Click **“Train model”** (can be pressed again to retrain with new hyperparameters).


The app will: 
1. Handle missing values:
   - Drop rows where `y` is missing  
   - Replace missing `X` values with column means
2. Train/val/test split:
   - 60% train, 20% validation, 20% test
3. Standardise features 
4. Train a feedforward neural network 

The UI displays:

- Train / validation / test sizes  
- Validation RMSE

### Step 3 – Predict

- Use five sliders (one per feature):
  - Ranges come from min/max of the uploaded dataset.
- Click **“Run prediction”** to get a prediction for that 5D input.

---

## 4. Tests

(see the documentaion page for more details)


### 4.1 Run via script (recommended)

From the **repo root**:

```bash
./scripts/run-tests.sh
```

This script:

- Uses `docker compose run --rm backend pytest -sv tests` if Docker is available
- Otherwise runs `pytest -sv tests` locally inside `backend/`.

### 4.2 Run manually

**With Docker:**

```bash
docker compose run --rm backend pytest -sv tests
```

**Locally:**

```bash
cd backend
pytest -sv tests
```

---

## 5. Documentation (Sphinx)

See Sphinx documentation for details on:

- Installation & setup
- User Guide
- Practical Usage Examples
- API reference 
- Description of the test suite
- Performances and profiling 

### Build docs

From the **repo root**:

```bash
./scripts/build_docs.sh
```

This will use docker container to build docs if available, otherwise it will build locally.

HTML is generated in:

```text
backend/docs/_build/html/
```

Open `index.html` using the generated URL, example:

```text
file:///…/backend/docs/_build/html/index.html
```

If using WSL Linux the URL will open like in the following example: 
```text
file://wsl.localhost/Ubuntu/home/nabeel/projects/c1/InterpolatorApp/backend/docs/_build/html/index.html
```

---

## Project Structure

```
InterpolatorApp/
├── backend/                     # FastAPI backend
│   ├── fivedreg/               # Main package
│   │   ├── __init__.py
│   │   ├── data.py             # Data loading and preprocessing
│   │   └── model.py            # Neural network model
│   ├── analysis/               # Benchmarking scripts
│   │   ├── benchmark.py        # Performance benchmarks
│   │   ├── plot_results.py     # Generate benchmark plots
│   │   └── plots/              # Generated plots
│   ├── tests/                  # Test suite
│   │   ├── test_data.py
│   │   ├── test_model.py
│   │   └── test_main.py
│   ├── docs/                   # Sphinx documentation
│   ├── main.py                 # FastAPI application
│   ├── pyproject.toml          # Python dependencies
│   └── Dockerfile
├── frontend/                   # Next.js frontend
│   ├── src/app/
│   │   ├── page.tsx            # Main UI component
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── package.json            # Node dependencies
│   └── Dockerfile
├── scripts/                    # Helper scripts
│   ├── docker-start.sh         # Start Docker containers
│   ├── docker-stop.sh          # Stop Docker containers
│   ├── docker-logs.sh          # View container logs
│   ├── run-tests.sh            # Run test suite
│   ├── run-benchmark.sh        # Run benchmarks
│   └── build_docs.sh           # Build Sphinx docs
├── docker-compose.yml          # Docker Compose (production)
├── docker-compose.override.yml # Docker Compose (development)
├── LICENSE
└── README.md
```

---

### Declaration of Use of Autogeneration Tools

Claude console was used in the following cases:
- Write docker files by passing Calude the provided example files to use as a template 
- Provide suggestions on how to make the docker build faster and then I implemented one of them - which was to only download the CPU version of torch
- Help write shell script files that were useful during development 
- Make edits to page.tsx for the frontend UI 

Copilot was used in the following cases:
- Copilot auto-suggestion was used when it's auto suggestions were relevant. This help speed up writing commentary,docstrings and to fix spelling/grammer errors within python files and sphinx documentation pages


