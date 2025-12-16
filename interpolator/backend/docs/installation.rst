Installation
============

Requirements
------------

- Docker and Docker Compose (recommended)
- Git (to clone the repository)
- For manual install (without Docker):
  
  - Python 3.12 or higher
  - Node.js 20.x or higher


Quick start (Docker - recommended)
----------------------------------

Clone the repository and move into it::

    git clone <your-repo-url>
    cd interpolator

Make sure Docker is running, then run::

    ./scripts/docker-start.sh

This script uses ``docker compose`` to build and start both the backend and frontend.

Once it finishes:

- Frontend UI: ``http://localhost:3000/``
- Backend API docs: ``http://localhost:8000/docs``

To stop the containers::

    ./scripts/docker-stop.sh


Manual setup (without Docker)
-----------------------------

If you prefer to run locally without Docker, you can start backend and frontend
separately.

**Backend (FastAPI)**

.. code-block:: bash

    cd backend

    # Create and activate virtual environment
    python3.12 -m venv .venv
    source .venv/bin/activate

    # Install package
    pip install -e .

    # Run API
    uvicorn main:app --host 0.0.0.0 --port 8000

**Faster Installation (Recommended)**

The Docker build automatically installs the CPU-only version of PyTorch
instead of the full GPU version for faster builds.
You can do the same if you would like to save some time:

.. code-block:: bash

    # Install PyTorch CPU-only first
    pip install torch --index-url https://download.pytorch.org/whl/cpu

    # Then install the rest of the dependencies
    pip install -e .


**Frontend (Next.js)**

.. code-block:: bash

    cd frontend
    npm install
    npm run dev

.. note::

   Make sure the backend is running before starting the frontend.

The application will then be available at:

- Frontend UI: ``http://localhost:3000``
- Backend API: ``http://localhost:8000``
