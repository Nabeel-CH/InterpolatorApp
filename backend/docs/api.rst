API Reference
=============

This page documents the **Python Modules** and **API** endpoints.

----

Python Modules
--------------

fivedreg.data
~~~~~~~~~~~~~

.. automodule:: fivedreg.data
   :members:
   :undoc-members:
   :show-inheritance:

----

fivedreg.model
~~~~~~~~~~~~~~

.. automodule:: fivedreg.model
   :members:
   :undoc-members:
   :show-inheritance:

----

API Endpoints
------------------

Base URL: ``http://localhost:8000``

.. list-table:: Endpoint Summary
   :header-rows: 1
   :widths: 15 15 70

   * - Method
     - Path
     - Description
   * - GET
     - ``/``
     - Welcome message
   * - GET
     - ``/health``
     - Health check
   * - POST
     - ``/upload``
     - Upload dataset (.pkl file)
   * - POST
     - ``/train``
     - Train neural network
   * - POST
     - ``/predict``
     - Make prediction

----

GET /
~~~~~

Welcome endpoint to verify the backend is running.

**Response:**

.. code-block:: json

    {"message": "Interpolator backend is running :)"}

----

GET /health
~~~~~~~~~~~

Health check endpoint.

**Response:**

.. code-block:: json

    {"status": "ok"}

----

POST /upload
~~~~~~~~~~~~

Upload dataset in ``.pkl`` format.

**Request:**

- Content-Type: ``multipart/form-data``
- Field: ``file`` — a ``.pkl`` file containing:

  - Dict: ``{"X": array, "y": array}``
  - Or tuple: ``(X, y, ...)``

  Where ``X`` has shape ``(n_samples, 5)`` and ``y`` has shape ``(n_samples,)``.

**Response (200):**

.. code-block:: json

    {
      "message": "Dataset uploaded successfully.",
      "n_samples": 5000,
      "n_features": 5,
      "feature_mins": [0.0, 0.0, 0.0, 0.0, 0.0],
      "feature_maxs": [1.0, 1.0, 1.0, 1.0, 1.0],
      "y_min": -1.14,
      "y_max": 1.62,
      "preview_X": [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
      "preview_y": [0.1, 0.2]
    }

**Errors:**

- ``400``: Invalid file (wrong extension, shape, or missing keys)

----

POST /train
~~~~~~~~~~~

Train the neural network on the uploaded dataset.

**Request:**

- Content-Type: ``application/json``

.. code-block:: json

    {
      "hidden_layers": [64, 32, 16],
      "learning_rate": 0.001,
      "max_epochs": 200,
      "batch_size": 64
    }

All fields are optional (defaults shown above).

**Response (200):**

.. code-block:: json

    {
      "message": "Model trained successfully.",
      "train_samples": 3000,
      "val_samples": 1000,
      "test_samples": 1000,
      "val_mse": 0.00305
    }

**Errors:**

- ``400``: No dataset uploaded (call ``/upload`` first)

----

POST /predict
~~~~~~~~~~~~~

Make a prediction for a 5D input.

**Request:**

- Content-Type: ``application/json``

.. code-block:: json

    {
      "v1": 0.1,
      "v2": 0.2,
      "v3": 0.3,
      "v4": 0.4,
      "v5": 0.5
    }

**Response (200):**

.. code-block:: json

    {
      "input": [0.1, 0.2, 0.3, 0.4, 0.5],
      "prediction": 0.1234
    }

**Errors:**

- ``400``: Model not trained (call ``/train`` first)
