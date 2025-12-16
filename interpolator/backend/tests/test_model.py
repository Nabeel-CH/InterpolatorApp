# test all model.py functions
# Note: NeuralNetwork uses MLP class so these tests also test MLP class by default

import numpy as np
from fivedreg.model import NeuralNetwork


def test_neural_network_init():
    """
    Test init of NeuralNetwork
    """
    model = NeuralNetwork(
        input_dim=5,
        hidden_layers=[10, 5],
        learning_rate=1e-3,
        max_epochs=10,
        batch_size=8,
    )
    assert model.input_dim == 5
    assert model.hidden_layers == [10, 5]
    assert model.learning_rate == 1e-3
    assert model.max_epochs == 10
    assert model.batch_size == 8


def test_neural_network():
    """
    Train on a simple dataset and check prediction output is in correct format
    """
    # create dataset
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5)).astype("float32")
    y = X.sum(axis=1).astype("float32")  

    model = NeuralNetwork(
        input_dim=5,
        hidden_layers=[8, 4],
        learning_rate=1e-3,
        max_epochs=5,  
        batch_size=16,
    )

    model.fit(X, y)
    predictions = model.predict(X)

    assert predictions.shape == (50,)
