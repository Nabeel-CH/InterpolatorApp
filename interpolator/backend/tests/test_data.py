# test data.py functions

import pickle
import numpy as np
import pytest

from fivedreg.data import (
    load_dataset,
    handle_missing_values,
    split_and_standardise,
    load_from_bytes
)


def test_load_dataset(tmp_path):
    """
    load_dataset should read a .pkl file containing X (5 features) and y,
    and return train/val/test splits with correct shapes 
    """
    # Create fake dataset
    X = np.random.rand(100, 5)
    y = np.random.rand(100)

    data = {"X": X, "y": y}
    pkl_path = tmp_path / "test_dataset.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(data, f)

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_mean,
        feature_std,
    ) = load_dataset(str(pkl_path))

    # Total rows should add up
    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == 100
    assert y_train.shape[0] + y_val.shape[0] + y_test.shape[0] == 100

    # Each split shoudl have 5 features
    assert X_train.shape[1] == 5
    assert X_val.shape[1] == 5
    assert X_test.shape[1] == 5

    # Means/stds should match number of features
    assert feature_mean.shape[0] == 5
    assert feature_std.shape[0] == 5

def test_load_dataset_fails(tmp_path):
    """
    If X does not have 5 features, load_dataset should fail
    """
    # create dataset with wrong number of features
    X = np.random.rand(50, 3)  
    y = np.random.rand(50)

    # save dataset to .pkl file
    pkl_path = tmp_path / "bad_dataset.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)

    with pytest.raises(ValueError):
        load_dataset(str(pkl_path))


def test_handle_missing_values():
    """
    Rows where y is NaN should be removed and 
    NaNs in X should be replaced with the column mean
    """
    # create dataset with NaNs
    X = np.array([
        [1.0, np.nan, 3.0, 4.0, 5.0],
        [np.nan, 2.0, 3.0, 4.0, 5.0],
        [9.0, 8.0, 7.0, 6.0, 5.0],
    ])
    y = np.array([1.0, np.nan, 3.0])


    # X after dropping rows where y is NaN
    X_after_drop = X[[0, 2]]  # rows 0 and 3
    # Column means ignoring NaNs
    col_means = np.nanmean(X_after_drop, axis=0)

    X_clean, y_clean = handle_missing_values(X, y)

    # Row with NaN in y should be dropped 
    # one row should be dropped due to NaN in y
    assert y_clean.shape == (2,)
    assert X_clean.shape == (2, 5)
    # check no NaNs in X
    assert not np.isnan(X_clean).any()

    # check the missing value in X now equals the column mean
    assert np.isclose(X_clean[0, 1], col_means[1])

    # Other non-missing entries for that row should be unchanged
    assert np.allclose(X_clean[0, [0, 2, 3, 4]], X_after_drop[0, [0, 2, 3, 4]])


def test_split_and_standardise():
    """
    split_and_standardise should:
    - split into train/val/test
    - return feature mean/std consistent with X
    - standardise training data to approx 0 mean and approx 1 std
    """
    # create dataset
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    y = rng.normal(size=(120,))

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_mean,
        feature_std,
    ) = split_and_standardise(X, y, random_state=0)

    # check X dimensions
    assert X_train.shape[1] == 5
    assert X_val.shape[1] == 5
    assert X_test.shape[1] == 5

    # check y dimensions
    assert y_train.ndim == 1
    assert y_val.ndim == 1
    assert y_test.ndim == 1

    # check means/std arrays length
    assert feature_mean.shape == (5,)
    assert feature_std.shape == (5,)

    # Check that train data has approx 0 mean and approx 1 std 
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0, ddof=0)

    np.testing.assert_allclose(train_mean, np.zeros(5), rtol=1e-1, atol=1e-1)
    np.testing.assert_allclose(train_std, np.ones(5), rtol=1e-1, atol=1e-1)

def test_load_from_bytes(tmp_path):
    """
    test load_from_bytes with correct dataset
    """
    # create dataset
    X = np.zeros((10,5))
    y = np.arange(10.0)
    b = pickle.dumps({"X": X, "y": y})

    X2, y2 = load_from_bytes(b)

    # check shapes
    assert X2.shape == (10,5)
    assert (y2 == y).all()

def test_load_from_bytes_fails():
    """
    test load_from_bytes with invalid dataset
    """
    # create invalid dataset
    bad_bytes = pickle.dumps({"foo": 1})
    with pytest.raises(ValueError):
        load_from_bytes(bad_bytes)
