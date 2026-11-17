
import numpy as np
import pytest

from neuroecosistema.neural_network.model_loader import (
    load_trained_network,
)
from neuroecosistema.neural_network.network import NeuralNetwork

NEURAL_CONFIG = {
    "input_size": 8,
    "hidden_layers": [16, 12],
    "output_size": 2,
    "activation_hidden": "relu",
    "activation_output": "tanh",
}


def create_network(seed=42):
    return NeuralNetwork(
        input_size=8,
        hidden_layers=[16, 12],
        output_size=2,
        seed=seed,
    )


def test_load_trained_network_restores_genome(tmp_path):
    original = create_network()
    model_path = tmp_path / "best_prey.npz"

    np.savez_compressed(
        model_path,
        genome=original.get_genome(),
    )

    loaded = load_trained_network(
        model_path,
        NEURAL_CONFIG,
    )

    np.testing.assert_array_equal(
        loaded.get_genome(),
        original.get_genome(),
    )


def test_loaded_network_produces_same_output(tmp_path):
    original = create_network()
    model_path = tmp_path / "best_prey.npz"

    np.savez_compressed(
        model_path,
        genome=original.get_genome(),
    )

    loaded = load_trained_network(
        model_path,
        NEURAL_CONFIG,
    )

    inputs = np.ones(8)

    np.testing.assert_allclose(
        loaded.forward(inputs),
        original.forward(inputs),
    )


def test_missing_model_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_trained_network(
            tmp_path / "missing.npz",
            NEURAL_CONFIG,
        )


def test_invalid_model_raises_error(tmp_path):
    model_path = tmp_path / "invalid.npz"

    np.savez_compressed(
        model_path,
        incorrect=np.ones(374),
    )

    with pytest.raises(ValueError):
        load_trained_network(
            model_path,
            NEURAL_CONFIG,
        )


def test_incompatible_genome_raises_error(tmp_path):
    model_path = tmp_path / "incompatible.npz"

    np.savez_compressed(
        model_path,
        genome=np.ones(100),
    )

    with pytest.raises(ValueError):
        load_trained_network(
            model_path,
            NEURAL_CONFIG,
        )