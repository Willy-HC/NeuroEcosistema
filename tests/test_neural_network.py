
import numpy as np
import pytest

from neuroecosistema.neural_network.activations import (
    get_activation,
    relu,
    tanh,
)
from neuroecosistema.neural_network.network import NeuralNetwork


def create_network(seed=42):
    return NeuralNetwork(
        input_size=8,
        hidden_layers=[16, 12],
        output_size=2,
        hidden_activation="relu",
        output_activation="tanh",
        seed=seed,
    )


def test_network_has_correct_architecture():
    network = create_network()

    assert len(network.weights) == 3
    assert network.weights[0].shape == (16, 8)
    assert network.weights[1].shape == (12, 16)
    assert network.weights[2].shape == (2, 12)


def test_network_returns_two_outputs():
    network = create_network()

    inputs = np.zeros(8, dtype=np.float32)
    outputs = network.forward(inputs)

    assert outputs.shape == (2,)


def test_network_outputs_are_between_minus_one_and_one():
    network = create_network()

    inputs = np.random.default_rng(10).random(8)
    outputs = network.forward(inputs)

    assert np.all(outputs >= -1.0)
    assert np.all(outputs <= 1.0)


def test_network_is_reproducible_with_same_seed():
    network_a = create_network(seed=42)
    network_b = create_network(seed=42)

    inputs = np.ones(8, dtype=np.float32)

    output_a = network_a.forward(inputs)
    output_b = network_b.forward(inputs)

    np.testing.assert_array_equal(output_a, output_b)


def test_network_rejects_incorrect_input_size():
    network = create_network()

    with pytest.raises(ValueError):
        network.forward(np.zeros(5))


def test_relu_activation():
    values = np.array([-2.0, 0.0, 3.0])

    result = relu(values)

    np.testing.assert_array_equal(
        result,
        np.array([0.0, 0.0, 3.0]),
    )


def test_tanh_activation():
    values = np.array([-100.0, 0.0, 100.0])

    result = tanh(values)

    assert np.all(result >= -1.0)
    assert np.all(result <= 1.0)


def test_invalid_activation_raises_error():
    with pytest.raises(ValueError):
        get_activation("sigmoid")
