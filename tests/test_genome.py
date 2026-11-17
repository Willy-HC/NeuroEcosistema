
import numpy as np
import pytest

from neuroecosistema.neural_network.network import NeuralNetwork


def create_network(seed=42):
    return NeuralNetwork(
        input_size=8,
        hidden_layers=[16, 12],
        output_size=2,
        seed=seed,
    )


def test_genome_has_correct_size():
    network = create_network()

    genome = network.get_genome()

    assert genome.shape == (374,)
    assert network.count_parameters() == 374


def test_genome_can_be_restored():
    network = create_network()

    original_genome = network.get_genome()

    network.set_genome(original_genome)

    restored_genome = network.get_genome()

    np.testing.assert_array_equal(
        original_genome,
        restored_genome,
    )


def test_genome_changes_network_parameters():
    network = create_network()

    original_genome = network.get_genome()

    modified_genome = original_genome.copy()
    modified_genome[0] += 0.5

    network.set_genome(modified_genome)

    assert network.get_genome()[0] == pytest.approx(
        original_genome[0] + 0.5
    )


def test_genome_rejects_incorrect_size():
    network = create_network()

    with pytest.raises(ValueError):
        network.set_genome(np.zeros(100))


def test_genome_rejects_non_finite_values():
    network = create_network()

    genome = network.get_genome()
    genome[0] = np.nan

    with pytest.raises(ValueError):
        network.set_genome(genome)


def test_genome_preserves_network_output():
    network = create_network()

    inputs = np.ones(8, dtype=np.float32)

    original_output = network.forward(inputs)

    genome = network.get_genome()
    network.set_genome(genome)

    restored_output = network.forward(inputs)

    np.testing.assert_array_equal(
        original_output,
        restored_output,
    )