
import numpy as np
import pytest

from neuroecosistema.evolution.episode_evaluator import (
    EpisodeEvaluator,
)
from neuroecosistema.neural_network.network import NeuralNetwork

SIMULATION_CONFIG = {
    "simulation": {
        "width": 500,
        "height": 400,
        "max_steps": 100,
    },
    "population": {
        "prey": 5,
        "predators": 2,
    },
    "environment": {
        "initial_food": 20,
    },
    "energy": {
        "food_reward": 20,
        "capture_reward": 30,
        "movement_cost": 0.1,
    },
}

NEURAL_CONFIG = {
    "input_size": 8,
    "hidden_layers": [16, 12],
    "output_size": 2,
    "activation_hidden": "relu",
    "activation_output": "tanh",
}


def create_genomes(quantity=5):
    genomes = []

    for seed in range(quantity):
        network = NeuralNetwork(
            input_size=8,
            hidden_layers=[16, 12],
            output_size=2,
            seed=seed,
        )
        genomes.append(network.get_genome())

    return genomes


def test_episode_returns_fitness_for_every_genome():
    evaluator = EpisodeEvaluator(
        SIMULATION_CONFIG,
        NEURAL_CONFIG,
    )

    fitness = evaluator.evaluate(
        create_genomes(5),
        species="prey",
    )

    assert fitness.shape == (5,)


def test_episode_fitness_is_finite():
    evaluator = EpisodeEvaluator(
        SIMULATION_CONFIG,
        NEURAL_CONFIG,
    )

    fitness = evaluator.evaluate(
        create_genomes(5),
        species="predator",
    )

    assert np.all(np.isfinite(fitness))


def test_episode_is_reproducible():
    genomes = create_genomes(5)

    evaluator_a = EpisodeEvaluator(
        SIMULATION_CONFIG,
        NEURAL_CONFIG,
        seed=42,
    )

    evaluator_b = EpisodeEvaluator(
        SIMULATION_CONFIG,
        NEURAL_CONFIG,
        seed=42,
    )

    result_a = evaluator_a.evaluate(genomes)
    result_b = evaluator_b.evaluate(genomes)

    np.testing.assert_array_equal(
        result_a,
        result_b,
    )


def test_episode_rejects_invalid_species():
    evaluator = EpisodeEvaluator(
        SIMULATION_CONFIG,
        NEURAL_CONFIG,
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            create_genomes(5),
            species="unknown",
        )


def test_episode_rejects_empty_population():
    evaluator = EpisodeEvaluator(
        SIMULATION_CONFIG,
        NEURAL_CONFIG,
    )

    with pytest.raises(ValueError):
        evaluator.evaluate([], species="prey")
