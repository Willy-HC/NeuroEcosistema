
import numpy as np
import pytest

from neuroecosistema.evolution.coevolution_trainer import (
    CoevolutionTrainer,
)
from neuroecosistema.evolution.episode_evaluator import (
    EpisodeEvaluator,
)


@pytest.fixture
def configs():
    simulation = {
        "simulation": {
            "width": 500,
            "height": 400,
            "max_steps": 20,
        },
        "population": {
            "prey": 4,
            "predators": 4,
        },
        "environment": {
            "initial_food": 10,
            "food_respawn": 5,
        },
        "energy": {
            "food_reward": 20,
            "capture_reward": 30,
            "movement_cost": 0.1,
        },
    }

    neural = {
        "input_size": 8,
        "hidden_layers": [16, 12],
        "output_size": 2,
        "activation_hidden": "relu",
        "activation_output": "tanh",
    }

    evolution = {
        "population_size": 4,
        "generations": 2,
        "elitism": 1,
        "mutation_rate": 0.1,
        "mutation_strength": 0.15,
        "crossover_rate": 0.7,
        "selection": "tournament",
        "seed": 42,
    }

    return simulation, neural, evolution


def create_trainer(configs):
    simulation, neural, evolution = configs

    return CoevolutionTrainer(
        simulation_config=simulation,
        neural_config=neural,
        evolution_config=evolution,
        evaluation_seeds=(42, 123),
    )


def test_initial_populations_have_correct_size(configs):
    trainer = create_trainer(configs)

    prey = trainer.prey_trainer.create_initial_population()
    predators = (
        trainer.predator_trainer.create_initial_population()
    )

    assert len(prey) == 4
    assert len(predators) == 4


def test_joint_evaluation_returns_two_fitness_arrays(configs):
    trainer = create_trainer(configs)

    prey = trainer.prey_trainer.create_initial_population()
    predators = (
        trainer.predator_trainer.create_initial_population()
    )

    prey_fitness, predator_fitness = (
        trainer.evaluate_populations(prey, predators)
    )

    assert prey_fitness.shape == (4,)
    assert predator_fitness.shape == (4,)

    assert np.all(np.isfinite(prey_fitness))
    assert np.all(np.isfinite(predator_fitness))


def test_joint_evaluation_rejects_different_sizes(configs):
    trainer = create_trainer(configs)
    simulation, neural, _ = configs

    prey = trainer.prey_trainer.create_initial_population()
    predators = (
        trainer.predator_trainer.create_initial_population()
    )

    evaluator = EpisodeEvaluator(
        simulation_config=simulation,
        neural_config=neural,
        seed=42,
    )

    with pytest.raises(ValueError):
        evaluator.evaluate_both(
            prey,
            predators[:2],
        )


def test_joint_evaluation_is_reproducible(configs):
    trainer = create_trainer(configs)

    prey = trainer.prey_trainer.create_initial_population()
    predators = (
        trainer.predator_trainer.create_initial_population()
    )

    first_prey, first_predators = (
        trainer.evaluate_populations(prey, predators)
    )

    second_prey, second_predators = (
        trainer.evaluate_populations(prey, predators)
    )

    np.testing.assert_array_equal(
        first_prey,
        second_prey,
    )

    np.testing.assert_array_equal(
        first_predators,
        second_predators,
    )


def test_coevolution_trains_both_species(configs):
    trainer = create_trainer(configs)

    prey_genome, predator_genome = trainer.train()

    assert prey_genome.ndim == 1
    assert predator_genome.ndim == 1

    assert prey_genome.size == predator_genome.size
    assert len(trainer.prey_trainer.history) == 2
    assert len(trainer.predator_trainer.history) == 2

    assert np.isfinite(trainer.prey_trainer.best_fitness)
    assert np.isfinite(trainer.predator_trainer.best_fitness)


def test_coevolution_saves_both_models(configs, tmp_path):
    trainer = create_trainer(configs)
    trainer.train()

    paths = trainer.save_results(tmp_path)

    for species in ("prey", "predator"):
        assert paths[species]["model"].exists()
        assert paths[species]["history"].exists()
        assert paths[species]["metadata"].exists()

        with np.load(paths[species]["model"]) as data:
            assert data["genome"].ndim == 1


def test_coevolution_keeps_historical_best(configs):
    trainer = create_trainer(configs)
    trainer.train()

    for species_trainer in (
        trainer.prey_trainer,
        trainer.predator_trainer,
    ):
        recorded_best = max(
            row["best_fitness"]
            for row in species_trainer.history
        )

        assert species_trainer.best_fitness == pytest.approx(
            recorded_best
        )