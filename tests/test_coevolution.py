
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


def make_diagnostics(fitness_values, rotation):
    diagnostics = []

    for genome_index, fitness in enumerate(fitness_values):
        diagnostics.append(
            {
                "genome_index": genome_index,
                "survival": rotation + genome_index,
                "food": 10 * rotation + genome_index,
                "exploration": 2 * rotation + genome_index,
                "stagnation": -3 * rotation - genome_index,
                "fitness": fitness,
                "food_eaten": 100 * rotation + genome_index,
                "steps_survived": 200 * rotation + genome_index,
                "visited_regions": 300 * rotation + genome_index,
                "stagnation_steps": 400 * rotation + genome_index,
            }
        )

    return list(reversed(diagnostics))


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


def test_joint_diagnostics_average_per_genome_and_rotation(
    configs,
    monkeypatch,
):
    trainer = create_trainer(configs)
    prey_genomes = trainer.prey_trainer.create_initial_population()
    predator_genomes = (
        trainer.predator_trainer.create_initial_population()
    )
    evaluations = []

    def fake_evaluate_both(
        evaluator,
        evaluated_prey,
        evaluated_predators,
        rotation=0,
        include_diagnostics=False,
    ):
        evaluations.append(
            (evaluator.seed, rotation, include_diagnostics)
        )
        prey_fitness = np.asarray(
            [
                0.01 * rotation + genome_index * 0.001
                for genome_index in range(len(evaluated_prey))
            ]
        )
        predator_fitness = np.asarray(
            [
                0.03 * rotation + genome_index * 0.002
                for genome_index in range(len(evaluated_predators))
            ]
        )

        if include_diagnostics:
            return (
                prey_fitness,
                predator_fitness,
                make_diagnostics(prey_fitness, rotation),
            )

        return prey_fitness, predator_fitness

    monkeypatch.setattr(
        EpisodeEvaluator,
        "evaluate_both",
        fake_evaluate_both,
    )

    default_prey, default_predators = trainer.evaluate_populations(
        prey_genomes,
        predator_genomes,
        generation=5,
    )
    evaluations.clear()

    prey_fitness, predator_fitness, diagnostics = (
        trainer.evaluate_populations(
            prey_genomes,
            predator_genomes,
            generation=5,
            include_diagnostics=True,
        )
    )

    np.testing.assert_array_equal(prey_fitness, default_prey)
    np.testing.assert_array_equal(predator_fitness, default_predators)
    assert evaluations == [(42, 5, True), (123, 6, True)]
    assert len(diagnostics) == len(prey_genomes)
    assert [row["genome_index"] for row in diagnostics] == list(
        range(len(prey_genomes))
    )

    diagnostic = diagnostics[2]
    assert diagnostic == {
        "genome_index": 2,
        "survival": 7.5,
        "food": 57.0,
        "exploration": 13.0,
        "stagnation": -18.5,
        "fitness": prey_fitness[2],
        "food_eaten": 552.0,
        "steps_survived": 1102.0,
        "visited_regions": 1652.0,
        "stagnation_steps": 2202.0,
    }
    assert all(
        row["fitness"] == prey_fitness[row["genome_index"]]
        for row in diagnostics
    )


def test_joint_diagnostics_are_reproducible(configs):
    trainer = create_trainer(configs)
    prey_genomes = trainer.prey_trainer.create_initial_population()
    predator_genomes = (
        trainer.predator_trainer.create_initial_population()
    )

    first_prey, first_predators, first_diagnostics = (
        trainer.evaluate_populations(
            prey_genomes,
            predator_genomes,
            include_diagnostics=True,
        )
    )
    second_prey, second_predators, second_diagnostics = (
        trainer.evaluate_populations(
            prey_genomes,
            predator_genomes,
            include_diagnostics=True,
        )
    )

    np.testing.assert_array_equal(first_prey, second_prey)
    np.testing.assert_array_equal(first_predators, second_predators)
    assert first_diagnostics == second_diagnostics


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