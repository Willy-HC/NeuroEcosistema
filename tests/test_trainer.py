
import json

import numpy as np
import pytest

from neuroecosistema.evolution.trainer import EvolutionTrainer

SIMULATION_CONFIG = {
    "simulation": {
        "width": 500,
        "height": 400,
        "max_steps": 20,
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

EVOLUTION_CONFIG = {
    "population_size": 6,
    "generations": 3,
    "elitism": 2,
    "mutation_rate": 0.10,
    "mutation_strength": 0.15,
    "crossover_rate": 0.70,
    "selection": "tournament",
    "seed": 42,
}


def create_trainer(species="prey"):
    return EvolutionTrainer(
        SIMULATION_CONFIG,
        NEURAL_CONFIG,
        EVOLUTION_CONFIG,
        species=species,
        evaluation_seeds=(42, 123),
    )


def test_initial_population_size():
    trainer = create_trainer()

    genomes = trainer.create_initial_population()

    assert len(genomes) == 6
    assert all(genome.shape == (374,) for genome in genomes)


def test_initial_population_is_reproducible():
    trainer_a = create_trainer()
    trainer_b = create_trainer()

    genomes_a = trainer_a.create_initial_population()
    genomes_b = trainer_b.create_initial_population()

    for genome_a, genome_b in zip(genomes_a, genomes_b):
        np.testing.assert_array_equal(genome_a, genome_b)


def test_initial_genomes_are_different():
    trainer = create_trainer()
    genomes = trainer.create_initial_population()

    assert not np.array_equal(genomes[0], genomes[1])


def test_population_evaluation_returns_fitness():
    trainer = create_trainer()
    genomes = trainer.create_initial_population()

    fitness = trainer.evaluate_population(genomes)

    assert fitness.shape == (6,)
    assert np.all(np.isfinite(fitness))


def test_training_records_generations(monkeypatch):
    trainer = create_trainer()

    def fake_evaluate(genomes):
        return np.arange(len(genomes), dtype=float)

    monkeypatch.setattr(
        trainer,
        "evaluate_population",
        fake_evaluate,
    )

    best = trainer.train()

    assert best.shape == (374,)
    assert len(trainer.history) == 3
    assert trainer.best_generation == 1
    assert trainer.best_fitness == 5.0
    assert trainer.history[-1]["generation"] == 3


def test_training_preserves_best_historical_model(monkeypatch):
    trainer = create_trainer()
    call_count = 0

    def fake_evaluate(genomes):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return np.array([1, 2, 3, 4, 5, 10], dtype=float)

        return np.zeros(len(genomes))

    monkeypatch.setattr(
        trainer,
        "evaluate_population",
        fake_evaluate,
    )

    best = trainer.train()

    assert trainer.best_fitness == 10.0
    assert trainer.best_generation == 1
    np.testing.assert_array_equal(best, trainer.best_genome)


def test_save_results(tmp_path, monkeypatch):
    trainer = create_trainer()

    monkeypatch.setattr(
        trainer,
        "evaluate_population",
        lambda genomes: np.arange(len(genomes), dtype=float),
    )

    trainer.train()
    paths = trainer.save_results(tmp_path)

    assert all(path.exists() for path in paths.values())

    with np.load(paths["model"]) as data:
        np.testing.assert_array_equal(
            data["genome"],
            trainer.best_genome,
        )

    metadata = json.loads(
        paths["metadata"].read_text(encoding="utf-8")
    )

    assert metadata["species"] == "prey"
    assert metadata["best_fitness"] == 5.0
    assert metadata["evaluation_seeds"] == [42, 123]


def test_save_before_training_fails(tmp_path):
    trainer = create_trainer()

    with pytest.raises(RuntimeError):
        trainer.save_results(tmp_path)


def test_invalid_species():
    with pytest.raises(ValueError):
        create_trainer(species="unknown")