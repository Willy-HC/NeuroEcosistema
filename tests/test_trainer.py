
import json

import numpy as np
import pytest

from neuroecosistema.evolution.episode_evaluator import EpisodeEvaluator
from neuroecosistema.evolution.trainer import EvolutionTrainer
from neuroecosistema.neural_network.network import NeuralNetwork

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


def create_trainer(
    species="prey",
    fitness_mode="baseline",
    food_progress_weight=0.05,
    pretrained_model_path=None,
    pretrained_fraction=0.5,
    pretrained_noise=0.02,
):
    return EvolutionTrainer(
        SIMULATION_CONFIG,
        NEURAL_CONFIG,
        EVOLUTION_CONFIG,
        species=species,
        evaluation_seeds=(42, 123),
        fitness_mode=fitness_mode,
        food_progress_weight=food_progress_weight,
        pretrained_model_path=pretrained_model_path,
        pretrained_fraction=pretrained_fraction,
        pretrained_noise=pretrained_noise,
    )


def make_diagnostics(fitness_values, offset):
    diagnostics = []

    for genome_index, fitness in enumerate(fitness_values):
        diagnostics.append(
            {
                "genome_index": genome_index,
                "survival": offset + genome_index,
                "food": 2 * offset + genome_index,
                "exploration": 3 * offset + genome_index,
                "stagnation": -offset - genome_index,
                "fitness": fitness,
                "food_eaten": 10 * offset + genome_index,
                "steps_survived": 20 * offset + genome_index,
                "visited_regions": 30 * offset + genome_index,
                "stagnation_steps": 40 * offset + genome_index,
                "food_progress_score": 0.1 * offset,
                "food_progress": 0.005 * offset,
            }
        )

    return diagnostics


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


def test_pretrained_population_keeps_random_diversity_and_is_reproducible(
    tmp_path,
):
    seed_network = NeuralNetwork(
        input_size=8,
        hidden_layers=[16, 12],
        output_size=2,
        seed=6,
    )
    model_path = tmp_path / "best_prey.npz"
    np.savez_compressed(model_path, genome=seed_network.get_genome())
    options = {
        "pretrained_model_path": model_path,
        "pretrained_fraction": 0.5,
        "pretrained_noise": 0.01,
    }

    first = create_trainer(**options).create_initial_population()
    second = create_trainer(**options).create_initial_population()
    base_genome = seed_network.get_genome()

    assert len(first) == 6
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(first[0], base_genome)
    assert np.linalg.norm(first[1] - base_genome) < 1.0
    assert np.linalg.norm(first[3] - base_genome) > 1.0


def test_pretrained_fraction_leaves_a_random_member(tmp_path):
    network = NeuralNetwork(
        input_size=8,
        hidden_layers=[16, 12],
        output_size=2,
        seed=7,
    )
    model_path = tmp_path / "best_prey.npz"
    np.savez_compressed(model_path, genome=network.get_genome())

    with pytest.raises(ValueError):
        create_trainer(
            pretrained_model_path=model_path,
            pretrained_fraction=1.0,
        )


def test_population_evaluation_returns_fitness():
    trainer = create_trainer()
    genomes = trainer.create_initial_population()

    fitness = trainer.evaluate_population(genomes)

    assert fitness.shape == (6,)
    assert np.all(np.isfinite(fitness))


def test_population_diagnostics_average_per_genome_and_seed(monkeypatch):
    trainer = create_trainer()
    genomes = trainer.create_initial_population()
    evaluated_seeds = []

    def fake_evaluate(
        evaluator,
        evaluated_genomes,
        species="prey",
        include_diagnostics=False,
    ):
        offset = 1 if evaluator.seed == 42 else 3
        evaluated_seeds.append(evaluator.seed)
        fitness = np.asarray(
            [
                0.1 * offset + genome_index * 0.01
                for genome_index in range(len(evaluated_genomes))
            ],
            dtype=np.float64,
        )
        diagnostics = make_diagnostics(fitness, offset)

        if include_diagnostics:
            return fitness, diagnostics

        return fitness

    monkeypatch.setattr(EpisodeEvaluator, "evaluate", fake_evaluate)

    default_fitness = trainer.evaluate_population(genomes)
    evaluated_seeds.clear()
    mean_fitness, diagnostics = trainer.evaluate_population(
        genomes,
        include_diagnostics=True,
    )

    np.testing.assert_array_equal(mean_fitness, default_fitness)
    assert evaluated_seeds == [42, 123]
    assert len(diagnostics) == len(genomes)
    assert [row["genome_index"] for row in diagnostics] == list(
        range(len(genomes))
    )

    diagnostic = diagnostics[2]
    assert diagnostic == {
        "genome_index": 2,
        "survival": 4.0,
        "food": 6.0,
        "exploration": 8.0,
        "stagnation": -4.0,
        "fitness": mean_fitness[2],
        "food_eaten": 22.0,
        "steps_survived": 42.0,
        "visited_regions": 62.0,
        "stagnation_steps": 82.0,
        "food_progress_score": 0.2,
        "food_progress": 0.01,
    }
    assert all(
        row["fitness"] == mean_fitness[row["genome_index"]]
        for row in diagnostics
    )


def test_predator_population_diagnostics_are_empty():
    trainer = create_trainer(species="predator")
    genomes = trainer.create_initial_population()

    fitness, diagnostics = trainer.evaluate_population(
        genomes,
        include_diagnostics=True,
    )

    assert fitness.shape == (len(genomes),)
    assert np.all(np.isfinite(fitness))
    assert diagnostics == []


def test_population_diagnostics_are_reproducible():
    trainer = create_trainer()
    genomes = trainer.create_initial_population()

    first_fitness, first_diagnostics = trainer.evaluate_population(
        genomes,
        include_diagnostics=True,
    )
    second_fitness, second_diagnostics = trainer.evaluate_population(
        genomes,
        include_diagnostics=True,
    )

    np.testing.assert_array_equal(first_fitness, second_fitness)
    assert first_diagnostics == second_diagnostics


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
    assert metadata["fitness_mode"] == "baseline"
    assert metadata["food_progress_weight"] == 0.0
    assert metadata["pretrained_model_path"] is None
    assert metadata["pretrained_fraction"] == 0.5
    assert metadata["pretrained_noise"] == 0.02


def test_save_before_training_fails(tmp_path):
    trainer = create_trainer()

    with pytest.raises(RuntimeError):
        trainer.save_results(tmp_path)


def test_experimental_fitness_settings_are_saved(
    tmp_path,
    monkeypatch,
):
    trainer = create_trainer(
        fitness_mode="food_progress",
        food_progress_weight=0.2,
    )
    monkeypatch.setattr(
        trainer,
        "evaluate_population",
        lambda genomes: np.zeros(len(genomes)),
    )

    trainer.train()
    metadata_path = trainer.save_results(tmp_path)["metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["fitness_mode"] == "food_progress"
    assert metadata["food_progress_weight"] == 0.2


def test_invalid_species():
    with pytest.raises(ValueError):
        create_trainer(species="unknown")


def test_food_progress_mode_is_rejected_for_predators():
    with pytest.raises(ValueError):
        create_trainer(
            species="predator",
            fitness_mode="food_progress",
        )