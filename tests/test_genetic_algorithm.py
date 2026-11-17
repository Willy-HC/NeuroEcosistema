
import numpy as np
import pytest

from neuroecosistema.evolution.genetic_algorithm import (
    GeneticAlgorithm,
)


def create_population(size=10, genome_size=374):
    rng = np.random.default_rng(100)

    return [
        rng.normal(size=genome_size)
        for _ in range(size)
    ]


def create_algorithm(seed=42):
    return GeneticAlgorithm(
        population_size=10,
        elitism=2,
        mutation_rate=0.10,
        mutation_strength=0.15,
        crossover_rate=0.70,
        tournament_size=3,
        seed=seed,
    )


def test_evolution_preserves_population_size():
    algorithm = create_algorithm()
    genomes = create_population()
    fitness = np.arange(10, dtype=float)

    new_population = algorithm.evolve(genomes, fitness)

    assert len(new_population) == 10


def test_evolution_preserves_genome_size():
    algorithm = create_algorithm()
    genomes = create_population()
    fitness = np.arange(10, dtype=float)

    new_population = algorithm.evolve(genomes, fitness)

    assert all(
        genome.shape == (374,)
        for genome in new_population
    )


def test_evolution_preserves_elites():
    algorithm = create_algorithm()
    genomes = create_population()
    fitness = np.arange(10, dtype=float)

    new_population = algorithm.evolve(genomes, fitness)

    np.testing.assert_array_equal(
        new_population[0],
        genomes[9],
    )

    np.testing.assert_array_equal(
        new_population[1],
        genomes[8],
    )


def test_evolution_is_reproducible():
    genomes = create_population()
    fitness = np.arange(10, dtype=float)

    algorithm_a = create_algorithm(seed=42)
    algorithm_b = create_algorithm(seed=42)

    result_a = algorithm_a.evolve(genomes, fitness)
    result_b = algorithm_b.evolve(genomes, fitness)

    for genome_a, genome_b in zip(result_a, result_b):
        np.testing.assert_array_equal(
            genome_a,
            genome_b,
        )


def test_generation_counter_increases():
    algorithm = create_algorithm()
    genomes = create_population()
    fitness = np.arange(10, dtype=float)

    algorithm.evolve(genomes, fitness)

    assert algorithm.generation == 1

    algorithm.evolve(genomes, fitness)

    assert algorithm.generation == 2


def test_fitness_history_is_recorded():
    algorithm = create_algorithm()
    genomes = create_population()
    fitness = np.arange(10, dtype=float)

    algorithm.evolve(genomes, fitness)

    history = algorithm.get_history()

    assert len(history) == 1
    assert history[0]["best_fitness"] == 9.0
    assert history[0]["worst_fitness"] == 0.0
    assert history[0]["mean_fitness"] == 4.5


def test_evolution_rejects_invalid_population():
    algorithm = create_algorithm()
    genomes = create_population(size=5)
    fitness = np.arange(5, dtype=float)

    with pytest.raises(ValueError):
        algorithm.evolve(genomes, fitness)


def test_evolution_rejects_invalid_fitness():
    algorithm = create_algorithm()
    genomes = create_population()
    fitness = np.ones(10)
    fitness[0] = np.nan

    with pytest.raises(ValueError):
        algorithm.evolve(genomes, fitness)


def test_evolution_does_not_modify_original_genomes():
    algorithm = create_algorithm()
    genomes = create_population()
    originals = [genome.copy() for genome in genomes]
    fitness = np.arange(10, dtype=float)

    algorithm.evolve(genomes, fitness)

    for original, genome in zip(originals, genomes):
        np.testing.assert_array_equal(original, genome)


def test_evolution_rejects_different_genome_sizes():
    algorithm = create_algorithm()
    genomes = create_population()
    genomes[0] = np.zeros(100)
    fitness = np.arange(10, dtype=float)

    with pytest.raises(ValueError):
        algorithm.evolve(genomes, fitness)