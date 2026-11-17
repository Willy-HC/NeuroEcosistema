
import numpy as np
import pytest

from neuroecosistema.evolution.crossover import uniform_crossover
from neuroecosistema.evolution.mutation import gaussian_mutation
from neuroecosistema.evolution.selection import (
    select_elites,
    tournament_selection,
)


def test_tournament_selects_best_when_all_participate():
    fitness = [10, 20, 50, 30]

    winner = tournament_selection(
        fitness,
        tournament_size=4,
        rng=np.random.default_rng(42),
    )

    assert winner == 2


def test_tournament_is_reproducible():
    fitness = [10, 20, 30, 40, 50]

    winner_a = tournament_selection(
        fitness, 3, np.random.default_rng(42)
    )
    winner_b = tournament_selection(
        fitness, 3, np.random.default_rng(42)
    )

    assert winner_a == winner_b


def test_elitism_preserves_best_genomes():
    genomes = [
        np.full(4, 1.0),
        np.full(4, 2.0),
        np.full(4, 3.0),
    ]
    fitness = [10, 30, 20]

    elites = select_elites(genomes, fitness, elitism=2)

    np.testing.assert_array_equal(elites[0], genomes[1])
    np.testing.assert_array_equal(elites[1], genomes[2])


def test_elites_are_independent_copies():
    genomes = [np.array([1.0, 2.0])]
    elites = select_elites(genomes, [10], elitism=1)

    elites[0][0] = 99

    assert genomes[0][0] == 1.0


def test_crossover_uses_both_parents():
    parent_a = np.zeros(374)
    parent_b = np.ones(374)

    child = uniform_crossover(
        parent_a,
        parent_b,
        crossover_rate=1.0,
        rng=np.random.default_rng(42),
    )

    assert child.shape == (374,)
    assert np.any(child == 0)
    assert np.any(child == 1)


def test_crossover_without_crossing_copies_first_parent():
    parent_a = np.zeros(10)
    parent_b = np.ones(10)

    child = uniform_crossover(
        parent_a,
        parent_b,
        crossover_rate=0.0,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(child, parent_a)


def test_mutation_changes_genes():
    genome = np.zeros(374)

    mutated = gaussian_mutation(
        genome,
        mutation_rate=1.0,
        mutation_strength=0.15,
        rng=np.random.default_rng(42),
    )

    assert np.any(mutated != genome)


def test_mutation_preserves_original():
    genome = np.zeros(374)

    mutated = gaussian_mutation(
        genome,
        mutation_rate=1.0,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(genome, np.zeros(374))
    assert not np.shares_memory(genome, mutated)


def test_mutation_without_rate_does_not_change_genome():
    genome = np.ones(374)

    mutated = gaussian_mutation(
        genome,
        mutation_rate=0.0,
        rng=np.random.default_rng(42),
    )

    np.testing.assert_array_equal(mutated, genome)


def test_invalid_mutation_rate():
    with pytest.raises(ValueError):
        gaussian_mutation(np.ones(10), mutation_rate=1.5)


def test_invalid_tournament_size():
    with pytest.raises(ValueError):
        tournament_selection([10, 20], tournament_size=3)