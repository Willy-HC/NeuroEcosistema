
import numpy as np
import pytest

from neuroecosistema.evolution.episode_evaluator import (
    EpisodeEvaluator,
)
from neuroecosistema.neural_network.network import NeuralNetwork
from neuroecosistema.simulation.environment import Environment

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


def create_evaluator(max_steps=100, seed=42):
    simulation_config = {
        **SIMULATION_CONFIG,
        "simulation": {
            **SIMULATION_CONFIG["simulation"],
            "max_steps": max_steps,
        },
    }

    return EpisodeEvaluator(
        simulation_config,
        NEURAL_CONFIG,
        seed=seed,
    )


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


def test_evaluate_returns_numpy_array_by_default():
    evaluator = create_evaluator()

    result = evaluator.evaluate(create_genomes(3))

    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)


def test_evaluate_both_returns_two_arrays_by_default():
    evaluator = create_evaluator()
    genomes = create_genomes(3)

    result = evaluator.evaluate_both(genomes, genomes)

    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(values, np.ndarray) for values in result)
    assert all(values.shape == (3,) for values in result)


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


def test_food_progress_fitness_mode_adds_only_its_weighted_term(
    monkeypatch,
):
    def set_progress(environment, movement_cost=0.1):
        for prey in environment.prey:
            prey.food_progress_score = 0.4
            prey.steps_survived = 1

    monkeypatch.setattr(Environment, "update_agents", set_progress)
    monkeypatch.setattr(
        Environment,
        "resolve_interactions",
        lambda environment, capture_reward=30.0: None,
    )
    monkeypatch.setattr(Environment, "update_food", lambda environment: None)
    genomes = create_genomes(2)
    baseline = EpisodeEvaluator(
        SIMULATION_CONFIG,
        NEURAL_CONFIG,
        seed=17,
    )
    experimental = EpisodeEvaluator(
        SIMULATION_CONFIG,
        NEURAL_CONFIG,
        seed=17,
        fitness_mode="food_progress",
        food_progress_weight=0.2,
    )

    baseline_fitness = baseline.evaluate(genomes)
    experimental_fitness, diagnostics = experimental.evaluate(
        genomes,
        include_diagnostics=True,
    )

    np.testing.assert_allclose(
        experimental_fitness - baseline_fitness,
        0.08,
    )
    assert all(row["food_progress_score"] == 0.4 for row in diagnostics)
    assert all(
        row["food_progress"] == pytest.approx(0.08)
        for row in diagnostics
    )


def test_food_progress_mode_does_not_change_predator_fitness():
    genomes = create_genomes(3)
    baseline = EpisodeEvaluator(
        SIMULATION_CONFIG,
        NEURAL_CONFIG,
        seed=27,
    )
    experimental = EpisodeEvaluator(
        SIMULATION_CONFIG,
        NEURAL_CONFIG,
        seed=27,
        fitness_mode="food_progress",
        food_progress_weight=0.5,
    )

    np.testing.assert_array_equal(
        baseline.evaluate(genomes, species="predator"),
        experimental.evaluate(genomes, species="predator"),
    )


@pytest.mark.parametrize("evaluate_both", [False, True])
def test_diagnostics_are_reproducible(evaluate_both):
    genomes = create_genomes(3)
    evaluator_a = create_evaluator(seed=42)
    evaluator_b = create_evaluator(seed=42)

    if evaluate_both:
        result_a = evaluator_a.evaluate_both(
            genomes,
            genomes,
            include_diagnostics=True,
        )
        result_b = evaluator_b.evaluate_both(
            genomes,
            genomes,
            include_diagnostics=True,
        )
        np.testing.assert_array_equal(result_a[0], result_b[0])
        np.testing.assert_array_equal(result_a[1], result_b[1])
        assert result_a[2] == result_b[2]
    else:
        fitness_a, diagnostics_a = evaluator_a.evaluate(
            genomes,
            include_diagnostics=True,
        )
        fitness_b, diagnostics_b = evaluator_b.evaluate(
            genomes,
            include_diagnostics=True,
        )

        np.testing.assert_array_equal(fitness_a, fitness_b)
        assert diagnostics_a == diagnostics_b


@pytest.mark.parametrize("evaluate_both", [False, True])
def test_diagnostics_include_all_fields_and_match_fitness(
    evaluate_both,
):
    evaluator = create_evaluator(max_steps=1)
    genomes = create_genomes(3)

    if evaluate_both:
        prey_fitness, _, diagnostics = evaluator.evaluate_both(
            genomes,
            genomes,
            include_diagnostics=True,
        )
    else:
        prey_fitness, diagnostics = evaluator.evaluate(
            genomes,
            include_diagnostics=True,
        )

    expected_fields = {
        "genome_index",
        "survival",
        "food",
        "exploration",
        "stagnation",
        "fitness",
        "food_eaten",
        "steps_survived",
        "visited_regions",
        "stagnation_steps",
        "food_progress_score",
        "food_progress",
    }

    assert len(diagnostics) == len(genomes)

    for genome_index, diagnostic in enumerate(diagnostics):
        assert set(diagnostic) == expected_fields
        assert diagnostic["genome_index"] == genome_index
        assert diagnostic["fitness"] == prey_fitness[genome_index]
        assert diagnostic["food_progress"] == 0.0


@pytest.mark.parametrize("evaluate_both", [False, True])
def test_diagnostics_keep_original_genome_indices_for_dead_prey(
    monkeypatch,
    evaluate_both,
):
    def set_prey_metrics(environment, movement_cost=0.1):
        for genome_index, prey in enumerate(environment.prey):
            prey.food_eaten = genome_index
            prey.steps_survived = 10 * (genome_index + 1)
            prey.visited_regions = {
                (region, 0)
                for region in range(genome_index + 3)
            }
            prey.stagnation_steps = genome_index

            if genome_index == 1:
                prey.alive = False

        environment.prey = [
            prey for prey in environment.prey if prey.alive
        ]

    monkeypatch.setattr(Environment, "update_agents", set_prey_metrics)
    monkeypatch.setattr(
        Environment,
        "resolve_interactions",
        lambda environment, capture_reward=30.0: None,
    )

    evaluator = create_evaluator(max_steps=1)
    genomes = create_genomes(3)

    if evaluate_both:
        prey_fitness, _, diagnostics = evaluator.evaluate_both(
            genomes,
            genomes,
            include_diagnostics=True,
        )
    else:
        prey_fitness, diagnostics = evaluator.evaluate(
            genomes,
            include_diagnostics=True,
        )

    assert [row["genome_index"] for row in diagnostics] == [0, 1, 2]
    assert [row["food_eaten"] for row in diagnostics] == [0, 1, 2]
    assert [row["steps_survived"] for row in diagnostics] == [10, 20, 30]
    assert [row["visited_regions"] for row in diagnostics] == [3, 4, 5]
    assert [row["stagnation_steps"] for row in diagnostics] == [0, 1, 2]
    assert [
        row["fitness"] for row in diagnostics
    ] == prey_fitness.tolist()


@pytest.mark.parametrize("evaluate_both", [False, True])
def test_diagnostics_do_not_change_fitness_vectors(evaluate_both):
    evaluator = create_evaluator()
    genomes = create_genomes(3)

    if evaluate_both:
        original_prey, original_predators = evaluator.evaluate_both(
            genomes,
            genomes,
        )
        diagnostic_prey, diagnostic_predators, _ = (
            evaluator.evaluate_both(
                genomes,
                genomes,
                include_diagnostics=True,
            )
        )
        np.testing.assert_array_equal(original_prey, diagnostic_prey)
        np.testing.assert_array_equal(
            original_predators,
            diagnostic_predators,
        )
    else:
        original_fitness = evaluator.evaluate(genomes)
        diagnostic_fitness, _ = evaluator.evaluate(
            genomes,
            include_diagnostics=True,
        )
        np.testing.assert_array_equal(
            original_fitness,
            diagnostic_fitness,
        )


@pytest.mark.parametrize("evaluate_both", [False, True])
def test_diagnostics_do_not_run_extra_simulation_steps(
    monkeypatch,
    evaluate_both,
):
    update_calls = []

    def count_updates(environment, movement_cost=0.1):
        update_calls.append(1)

    monkeypatch.setattr(Environment, "update_agents", count_updates)
    monkeypatch.setattr(
        Environment,
        "resolve_interactions",
        lambda environment, capture_reward=30.0: None,
    )

    evaluator = create_evaluator(max_steps=3)
    genomes = create_genomes(2)

    if evaluate_both:
        evaluator.evaluate_both(
            genomes,
            genomes,
            include_diagnostics=True,
        )
    else:
        evaluator.evaluate(
            genomes,
            include_diagnostics=True,
        )

    assert len(update_calls) == 3


def test_predator_evaluation_has_no_prey_genome_diagnostics():
    evaluator = create_evaluator(max_steps=1)
    genomes = create_genomes(2)

    fitness, diagnostics = evaluator.evaluate(
        genomes,
        species="predator",
        include_diagnostics=True,
    )

    assert fitness.shape == (2,)
    assert diagnostics == []


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

def test_episode_continues_without_predators(monkeypatch):
    steps_executed = []

    def fake_update_agents(self, movement_cost=0.1):
        steps_executed.append(1)
        self.predators.clear()

    monkeypatch.setattr(
        Environment,
        "update_agents",
        fake_update_agents,
    )

    simulation_config = {
        "simulation": {
            "width": 500,
            "height": 500,
            "max_steps": 5,
        },
        "population": {
            "prey": 1,
            "predators": 1,
        },
        "environment": {
            "initial_food": 5,
            "food_respawn": 0,
        },
        "energy": {
            "food_reward": 20,
            "movement_cost": 0.1,
            "capture_reward": 30,
        },
    }

    neural_config = {
        "input_size": 8,
        "hidden_layers": [16, 12],
        "output_size": 2,
        "activation_hidden": "relu",
        "activation_output": "tanh",
    }

    evaluator = EpisodeEvaluator(
        simulation_config,
        neural_config,
    )

    genome = evaluator.create_network(
        np.zeros(374)
    ).get_genome()

    evaluator.evaluate([genome], species="prey")

    assert len(steps_executed) == 5


def test_episode_continues_without_prey(monkeypatch):
    steps_executed = []

    def fake_update_agents(self, movement_cost=0.1):
        steps_executed.append(1)
        self.prey.clear()

    monkeypatch.setattr(
        Environment,
        "update_agents",
        fake_update_agents,
    )

    simulation_config = {
        "simulation": {
            "width": 500,
            "height": 500,
            "max_steps": 5,
        },
        "population": {
            "prey": 1,
            "predators": 1,
        },
        "environment": {
            "initial_food": 5,
            "food_respawn": 0,
        },
        "energy": {
            "food_reward": 20,
            "movement_cost": 0.1,
            "capture_reward": 30,
        },
    }

    neural_config = {
        "input_size": 8,
        "hidden_layers": [16, 12],
        "output_size": 2,
        "activation_hidden": "relu",
        "activation_output": "tanh",
    }

    evaluator = EpisodeEvaluator(
        simulation_config,
        neural_config,
    )

    genome = evaluator.create_network(
        np.zeros(374)
    ).get_genome()

    evaluator.evaluate([genome], species="predator")

    assert len(steps_executed) == 5