
import csv

import numpy as np
import pytest

from neuroecosistema.evaluate_behavior import (
    run_evaluations,
    save_results,
)
from neuroecosistema.evolution.behavior_evaluator import (
    BehaviorEvaluator,
)
from neuroecosistema.neural_network.network import NeuralNetwork


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
            "predators": 2,
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

    return simulation, neural


@pytest.fixture
def model_dir(tmp_path, configs):
    _, neural = configs

    for species in ("prey", "predator"):
        network = NeuralNetwork(
            input_size=neural["input_size"],
            hidden_layers=neural["hidden_layers"],
            output_size=neural["output_size"],
            hidden_activation=neural["activation_hidden"],
            output_activation=neural["activation_output"],
            seed=42,
        )

        np.savez_compressed(
            tmp_path / f"best_{species}.npz",
            genome=network.get_genome(),
        )

    return tmp_path


def test_random_evaluation_returns_both_species(configs):
    simulation, neural = configs
    evaluator = BehaviorEvaluator(simulation, neural)

    result = evaluator.evaluate(seed=42)

    assert set(result) == {"prey", "predator"}
    assert result["prey"]["agents"] == 4
    assert result["predator"]["agents"] == 2


def test_behavior_metrics_are_finite(configs):
    simulation, neural = configs
    evaluator = BehaviorEvaluator(simulation, neural)

    result = evaluator.evaluate(seed=42)

    for species in ("prey", "predator"):
        row = result[species]

        for key in (
            "mean_survival",
            "mean_food_eaten",
            "mean_captures",
            "mean_exploration",
            "mean_wall_ratio",
            "mean_distance",
        ):
            assert np.isfinite(row[key])


def test_wall_ratio_is_between_zero_and_one(configs):
    simulation, neural = configs
    evaluator = BehaviorEvaluator(simulation, neural)

    result = evaluator.evaluate(seed=42)

    for species in ("prey", "predator"):
        ratio = result[species]["mean_wall_ratio"]
        assert 0.0 <= ratio <= 1.0


def test_random_evaluation_is_reproducible(configs):
    simulation, neural = configs
    evaluator = BehaviorEvaluator(simulation, neural)

    first = evaluator.evaluate(seed=42)
    second = evaluator.evaluate(seed=42)

    assert first == second


def test_trained_evaluation_loads_models(configs, model_dir):
    simulation, neural = configs
    evaluator = BehaviorEvaluator(simulation, neural)

    result = evaluator.evaluate(
        seed=42,
        model_dir=model_dir,
    )

    assert result["prey"]["agents"] == 4
    assert result["predator"]["agents"] == 2


def test_trained_evaluation_is_reproducible(
    configs,
    model_dir,
):
    simulation, neural = configs
    evaluator = BehaviorEvaluator(simulation, neural)

    first = evaluator.evaluate(
        seed=42,
        model_dir=model_dir,
    )
    second = evaluator.evaluate(
        seed=42,
        model_dir=model_dir,
    )

    assert first == second


def test_multiple_seeds_produce_two_rows_each(configs):
    simulation, neural = configs

    rows = run_evaluations(
        simulation,
        neural,
        seeds=(42, 123, 456),
    )

    assert len(rows) == 6


def test_empty_seed_list_is_rejected(configs):
    simulation, neural = configs

    with pytest.raises(ValueError):
        run_evaluations(
            simulation,
            neural,
            seeds=(),
        )


def test_results_are_saved(configs, tmp_path):
    simulation, neural = configs

    rows = run_evaluations(
        simulation,
        neural,
        seeds=(42, 123),
    )

    paths = save_results(
        rows,
        tmp_path / "behavior",
    )

    assert paths["episodes"].exists()
    assert paths["summary"].exists()

    with paths["episodes"].open(
        encoding="utf-8",
        newline="",
    ) as file:
        episode_rows = list(csv.DictReader(file))

    with paths["summary"].open(
        encoding="utf-8",
        newline="",
    ) as file:
        summary_rows = list(csv.DictReader(file))

    assert len(episode_rows) == 4
    assert len(summary_rows) == 2


def test_empty_results_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_results([], tmp_path)