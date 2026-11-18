
import csv

import pytest

from neuroecosistema.agents.predator import Predator
from neuroecosistema.agents.prey import Prey
from neuroecosistema.evaluate_behavior import save_results
from neuroecosistema.evolution.behavior_evaluator import (
    BehaviorEvaluator,
)


def test_prey_summary_includes_net_progress():
    first = Prey(x=100, y=100)
    second = Prey(x=200, y=200)

    first.food_net_progress = 10.0
    second.food_net_progress = 30.0

    result = BehaviorEvaluator.summarize(
        agents=[first, second],
        species="prey",
        seed=42,
        steps_executed=10,
    )

    assert result["mean_food_net_progress"] == pytest.approx(20.0)


def test_predator_summary_has_zero_net_progress():
    predator = Predator(x=100, y=100)

    result = BehaviorEvaluator.summarize(
        agents=[predator],
        species="predator",
        seed=42,
        steps_executed=10,
    )

    assert result["mean_food_net_progress"] == 0.0


def test_net_progress_is_exported_to_csv(tmp_path):
    prey = Prey(x=100, y=100)
    predator = Predator(x=200, y=200)

    prey.food_net_progress = 15.0

    rows = [
        BehaviorEvaluator.summarize(
            agents=[prey],
            species="prey",
            seed=42,
            steps_executed=10,
        ),
        BehaviorEvaluator.summarize(
            agents=[predator],
            species="predator",
            seed=42,
            steps_executed=10,
        ),
    ]

    paths = save_results(
        rows=rows,
        output_dir=tmp_path,
    )

    with paths["episodes"].open(
        newline="",
        encoding="utf-8",
    ) as file:
        episode_rows = list(csv.DictReader(file))

    with paths["summary"].open(
        newline="",
        encoding="utf-8",
    ) as file:
        summary_rows = list(csv.DictReader(file))

    assert float(
        episode_rows[0]["mean_food_net_progress"]
    ) == pytest.approx(15.0)

    assert float(
        summary_rows[0]["mean_food_net_progress"]
    ) == pytest.approx(15.0)

    assert float(
        summary_rows[0]["std_mean_food_net_progress"]
    ) == pytest.approx(0.0)