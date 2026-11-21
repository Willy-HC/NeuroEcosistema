import csv
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from neuroecosistema import evaluate_behavior


def run_cli_with_mocked_evaluation(monkeypatch, args):
    captured = []

    def capture_evaluations(
        simulation_config,
        neural_config,
        **kwargs,
    ):
        captured.append(
            (deepcopy(simulation_config), neural_config, kwargs)
        )
        return []

    monkeypatch.setattr(
        evaluate_behavior,
        "run_evaluations",
        capture_evaluations,
    )
    monkeypatch.setattr(
        evaluate_behavior,
        "save_results",
        lambda rows, output_dir: {},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_behavior", *args],
    )

    evaluate_behavior.main()

    return captured[0]


@pytest.mark.parametrize(
    ("population_args", "expected"),
    [
        ([], (50, 10)),
        (["--prey-count", "7"], (7, 10)),
        (["--predator-count", "4"], (50, 4)),
        (
            ["--prey-count", "7", "--predator-count", "4"],
            (7, 4),
        ),
    ],
)
def test_population_count_options_preserve_defaults_and_yaml(
    monkeypatch,
    population_args,
    expected,
):
    root = Path(__file__).resolve().parents[1]
    simulation_yaml = root / "config" / "simulation.yaml"
    original_yaml = simulation_yaml.read_bytes()

    simulation, _, _ = run_cli_with_mocked_evaluation(
        monkeypatch,
        population_args,
    )

    assert (
        simulation["population"]["prey"],
        simulation["population"]["predators"],
    ) == expected
    assert simulation_yaml.read_bytes() == original_yaml


def test_existing_cli_options_are_forwarded(
    monkeypatch,
):
    simulation, _, kwargs = run_cli_with_mocked_evaluation(
        monkeypatch,
        [
            "--model-dir",
            r"results\coevolution_diagnostics_5gen",
            "--strategy",
            "neural",
            "--prey-strategy",
            "neural",
            "--predator-strategy",
            "random",
            "--seeds",
            "1001",
            "1002",
            "--max-steps",
            "17",
            "--prey-count",
            "3",
            "--predator-count",
            "2",
            "--output",
            "results/test-output",
        ],
    )

    assert simulation["population"] == {
        "prey": 3,
        "predators": 2,
    }
    assert simulation["simulation"]["max_steps"] == 17
    assert kwargs == {
        "seeds": [1001, 1002],
        "model_dir": r"results\coevolution_diagnostics_5gen",
        "strategy": "neural",
        "prey_strategy": "neural",
        "predator_strategy": "random",
    }


@pytest.mark.parametrize(
    "invalid_args",
    [
        ["--prey-count", "0"],
        ["--prey-count", "-1"],
        ["--prey-count", "many"],
        ["--predator-count", "0"],
        ["--predator-count", "-1"],
        ["--predator-count", "many"],
    ],
)
def test_invalid_population_counts_are_rejected(
    monkeypatch,
    invalid_args,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate_behavior", *invalid_args],
    )

    with pytest.raises(SystemExit) as error:
        evaluate_behavior.main()

    assert error.value.code == 2


def test_requested_populations_are_recorded_and_reproducible(
    monkeypatch,
    tmp_path,
):
    output_dirs = (
        tmp_path / "first",
        tmp_path / "second",
    )
    episode_rows = []

    for output_dir in output_dirs:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate_behavior",
                "--prey-count",
                "2",
                "--predator-count",
                "3",
                "--strategy",
                "random",
                "--seeds",
                "42",
                "--max-steps",
                "2",
                "--output",
                str(output_dir),
            ],
        )

        evaluate_behavior.main()

        episodes_path = output_dir / "behavior_episodes.csv"
        with episodes_path.open(
            newline="",
            encoding="utf-8",
        ) as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            assert reader.fieldnames == [
                "seed",
                "species",
                "agents",
                "steps_executed",
                "mean_survival",
                "mean_food_eaten",
                "mean_captures",
                "mean_exploration",
                "mean_wall_ratio",
                "mean_distance",
                "mean_food_approach_progress",
                "mean_food_net_progress",
                "mean_food_target_switches",
            ]
        episode_rows.append(rows)

    assert episode_rows[0] == episode_rows[1]
    assert [
        (row["species"], int(row["agents"]))
        for row in episode_rows[0]
    ] == [
        ("prey", 2),
        ("predator", 3),
    ]
    assert all(row["seed"] == "42" for row in episode_rows[0])
