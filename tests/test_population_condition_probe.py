import csv
import json
from pathlib import Path

import numpy as np
import pytest

from neuroecosistema.analysis.population_condition_probe import (
    CONDITIONS,
    _create_environment,
    _create_initial_layout,
    _load_models,
    default_model_paths,
    run_population_condition_probe,
)
from neuroecosistema.evolution.behavior_evaluator import BehaviorEvaluator
from neuroecosistema.train import load_configs


def test_existing_models_load_with_expected_architecture_and_hashes():
    _, neural_config, _ = load_configs()
    models = _load_models(default_model_paths(), neural_config)

    assert set(models) == {"5G", "30G"}
    for model_species in models.values():
        assert set(model_species) == {"prey", "predator"}
        for model in model_species.values():
            network = model["network"]
            assert (
                network.input_size,
                *network.hidden_layers,
                network.output_size,
            ) == (8, 16, 12, 2)
            assert network.count_parameters() == 374
            assert len(model["sha256"]) == 64
            np.testing.assert_array_equal(
                network.get_genome(),
                model["genome"],
            )


def test_explicit_layout_is_shared_across_population_conditions():
    simulation, _, _ = load_configs()
    layout = _create_initial_layout(
        seed=42,
        width=simulation["simulation"]["width"],
        height=simulation["simulation"]["height"],
        food_count=simulation["environment"]["initial_food"],
    )
    environments = [
        _create_environment(
            simulation,
            simulation["energy"],
            condition,
            seed=42,
            layout=layout,
        )
        for condition in CONDITIONS
    ]

    assert [(plant.x, plant.y) for plant in environments[0].plants] == [
        (plant.x, plant.y) for plant in environments[1].plants
    ]
    assert [(plant.x, plant.y) for plant in environments[0].plants] == [
        (plant.x, plant.y) for plant in environments[2].plants
    ]
    assert [(agent.x, agent.y) for agent in environments[0].prey] == [
        (agent.x, agent.y) for agent in environments[2].prey
    ]
    assert [(agent.x, agent.y) for agent in environments[0].prey] == [
        (agent.x, agent.y) for agent in environments[1].prey[:10]
    ]
    assert [(agent.x, agent.y) for agent in environments[0].predators] == [
        (agent.x, agent.y) for agent in environments[1].predators
    ]
    assert [len(environment.prey) for environment in environments] == [
        10,
        50,
        10,
    ]
    assert [len(environment.predators) for environment in environments] == [
        10,
        10,
        0,
    ]


@pytest.mark.parametrize(
    ("seeds", "max_steps"),
    [
        ((), 1),
        ((42, 42), 1),
        ((-1,), 1),
        ((True,), 1),
        ((42,), 0),
        ((42,), 1.5),
    ],
)
def test_invalid_seeds_or_step_counts_are_rejected(
    tmp_path,
    seeds,
    max_steps,
):
    with pytest.raises(ValueError):
        run_population_condition_probe(
            tmp_path / "output",
            seeds=seeds,
            max_steps=max_steps,
        )

    assert not (tmp_path / "output").exists()


def test_probe_records_fitness_attribution_population_sizes_and_immutability(
    tmp_path,
):
    model_paths = default_model_paths()
    original_model_bytes = {
        path: Path(path).read_bytes()
        for model_species in model_paths.values()
        for path in model_species.values()
    }

    output_dir = tmp_path / "output"
    outputs = run_population_condition_probe(
        output_dir,
        seeds=(42,),
        max_steps=2,
    )

    assert outputs["episode_count"] == 10
    assert outputs["agent_diagnostic_rows"] == 340
    assert outputs["episode_summary_rows"] == 18

    with outputs["details"].open(
        newline="",
        encoding="utf-8",
    ) as file:
        detail_rows = list(csv.DictReader(file))
    with outputs["summary"].open(
        newline="",
        encoding="utf-8",
    ) as file:
        summary_rows = list(csv.DictReader(file))
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))

    assert len(detail_rows) == 340
    assert len(summary_rows) == 18
    assert manifest["seeds"] == [42]
    assert manifest["additional_seeds"] == []
    assert manifest["population_policy_type"] == ("clones of one genome per species")
    assert len(manifest["models"]["5G"]["prey"]["sha256"]) == 64
    assert len(manifest["models"]["30G"]["predator"]["sha256"]) == 64
    assert all(row["population_policy_type"] == "clones" for row in detail_rows)

    counts_by_condition = {
        condition.condition_id: (
            condition.prey_count,
            condition.predator_count,
        )
        for condition in CONDITIONS
    }
    for row in detail_rows:
        assert (
            int(row["prey_agents_initial"]),
            int(row["predator_agents_initial"]),
        ) == counts_by_condition[row["condition_id"]]
        components = [
            float(row[column])
            for column in (
                "fitness_survival",
                "fitness_food",
                "fitness_capture",
                "fitness_exploration",
                "fitness_stagnation",
            )
            if row[column] != ""
        ]
        assert float(row["fitness"]) == pytest.approx(max(0.0, sum(components)))
        if row["species"] == "prey":
            assert row["food_eaten"] != ""
            assert row["fitness_food"] != ""
            assert row["captures"] == ""
            assert row["policy_model_id"] == row["prey_model_id"]
            assert row["policy_model_sha256"] == row["prey_model_sha256"]
        else:
            assert row["food_eaten"] == ""
            assert row["fitness_capture"] != ""
            assert row["captures"] != ""
            assert row["policy_model_id"] == row["predator_model_id"]
            assert row["policy_model_sha256"] == row["predator_model_sha256"]

    without_predators = [
        row for row in detail_rows if row["condition_id"] == "prey10_pred0"
    ]
    assert len(without_predators) == 20
    assert {row["species"] for row in without_predators} == {"prey"}

    assert original_model_bytes == {
        path: Path(path).read_bytes() for path in original_model_bytes
    }
    assert {path.name for path in output_dir.iterdir()} == {
        "agent_diagnostics.csv",
        "episode_summary.csv",
        "manifest.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == {"output"}
    written_bytes = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    with pytest.raises(FileExistsError):
        run_population_condition_probe(
            output_dir,
            seeds=(42,),
            max_steps=2,
        )
    assert written_bytes == {
        path.name: path.read_bytes() for path in output_dir.iterdir()
    }


def test_probe_is_reproducible_and_accepts_additional_seeds(tmp_path):
    first = run_population_condition_probe(
        tmp_path / "first",
        seeds=(987,),
        max_steps=1,
    )
    second = run_population_condition_probe(
        tmp_path / "second",
        seeds=(987,),
        max_steps=1,
    )

    for key in ("details", "summary", "manifest"):
        assert first[key].read_bytes() == second[key].read_bytes()

    manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
    assert manifest["additional_seeds"] == [987]


def test_existing_behavior_evaluator_remains_compatible():
    simulation, neural, _ = load_configs()
    simulation["simulation"]["max_steps"] = 2
    simulation["population"]["prey"] = 2
    simulation["population"]["predators"] = 0
    simulation["environment"]["initial_food"] = 2
    simulation["environment"]["food_respawn"] = 0
    model_dir = default_model_paths()["5G"]["prey"].parent

    result = BehaviorEvaluator(simulation, neural).evaluate(
        seed=42,
        model_dir=model_dir,
        prey_strategy="neural",
        predator_strategy="neural",
    )

    assert result["prey"]["agents"] == 2
    assert result["prey"]["steps_executed"] == 2
    assert "predator" not in result
