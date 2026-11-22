import argparse
import csv
import json

import numpy as np
import pytest

from neuroecosistema.evaluate_learning import (
    parse_scenario,
    run_learning_comparison,
)
from neuroecosistema.neural_network.network import NeuralNetwork


def _write_models(directory):
    directory.mkdir()
    for species in ("prey", "predator"):
        network = NeuralNetwork(
            input_size=8,
            hidden_layers=[8, 6],
            output_size=2,
            seed=14,
        )
        np.savez_compressed(
            directory / f"best_{species}.npz",
            genome=network.get_genome(),
        )


def test_parse_scenario_validates_counts():
    assert parse_scenario("10,2,30") == (10, 2, 30)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_scenario("0,-1,30")


def test_learning_comparison_writes_all_policy_species_rows(tmp_path):
    pretrained_dir = tmp_path / "pretrained"
    coevolved_dir = tmp_path / "coevolved"
    _write_models(pretrained_dir)
    _write_models(coevolved_dir)
    output_dir = tmp_path / "comparison"
    simulation = {
        "simulation": {
            "width": 500,
            "height": 400,
            "max_steps": 2,
        },
        "population": {
            "prey": 2,
            "predators": 1,
        },
        "environment": {
            "initial_food": 3,
            "food_respawn": 0,
        },
        "energy": {
            "prey_initial": 100,
            "predator_initial": 100,
            "movement_cost": 0.1,
            "food_reward": 20,
            "capture_reward": 30,
        },
    }
    neural = {
        "input_size": 8,
        "hidden_layers": [8, 6],
        "output_size": 2,
        "activation_hidden": "relu",
        "activation_output": "tanh",
    }

    paths = run_learning_comparison(
        simulation,
        neural,
        pretrained_model_dir=pretrained_dir,
        coevolved_model_dir=coevolved_dir,
        output_dir=output_dir,
        seeds=(42,),
        scenarios=((2, 1, 3),),
        max_steps=2,
        untrained_seed=15,
    )

    with paths["episodes"].open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))
    manifest = json.loads(
        paths["manifest"].read_text(encoding="utf-8")
    )

    assert len(rows) == 10
    assert {
        row["policy"] for row in rows
    } == {
        "random",
        "heuristic",
        "untrained",
        "pretrained",
        "coevolved",
    }
    assert {
        row["species"] for row in rows
    } == {"prey", "predator"}
    assert manifest["episode_seeds"] == [42]
    assert len(manifest["model_sha256"]["untrained"]) == 2
    assert (output_dir / "untrained_models" / "best_prey.npz").is_file()

    with pytest.raises(FileExistsError):
        run_learning_comparison(
            simulation,
            neural,
            pretrained_model_dir=pretrained_dir,
            coevolved_model_dir=coevolved_dir,
            output_dir=output_dir,
            seeds=(42,),
            scenarios=((2, 1, 3),),
            max_steps=2,
            untrained_seed=15,
        )
