import sys
from types import SimpleNamespace

import pytest

from neuroecosistema import train as train_module
from neuroecosistema.evolution.coevolution_trainer import (
    CoevolutionTrainer,
)


def create_configs():
    simulation = {
        "simulation": {
            "width": 300,
            "height": 250,
            "max_steps": 1,
        },
        "population": {
            "prey": 3,
            "predators": 3,
        },
        "environment": {
            "initial_food": 1,
            "food_respawn": 0,
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

    evolution = {
        "population_size": 3,
        "generations": 1,
        "elitism": 1,
        "mutation_rate": 0.1,
        "mutation_strength": 0.15,
        "crossover_rate": 0.7,
        "selection": "tournament",
        "seed": 42,
    }

    return simulation, neural, evolution


def test_record_diagnostics_is_rejected_for_independent_species(
    monkeypatch,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--species",
            "prey",
            "--record-diagnostics",
        ],
    )

    with pytest.raises(SystemExit) as error:
        train_module.main()

    assert error.value.code == 2


def test_cli_checks_conflicts_before_starting_training(
    monkeypatch,
    tmp_path,
):
    output_dir = tmp_path / "existing-results"
    output_dir.mkdir()
    conflict = output_dir / "metadata_predator.json"
    conflict.write_text("existing", encoding="utf-8")

    monkeypatch.setattr(
        train_module,
        "load_configs",
        create_configs,
    )

    def fail_if_training_starts(
        trainer,
        record_diagnostics=False,
    ):
        pytest.fail("Training must not start when outputs conflict.")

    monkeypatch.setattr(
        CoevolutionTrainer,
        "train",
        fail_if_training_starts,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--species",
            "both",
            "--record-diagnostics",
            "--output",
            str(output_dir),
        ],
    )

    with pytest.raises(FileExistsError) as error:
        train_module.main()

    assert str(conflict) in str(error.value)
    assert conflict.read_text(encoding="utf-8") == "existing"


def test_cli_passes_diagnostics_option_to_coevolution_trainer(
    monkeypatch,
    tmp_path,
):
    calls = []

    class FakeCoevolutionTrainer:
        def __init__(
            self,
            simulation_config,
            neural_config,
            evolution_config,
        ):
            self.prey_trainer = SimpleNamespace(
                best_fitness=0.5,
                best_generation=1,
            )
            self.predator_trainer = SimpleNamespace(
                best_fitness=0.5,
                best_generation=1,
            )

        @staticmethod
        def validate_output_directory(output_dir):
            calls.append(("validate", output_dir))

        def train(self, record_diagnostics=False):
            calls.append(("train", record_diagnostics))

        @staticmethod
        def save_results(output_dir):
            return {}

    monkeypatch.setattr(
        train_module,
        "CoevolutionTrainer",
        FakeCoevolutionTrainer,
    )
    monkeypatch.setattr(
        train_module,
        "load_configs",
        create_configs,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--species",
            "both",
            "--record-diagnostics",
            "--output",
            str(tmp_path),
        ],
    )

    train_module.main()

    assert calls == [
        ("validate", str(tmp_path)),
        ("train", True),
    ]
