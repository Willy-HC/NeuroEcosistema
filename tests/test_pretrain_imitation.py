from types import SimpleNamespace

import numpy as np
import pytest

from neuroecosistema.agents.predator import Predator
from neuroecosistema.agents.prey import Prey
from neuroecosistema.neural_network.network import NeuralNetwork
from neuroecosistema.pretrain_imitation import (
    SCENARIO_SPLITS,
    generate_teacher_dataset,
    imitation_teacher_direction,
    train_supervised_network,
)
from neuroecosistema.simulation.environment import Plant


def test_prey_imitation_teacher_prioritizes_close_threat():
    prey = Prey(x=100, y=100)
    environment = SimpleNamespace(
        plants=[Plant(x=300, y=100)],
        predators=[Predator(x=125, y=100)],
        prey=[],
        width=1000,
        height=700,
    )

    direction = imitation_teacher_direction(
        prey,
        "prey",
        environment,
    )

    assert direction[0] < -0.9


def test_imitation_teacher_uses_existing_wall_escape_rule():
    prey = Prey(x=10, y=350)
    environment = SimpleNamespace(
        plants=[Plant(x=0, y=350)],
        predators=[],
        prey=[],
        width=1000,
        height=700,
    )

    direction = imitation_teacher_direction(
        prey,
        "prey",
        environment,
    )

    assert direction[0] > 0.9


def test_synthetic_splits_are_seeded_and_have_eight_features():
    first = generate_teacher_dataset(
        "prey",
        14,
        seed=17,
    )
    repeated = generate_teacher_dataset(
        "prey",
        14,
        seed=17,
    )
    validation = generate_teacher_dataset(
        "prey",
        14,
        seed=18,
    )

    assert first[0].shape == (14, 8)
    assert first[1].shape == (14, 2)
    assert np.array_equal(first[0], repeated[0])
    assert np.array_equal(first[1], repeated[1])
    assert set(first[2]).isdisjoint(validation[2])
    assert np.all(np.isfinite(first[0]))
    assert np.all(np.isfinite(first[1]))


def test_training_validation_and_test_use_disjoint_scenario_types():
    split_names = tuple(SCENARIO_SPLITS)
    split_types = [
        set(SCENARIO_SPLITS[split])
        for split in split_names
    ]

    assert set.union(*split_types) == {
        "objective_only",
        "threat_only",
        "objective_and_threat",
        "no_objective",
        "near_wall_objective",
        "near_wall_threat",
        "multiple_targets",
    }
    assert all(
        split_types[left].isdisjoint(split_types[right])
        for left in range(len(split_types))
        for right in range(left + 1, len(split_types))
    )
    for split in split_names:
        _, _, scenario_ids = generate_teacher_dataset(
            "prey",
            7,
            seed=31,
            scenario_types=SCENARIO_SPLITS[split],
        )
        assert {
            scenario_id.rsplit(":", 1)[-1]
            for scenario_id in scenario_ids
        } == set(SCENARIO_SPLITS[split])


def test_supervised_trainer_returns_finite_validation_checkpoint():
    network = NeuralNetwork(
        input_size=8,
        hidden_layers=[8],
        output_size=2,
        seed=23,
    )
    train_x, train_y, _ = generate_teacher_dataset(
        "predator",
        28,
        seed=24,
    )
    validation_x, validation_y, _ = generate_teacher_dataset(
        "predator",
        14,
        seed=25,
    )

    trained, history, best_epoch = train_supervised_network(
        network,
        train_x,
        train_y,
        validation_x,
        validation_y,
        seed=26,
        max_epochs=3,
        batch_size=7,
        patience=2,
    )

    assert 1 <= best_epoch <= len(history) <= 3
    assert np.all(np.isfinite(trained.get_genome()))
    assert all(
        np.isfinite(row["validation_loss"])
        for row in history
    )


def test_training_rejects_non_finite_learning_rate():
    network = NeuralNetwork(
        input_size=8,
        hidden_layers=[4],
        output_size=2,
        seed=31,
    )
    features = np.zeros((7, 8), dtype=np.float32)
    labels = np.zeros((7, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        train_supervised_network(
            network,
            features,
            labels,
            features,
            labels,
            seed=32,
            max_epochs=1,
            learning_rate=float("nan"),
        )
