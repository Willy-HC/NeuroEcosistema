from types import SimpleNamespace

import numpy as np
import pytest

from neuroecosistema.agents.heuristic_controller import (
    calculate_heuristic_direction,
)
from neuroecosistema.agents.predator import Predator
from neuroecosistema.agents.prey import Prey
from neuroecosistema.neural_network.model_loader import (
    load_trained_network,
)
from neuroecosistema.neural_network.network import NeuralNetwork
from neuroecosistema.pretrain_imitation import (
    CURRICULUM_PHASES,
    DISTANT_FOOD_THREAT_SCENARIO,
    PREY_CURRICULUM_PHASES,
    PREY_SCENARIO_TYPES,
    SCENARIO_TYPES,
    _direction_loss_and_gradient,
    _forward_with_cache,
    generate_teacher_dataset,
    imitation_teacher_direction,
    pretrain_species,
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


def test_imitation_teacher_keeps_food_direction_for_non_imminent_threat():
    prey = Prey(x=100, y=100)
    environment = SimpleNamespace(
        plants=[Plant(x=300, y=100)],
        predators=[Predator(x=200, y=100)],
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


def test_imitation_teacher_does_not_invent_direction_without_target():
    prey = Prey(x=100, y=100)
    environment = SimpleNamespace(
        plants=[],
        predators=[],
        prey=[],
        width=1000,
        height=700,
    )

    assert calculate_heuristic_direction(prey, environment) is None
    assert imitation_teacher_direction(
        prey,
        "prey",
        environment,
    ) == (0.0, 0.0)


def test_imitation_teacher_direction_matches_goal_controller_off_walls():
    prey = Prey(x=100, y=100)
    environment = SimpleNamespace(
        plants=[Plant(x=300, y=400)],
        predators=[],
        prey=[],
        width=1000,
        height=700,
    )

    assert imitation_teacher_direction(
        prey,
        "prey",
        environment,
    ) == calculate_heuristic_direction(prey, environment)


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


def test_curriculum_starts_with_goal_seeking_then_adds_safety():
    assert CURRICULUM_PHASES["goal_seeking"] == ("objective_only",)
    assert "threat_only" in CURRICULUM_PHASES["safety_and_mixed"]
    assert "objective_and_threat" in (
        CURRICULUM_PHASES["safety_and_mixed"]
    )
    assert "no_objective" in CURRICULUM_PHASES["safety_and_mixed"]


def test_prey_curriculum_rehearses_food_with_distant_predators_only():
    assert PREY_CURRICULUM_PHASES["goal_seeking"] == (
        CURRICULUM_PHASES["goal_seeking"]
    )
    assert PREY_CURRICULUM_PHASES["safety_and_mixed"].count(
        DISTANT_FOOD_THREAT_SCENARIO
    ) == 3
    assert DISTANT_FOOD_THREAT_SCENARIO not in (
        CURRICULUM_PHASES["safety_and_mixed"]
    )


def test_distant_predator_training_labels_keep_food_direction():
    features, labels, _ = generate_teacher_dataset(
        "prey",
        256,
        seed=91,
        scenario_types=(DISTANT_FOOD_THREAT_SCENARIO,),
    )
    map_diagonal = np.hypot(1000, 700)
    distant_and_safe = (
        (features[:, 5] * map_diagonal > 180.0)
        & (features[:, 7] > 0.2)
    )
    food_direction = np.stack(
        (features[:, 0] * 1000, features[:, 1] * 700),
        axis=1,
    )
    food_direction /= np.linalg.norm(
        food_direction,
        axis=1,
        keepdims=True,
    )
    alignments = np.sum(
        labels[distant_and_safe] * food_direction[distant_and_safe],
        axis=1,
    )

    assert np.all(features[:, 5] * map_diagonal > 180.0)
    assert np.count_nonzero(distant_and_safe) > 0
    assert np.all(alignments > 0.99)
    with pytest.raises(ValueError):
        generate_teacher_dataset(
            "predator",
            9,
            seed=92,
            scenario_types=(DISTANT_FOOD_THREAT_SCENARIO,),
        )


def test_direction_loss_gradient_matches_finite_differences():
    predictions = np.array([[0.3, -0.5], [-0.2, 0.6]])
    labels = np.array([[1.0, 0.0], [0.0, 1.0]])
    _, gradient = _direction_loss_and_gradient(predictions, labels)
    epsilon = 1e-6

    for row in range(predictions.shape[0]):
        for column in range(predictions.shape[1]):
            plus = predictions.copy()
            minus = predictions.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            plus_loss, _ = _direction_loss_and_gradient(plus, labels)
            minus_loss, _ = _direction_loss_and_gradient(minus, labels)
            numeric = (plus_loss - minus_loss) / (2 * epsilon)
            assert gradient[row, column] == pytest.approx(
                numeric,
                abs=1e-6,
            )


def test_cached_forward_uses_configured_network_activations():
    network = NeuralNetwork(
        8,
        [5],
        2,
        hidden_activation="tanh",
        output_activation="relu",
        seed=7,
    )
    features = np.array([[0.2, -0.1, 0.3, 0.1, 0.5, 0.8, 0.7, 0.4]])

    cached, _, _ = _forward_with_cache(network, features)

    np.testing.assert_allclose(cached, network.forward(features))


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


@pytest.mark.parametrize(
    ("species", "train_samples", "scenario_types"),
    (
        ("predator", 20, set(SCENARIO_TYPES)),
        ("prey", 28, set(PREY_SCENARIO_TYPES)),
    ),
)
def test_curriculum_pretraining_saves_compatible_model_and_metrics(
    tmp_path,
    species,
    train_samples,
    scenario_types,
):
    neural_config = {
        "input_size": 8,
        "hidden_layers": [8, 6],
        "output_size": 2,
        "activation_hidden": "relu",
        "activation_output": "tanh",
    }

    result = pretrain_species(
        species,
        tmp_path,
        neural_config,
        1000,
        700,
        seed=42,
        train_samples=train_samples,
        validation_samples=14,
        test_samples=14,
        max_epochs=2,
        batch_size=7,
        patience=1,
        learning_rate=0.001,
        l2=0.0001,
    )

    assert result["best_epochs_by_phase"].keys() == {
        "goal_seeking",
        "safety_and_mixed",
    }
    assert result["split_seeds"]["validation_safety"] != result["seed"]
    assert result["split_seeds"]["test_safety"] != (
        result["split_seeds"]["validation_safety"]
    )
    assert len(set(result["split_seeds"].values())) == len(
        result["split_seeds"]
    )
    assert result["metrics"]["test"]["evaluated_directions"] > 0
    assert set(result["scenario_metrics"]["test"]) == scenario_types
    if species == "prey":
        assert result["scenario_types_by_phase"][
            "safety_and_mixed"
        ].count(DISTANT_FOOD_THREAT_SCENARIO) == 3
    else:
        assert result["scenario_types_by_phase"] == {
            phase: list(scenarios)
            for phase, scenarios in CURRICULUM_PHASES.items()
        }
    assert 0.0 <= result["metrics"]["test"][
        "output_saturation_fraction"
    ] <= 1.0
    loaded = load_trained_network(
        tmp_path / f"best_{species}.npz",
        neural_config,
    )
    assert loaded.get_genome().shape == NeuralNetwork(
        8,
        [8, 6],
        2,
    ).get_genome().shape
