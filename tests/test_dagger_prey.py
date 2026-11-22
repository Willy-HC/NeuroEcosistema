from types import SimpleNamespace

import numpy as np
import pytest

from neuroecosistema.agents.heuristic_controller import (
    calculate_heuristic_direction,
)
from neuroecosistema.agents.perception import perceive_prey
from neuroecosistema.agents.predator import Predator
from neuroecosistema.agents.prey import Prey
from neuroecosistema.dagger_prey import (
    SCENARIOS,
    behavior_score,
    collect_state_observations,
    meets_acceptance,
    run_dagger,
)
from neuroecosistema.neural_network.network import NeuralNetwork
from neuroecosistema.simulation.environment import Plant


def test_dagger_records_exact_perception_and_heuristic_direction():
    prey = Prey(x=100, y=120)
    environment = SimpleNamespace(
        prey=[prey],
        plants=[Plant(x=310, y=230)],
        predators=[Predator(x=450, y=400)],
        width=1000,
        height=700,
    )

    features, labels = collect_state_observations(environment)

    np.testing.assert_array_equal(
        features[0],
        perceive_prey(
            prey,
            environment.plants,
            environment.predators,
            environment.width,
            environment.height,
        ),
    )
    np.testing.assert_allclose(
        labels[0],
        calculate_heuristic_direction(prey, environment),
    )


def test_dagger_excludes_states_without_a_deterministic_food_target():
    environment = SimpleNamespace(
        prey=[Prey(x=100, y=120)],
        plants=[],
        predators=[Predator(x=450, y=400)],
        width=1000,
        height=700,
    )

    features, labels = collect_state_observations(environment)

    assert features.shape == (0, 8)
    assert labels.shape == (0, 2)


def test_behavior_selection_uses_food_and_survival_in_both_scenarios():
    random_baseline = {
        name: {
            "mean_food_eaten": 1.0,
            "mean_survival_steps": 400.0,
        }
        for name, _ in SCENARIOS
    }
    initial_policy = {
        name: {
            "mean_food_eaten": 0.8,
            "mean_survival_steps": 500.0,
        }
        for name, _ in SCENARIOS
    }
    balanced = {
        name: {
            "mean_food_eaten": 1.1,
            "mean_survival_steps": 470.0,
        }
        for name, _ in SCENARIOS
    }
    food_only = {
        name: {
            "mean_food_eaten": 2.0,
            "mean_survival_steps": 300.0,
        }
        for name, _ in SCENARIOS
    }

    assert meets_acceptance(balanced, random_baseline, initial_policy)
    assert not meets_acceptance(food_only, random_baseline, initial_policy)
    assert behavior_score(
        balanced,
        random_baseline,
        initial_policy,
    ) > behavior_score(food_only, random_baseline, initial_policy)


def test_acceptance_requires_food_gain_in_each_density():
    random_baseline = {
        "low_density": {
            "mean_food_eaten": 1.0,
            "mean_survival_steps": 400.0,
        },
        "high_density": {
            "mean_food_eaten": 1.0,
            "mean_survival_steps": 400.0,
        },
    }
    initial_policy = {
        name: {
            "mean_food_eaten": 0.5,
            "mean_survival_steps": 500.0,
        }
        for name, _ in SCENARIOS
    }
    candidate = {
        "low_density": {
            "mean_food_eaten": 1.1,
            "mean_survival_steps": 500.0,
        },
        "high_density": {
            "mean_food_eaten": 0.9,
            "mean_survival_steps": 500.0,
        },
    }

    assert not meets_acceptance(candidate, random_baseline, initial_policy)


def test_dagger_run_saves_candidate_without_overwriting_initial_model(
    tmp_path,
):
    neural_config = {
        "input_size": 8,
        "hidden_layers": [4],
        "output_size": 2,
        "activation_hidden": "relu",
        "activation_output": "tanh",
    }
    initial_model = tmp_path / "initial.npz"
    initial_network = NeuralNetwork(
        8,
        [4],
        2,
        seed=3,
    )
    np.savez_compressed(
        initial_model,
        genome=initial_network.get_genome(),
    )
    initial_bytes = initial_model.read_bytes()
    simulation_config = {
        "simulation": {
            "width": 1000,
            "height": 700,
            "max_steps": 2,
        },
        "population": {
            "prey": 10,
            "predators": 2,
        },
        "environment": {
            "initial_food": 30,
            "food_respawn": 0,
        },
        "energy": {
            "food_reward": 20.0,
            "movement_cost": 0.1,
            "capture_reward": 30.0,
        },
    }

    metrics = run_dagger(
        simulation_config,
        neural_config,
        initial_model_path=initial_model,
        output_dir=tmp_path / "dagger",
        seed=41,
        rounds=1,
        collection_steps=2,
        evaluation_steps=2,
        evaluation_seeds=(8765,),
        replay_samples=7,
    )

    assert initial_model.read_bytes() == initial_bytes
    assert metrics["rounds_run"] == 1
    assert metrics["initial_model_unchanged"]
    assert (tmp_path / "dagger" / "best_prey.npz").is_file()
    assert {
        row["policy"] for row in metrics["evaluation_rows"]
    } == {
        "initial_neural",
        "random",
        "heuristic",
        "dagger_round_1",
    }


def test_dagger_rejects_more_than_three_rounds(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 3"):
        run_dagger(
            {},
            {},
            initial_model_path=tmp_path / "missing.npz",
            output_dir=tmp_path / "dagger",
            rounds=4,
        )
