
import pytest

from neuroecosistema.evolution.behavior_evaluator import (
    BehaviorEvaluator,
)
from neuroecosistema.simulation.environment import (
    Environment,
    Plant,
)

SIMULATION_CONFIG = {
    "simulation": {
        "width": 500,
        "height": 500,
        "max_steps": 20,
    },
    "population": {
        "prey": 3,
        "predators": 2,
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

NEURAL_CONFIG = {
    "input_size": 8,
    "hidden_layers": [16, 12],
    "output_size": 2,
    "activation_hidden": "relu",
    "activation_output": "tanh",
}


def create_evaluator():
    return BehaviorEvaluator(
        SIMULATION_CONFIG,
        NEURAL_CONFIG,
    )


@pytest.mark.parametrize(
    "strategy",
    ["random", "heuristic"],
)
def test_strategy_returns_species_metrics(strategy):
    evaluator = create_evaluator()

    results = evaluator.evaluate(
        seed=42,
        strategy=strategy,
    )

    assert set(results) == {"prey", "predator"}
    assert results["prey"]["agents"] == 3
    assert results["predator"]["agents"] == 2


def test_heuristic_strategy_is_reproducible():
    evaluator = create_evaluator()

    first = evaluator.evaluate(
        seed=42,
        strategy="heuristic",
    )

    second = evaluator.evaluate(
        seed=42,
        strategy="heuristic",
    )

    assert first == second


def test_invalid_strategy_raises_error():
    evaluator = create_evaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            strategy="invalid",
        )


def test_neural_strategy_requires_model():
    evaluator = create_evaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            strategy="neural",
        )


def test_random_strategy_rejects_model_directory():
    evaluator = create_evaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            strategy="random",
            model_dir="results/coevolution_01",
        )


def test_heuristic_prey_moves_toward_plant():
    environment = Environment(
        width=500,
        height=500,
        initial_food=0,
        seed=42,
    )

    environment.generate_agents(
        prey_quantity=1,
        predator_quantity=0,
    )

    prey = environment.prey[0]

    prey.x = 100
    prey.y = 100

    environment.plants = [
        Plant(x=200, y=100),
    ]

    previous_x = prey.x

    environment.update_agents(
        strategy="heuristic",
    )

    assert prey.x > previous_x
    assert prey.y == pytest.approx(100)


def test_random_strategy_does_not_use_neural_networks():
    environment = Environment(
        width=500,
        height=500,
        initial_food=0,
        seed=42,
    )

    environment.generate_agents(
        prey_quantity=1,
        predator_quantity=0,
    )

    prey = environment.prey[0]

    class FakeNetwork:
        def forward(self, inputs):
            raise AssertionError(
                "La red neuronal no debería ejecutarse."
            )

    environment.prey_networks[id(prey)] = FakeNetwork()

    environment.update_agents(
        strategy="random",
    )

    assert prey.steps_survived == 1


@pytest.mark.parametrize(
    ("prey_strategy", "predator_strategy"),
    [
        ("random", "random"),
        ("heuristic", "random"),
        ("random", "heuristic"),
        ("heuristic", "heuristic"),
    ],
)
def test_independent_strategies_return_metrics(
    prey_strategy,
    predator_strategy,
):
    evaluator = create_evaluator()

    results = evaluator.evaluate(
        seed=42,
        prey_strategy=prey_strategy,
        predator_strategy=predator_strategy,
    )

    assert set(results) == {"prey", "predator"}
    assert results["prey"]["agents"] == 3
    assert results["predator"]["agents"] == 2


def test_independent_strategies_are_reproducible():
    evaluator = create_evaluator()

    first = evaluator.evaluate(
        seed=42,
        prey_strategy="heuristic",
        predator_strategy="random",
    )

    second = evaluator.evaluate(
        seed=42,
        prey_strategy="heuristic",
        predator_strategy="random",
    )

    assert first == second


def test_independent_neural_strategy_requires_model():
    evaluator = create_evaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            prey_strategy="neural",
            predator_strategy="random",
        )


def test_invalid_prey_strategy_raises_error():
    evaluator = create_evaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            prey_strategy="invalid",
            predator_strategy="random",
        )


def test_invalid_predator_strategy_raises_error():
    evaluator = create_evaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            prey_strategy="random",
            predator_strategy="invalid",
        )
