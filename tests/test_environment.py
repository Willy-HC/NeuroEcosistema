import pytest

from neuroecosistema.simulation.environment import Environment

NETWORK_CONFIG = {
    "input_size": 8,
    "hidden_layers": [16, 12],
    "output_size": 2,
    "activation_hidden": "relu",
    "activation_output": "tanh",
}


def test_environment_generates_correct_number_of_plants():
    environment = Environment(1000, 700, 100, seed=42)

    assert len(environment.plants) == 100


def test_environment_generates_plants_inside_boundaries():
    environment = Environment(1000, 700, 100, seed=42)

    for plant in environment.plants:
        assert 20 <= plant.x <= 980
        assert 20 <= plant.y <= 680


def test_same_seed_generates_same_distribution():
    environment_a = Environment(1000, 700, 100, seed=42)
    environment_b = Environment(1000, 700, 100, seed=42)

    positions_a = [(plant.x, plant.y) for plant in environment_a.plants]
    positions_b = [(plant.x, plant.y) for plant in environment_b.plants]

    assert positions_a == positions_b


def test_stabilizer_history_is_cleared_when_networks_are_reinitialized():
    environment = Environment(
        500,
        400,
        0,
        seed=42,
        predator_max_turn_degrees=45,
    )
    environment.generate_agents(prey_quantity=0, predator_quantity=1)
    environment.predator_previous_directions[id(environment.predators[0])] = (
        1.0,
        0.0,
    )

    environment.initialize_neural_networks(NETWORK_CONFIG)

    assert environment.predator_previous_directions == {}
    assert id(environment.predators[0]) in environment.predator_networks


def test_stabilizer_history_is_removed_when_predator_dies(monkeypatch):
    from neuroecosistema.agents.predator import Predator
    from neuroecosistema.simulation import environment as environment_module

    environment = Environment(
        500,
        400,
        0,
        seed=42,
        predator_max_turn_degrees=45,
    )
    predator = Predator(x=200, y=200, energy=0.01)
    environment.predators.append(predator)
    environment.predator_networks[id(predator)] = object()
    monkeypatch.setattr(
        environment_module,
        "calculate_neural_direction",
        lambda *_: (1.0, 0.0),
    )

    environment.update_agents(movement_cost=1.0, strategy="neural")

    assert environment.predators == []
    assert id(predator) not in environment.predator_previous_directions
    assert id(predator) not in environment.predator_networks


def test_stabilizer_only_limits_neural_predators_and_never_changes_weights(
    monkeypatch,
):
    import math

    from neuroecosistema.simulation import environment as environment_module

    environment = Environment(
        500,
        400,
        0,
        seed=42,
        predator_max_turn_degrees=45,
    )
    environment.generate_agents(prey_quantity=1, predator_quantity=1)
    environment.initialize_neural_networks(NETWORK_CONFIG)
    prey = environment.prey[0]
    predator = environment.predators[0]
    prey_genome = environment.prey_networks[id(prey)].get_genome().copy()
    predator_genome = environment.predator_networks[id(predator)].get_genome().copy()
    environment.predator_previous_directions[id(predator)] = (1.0, 0.0)
    monkeypatch.setattr(
        environment_module,
        "calculate_neural_direction",
        lambda *_: (0.0, 1.0),
    )

    environment.update_agents(movement_cost=0.0, strategy="neural")

    assert math.degrees(predator.direction) == pytest.approx(45.0)
    assert math.degrees(prey.direction) == pytest.approx(90.0)
    assert environment.predator_networks[id(predator)].get_genome().tolist() == (
        predator_genome.tolist()
    )
    assert environment.prey_networks[id(prey)].get_genome().tolist() == (
        prey_genome.tolist()
    )


def test_stabilizer_does_not_change_heuristic_predator_direction(monkeypatch):
    import math

    import pytest

    from neuroecosistema.simulation import environment as environment_module

    environment = Environment(
        500,
        400,
        0,
        seed=42,
        predator_max_turn_degrees=30,
    )
    environment.generate_agents(prey_quantity=0, predator_quantity=1)
    environment.initialize_neural_networks(NETWORK_CONFIG)
    predator = environment.predators[0]
    environment.predator_previous_directions[id(predator)] = (1.0, 0.0)
    monkeypatch.setattr(
        environment_module,
        "calculate_heuristic_direction",
        lambda *_: (0.0, 1.0),
    )

    environment.update_agents(movement_cost=0.0, strategy="heuristic")

    assert math.degrees(predator.direction) == pytest.approx(90.0)
    assert environment.predator_previous_directions == {}