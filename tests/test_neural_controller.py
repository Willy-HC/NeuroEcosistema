
import numpy as np

from neuroecosistema.agents.neural_controller import (
    calculate_neural_direction,
)
from neuroecosistema.simulation.environment import Environment

NETWORK_CONFIG = {
    "input_size": 8,
    "hidden_layers": [16, 12],
    "output_size": 2,
    "activation_hidden": "relu",
    "activation_output": "tanh",
}


def create_environment():
    environment = Environment(
        width=1000,
        height=700,
        initial_food=10,
        seed=42,
    )

    environment.generate_agents(
        prey_quantity=3,
        predator_quantity=2,
    )

    environment.initialize_neural_networks(NETWORK_CONFIG)

    return environment


def test_each_agent_has_its_own_network():
    environment = create_environment()

    assert len(environment.prey_networks) == 3
    assert len(environment.predator_networks) == 2

    networks = [
        *environment.prey_networks.values(),
        *environment.predator_networks.values(),
    ]

    assert len({id(network) for network in networks}) == 5


def test_neural_direction_is_normalized():
    environment = create_environment()

    prey = environment.prey[0]
    network = environment.prey_networks[id(prey)]

    direction = calculate_neural_direction(
        prey,
        network,
        environment,
    )

    magnitude = np.linalg.norm(direction)

    assert np.isclose(magnitude, 1.0) or np.isclose(
        magnitude, 0.0
    )


def test_agents_move_using_neural_networks():
    environment = create_environment()

    prey = environment.prey[0]
    initial_position = (prey.x, prey.y)

    environment.update_agents(movement_cost=0.1)

    new_position = (prey.x, prey.y)

    assert new_position != initial_position


def test_networks_are_reproducible():
    environment_a = create_environment()
    environment_b = create_environment()

    prey_a = environment_a.prey[0]
    prey_b = environment_b.prey[0]

    network_a = environment_a.prey_networks[id(prey_a)]
    network_b = environment_b.prey_networks[id(prey_b)]

    inputs = np.ones(8, dtype=np.float32)

    np.testing.assert_array_equal(
        network_a.forward(inputs),
        network_b.forward(inputs),
    )