
import pytest

from neuroecosistema.agents.heuristic_controller import (
    calculate_heuristic_direction,
)
from neuroecosistema.agents.predator import Predator
from neuroecosistema.agents.prey import Prey
from neuroecosistema.simulation.environment import Environment, Plant


def create_environment():
    """Crea un entorno controlado para las pruebas."""

    return Environment(
        width=500,
        height=500,
        initial_food=0,
        seed=42,
    )


def test_prey_moves_toward_nearest_plant():
    environment = create_environment()

    environment.plants = [
        Plant(x=100, y=50),
        Plant(x=400, y=400),
    ]

    prey = Prey(x=50, y=50)

    direction = calculate_heuristic_direction(
        prey,
        environment,
    )

    assert direction == pytest.approx((1.0, 0.0))


def test_predator_moves_toward_nearest_prey():
    environment = create_environment()

    environment.prey = [
        Prey(x=100, y=50),
        Prey(x=400, y=400),
    ]

    predator = Predator(x=50, y=50)

    direction = calculate_heuristic_direction(
        predator,
        environment,
    )

    assert direction == pytest.approx((1.0, 0.0))


def test_prey_without_plants_returns_none():
    environment = create_environment()

    prey = Prey(x=50, y=50)

    direction = calculate_heuristic_direction(
        prey,
        environment,
    )

    assert direction is None


def test_predator_without_prey_returns_none():
    environment = create_environment()

    predator = Predator(x=50, y=50)

    direction = calculate_heuristic_direction(
        predator,
        environment,
    )

    assert direction is None


def test_dead_agent_returns_none():
    environment = create_environment()

    environment.plants = [
        Plant(x=100, y=50),
    ]

    prey = Prey(x=50, y=50)
    prey.alive = False

    direction = calculate_heuristic_direction(
        prey,
        environment,
    )

    assert direction is None


def test_prey_selects_nearest_target():
    environment = create_environment()

    environment.plants = [
        Plant(x=450, y=450),
        Plant(x=60, y=50),
    ]

    prey = Prey(x=50, y=50)

    direction = calculate_heuristic_direction(
        prey,
        environment,
    )

    assert direction == pytest.approx((1.0, 0.0))


def test_unsupported_agent_raises_type_error():
    environment = create_environment()

    class UnknownAgent:
        alive = True

    with pytest.raises(TypeError):
        calculate_heuristic_direction(
            UnknownAgent(),
            environment,
        )
