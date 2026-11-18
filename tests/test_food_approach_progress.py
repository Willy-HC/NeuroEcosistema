
import pytest

from neuroecosistema.agents.prey import Prey
from neuroecosistema.simulation.environment import (
    Environment,
    Plant,
)


def create_environment():
    return Environment(
        width=500,
        height=500,
        initial_food=0,
        seed=42,
    )


def test_heuristic_prey_accumulates_food_progress():
    environment = create_environment()

    prey = Prey(x=100, y=100)
    environment.prey = [prey]
    environment.plants = [Plant(x=200, y=100)]

    environment.update_agents(strategy="heuristic")

    assert prey.food_approach_progress == pytest.approx(
        prey.speed
    )


def test_no_plants_produces_zero_progress():
    environment = create_environment()

    prey = Prey(x=100, y=100)
    environment.prey = [prey]

    environment.update_agents(strategy="heuristic")

    assert prey.food_approach_progress == 0.0


def test_moving_away_does_not_add_progress():
    environment = create_environment()

    prey = Prey(x=100, y=100)
    prey.direction = 0.0

    environment.prey = [prey]
    environment.plants = [Plant(x=50, y=100)]

    environment.update_agents(strategy="random")

    assert prey.food_approach_progress == 0.0


def test_progress_accumulates_across_steps():
    environment = create_environment()

    prey = Prey(x=100, y=100)
    environment.prey = [prey]
    environment.plants = [Plant(x=200, y=100)]

    for _ in range(3):
        environment.update_agents(strategy="heuristic")

    assert prey.food_approach_progress == pytest.approx(
        prey.speed * 3
    )


def test_progress_is_independent_for_each_prey():
    environment = create_environment()

    first = Prey(x=100, y=100)
    second = Prey(x=300, y=100)

    environment.prey = [first, second]
    environment.plants = [Plant(x=200, y=100)]

    environment.update_agents(strategy="heuristic")

    assert first.food_approach_progress == pytest.approx(
        first.speed
    )

    assert second.food_approach_progress == pytest.approx(
        second.speed
    )
