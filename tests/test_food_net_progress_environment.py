
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


def test_heuristic_movement_adds_net_progress():
    environment = create_environment()
    prey = Prey(x=100, y=100)

    environment.prey = [prey]
    environment.plants = [Plant(x=200, y=100)]

    environment.update_agents(strategy="heuristic")

    assert prey.food_net_progress == pytest.approx(
        prey.speed
    )


def test_net_progress_accumulates_across_steps():
    environment = create_environment()
    prey = Prey(x=100, y=100)

    environment.prey = [prey]
    environment.plants = [Plant(x=200, y=100)]

    for _ in range(3):
        environment.update_agents(strategy="heuristic")

    assert prey.food_net_progress == pytest.approx(
        prey.speed * 3
    )


def test_moving_away_does_not_add_net_progress():
    environment = create_environment()
    prey = Prey(x=100, y=100)
    prey.direction = 0.0

    environment.prey = [prey]
    environment.plants = [Plant(x=50, y=100)]

    environment.update_agents(strategy="random")

    assert prey.food_net_progress == 0.0


def test_oscillation_does_not_duplicate_net_progress():
    environment = create_environment()
    prey = Prey(x=100, y=100)

    environment.prey = [prey]
    environment.plants = [Plant(x=200, y=100)]

    prey.direction = 0.0
    environment.update_agents(strategy="random")

    initial_progress = prey.food_net_progress

    prey.direction = 3.141592653589793
    environment.update_agents(strategy="random")

    prey.direction = 0.0
    environment.update_agents(strategy="random")

    assert prey.food_net_progress == pytest.approx(
        initial_progress
    )


def test_no_food_does_not_add_net_progress():
    environment = create_environment()
    prey = Prey(x=100, y=100)

    environment.prey = [prey]

    environment.update_agents(strategy="heuristic")

    assert prey.food_net_progress == 0.0
    assert prey.food_target_id is None


def test_new_target_starts_new_tracking():
    environment = create_environment()
    prey = Prey(x=100, y=100)

    first_plant = Plant(x=200, y=100)
    second_plant = Plant(x=100, y=200)

    environment.prey = [prey]
    environment.plants = [first_plant]

    environment.update_agents(strategy="heuristic")

    first_progress = prey.food_net_progress

    environment.plants = [second_plant]
    environment.update_agents(strategy="heuristic")

    assert prey.food_target_id == id(second_plant)
    assert prey.food_net_progress > first_progress