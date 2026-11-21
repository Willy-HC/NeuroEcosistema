
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


def test_first_target_does_not_count_as_switch():
    prey = Prey(x=100, y=100)

    prey.register_food_target_progress(1, 100.0)

    assert prey.food_target_switches == 0


def test_switching_to_another_target_increments_counter():
    prey = Prey(x=100, y=100)

    prey.register_food_target_progress(1, 100.0)
    prey.register_food_target_progress(2, 150.0)

    assert prey.food_target_switches == 1


def test_repeating_same_target_does_not_increment_counter():
    prey = Prey(x=100, y=100)

    prey.register_food_target_progress(1, 100.0)
    prey.register_food_target_progress(1, 80.0)
    prey.register_food_target_progress(1, 70.0)

    assert prey.food_target_switches == 0


def test_returning_to_previous_target_counts_as_switch():
    prey = Prey(x=100, y=100)

    prey.register_food_target_progress(1, 100.0)
    prey.register_food_target_progress(2, 150.0)
    prey.register_food_target_progress(1, 90.0)

    assert prey.food_target_switches == 2


def test_no_plants_clears_target_tracking():
    environment = create_environment()
    prey = Prey(x=100, y=100)

    environment.prey = [prey]
    environment.plants = [Plant(x=200, y=100)]

    environment.update_agents(strategy="heuristic")

    assert prey.food_target_id is not None

    environment.plants.clear()
    environment.update_agents(strategy="heuristic")

    assert prey.food_target_id is None
    assert prey.food_target_initial_distance == 0.0
    assert prey.food_target_min_distance == 0.0


def test_environment_detects_target_switch():
    environment = create_environment()
    prey = Prey(x=100, y=100)

    first_plant = Plant(x=200, y=100)
    second_plant = Plant(x=100, y=200)

    environment.prey = [prey]
    environment.plants = [first_plant]

    environment.update_agents(strategy="heuristic")

    assert prey.food_target_switches == 0

    environment.plants = [second_plant]
    environment.update_agents(strategy="heuristic")

    assert prey.food_target_switches == 1
    assert prey.food_target_id == id(second_plant)