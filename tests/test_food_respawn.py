
from neuroecosistema.simulation.environment import Environment


def create_environment(
    initial_food=0,
    food_respawn=5,
    max_food=10,
    seed=42,
):
    return Environment(
        width=500,
        height=400,
        initial_food=initial_food,
        seed=seed,
        food_respawn=food_respawn,
        max_food=max_food,
    )


def test_food_respawns_at_configured_interval():
    environment = create_environment()

    for _ in range(4):
        environment.update_food()

    assert len(environment.plants) == 0

    environment.update_food()

    assert len(environment.plants) == 1
    assert environment.step_count == 5


def test_food_does_not_exceed_limit():
    environment = create_environment(
        initial_food=2,
        food_respawn=1,
        max_food=3,
    )

    for _ in range(20):
        environment.update_food()

    assert len(environment.plants) == 3


def test_food_respawn_can_be_disabled():
    environment = create_environment(
        food_respawn=0,
    )

    for _ in range(20):
        environment.update_food()

    assert len(environment.plants) == 0
    assert environment.step_count == 20


def test_food_respawn_is_reproducible():
    first = create_environment(seed=42)
    second = create_environment(seed=42)

    for _ in range(25):
        first.update_food()
        second.update_food()

    first_positions = [
        (plant.x, plant.y)
        for plant in first.plants
    ]

    second_positions = [
        (plant.x, plant.y)
        for plant in second.plants
    ]

    assert first_positions == second_positions
    assert len(first_positions) == 5


def test_food_respawn_replenishes_consumed_plants():
    environment = create_environment(
        initial_food=3,
        food_respawn=1,
        max_food=3,
    )

    environment.plants.pop()

    assert len(environment.plants) == 2

    environment.update_food()

    assert len(environment.plants) == 3


def test_invalid_food_configuration():
    import pytest

    with pytest.raises(ValueError):
        create_environment(
            initial_food=5,
            max_food=3,
        )

    with pytest.raises(ValueError):
        create_environment(
            food_respawn=-1,
        )