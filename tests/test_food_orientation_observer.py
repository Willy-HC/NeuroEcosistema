
import pytest

from neuroecosistema.analysis.food_orientation_observer import (
    FoodOrientationObserver,
)
from neuroecosistema.simulation.environment import (
    Environment,
)


def make_environment():
    environment = Environment(
        width=1000,
        height=700,
        initial_food=0,
        seed=42,
        max_food=5,
    )
    environment.generate_agents(1, 0)
    return environment


def test_initial_reference_is_nearest():
    environment = make_environment()
    prey = environment.prey[0]

    prey.x = 100
    prey.y = 100

    first = environment.spawn_plant()
    first.x, first.y = 200, 100

    second = environment.spawn_plant()
    second.x, second.y = 500, 500

    observer = FoodOrientationObserver(environment)

    tracker = observer.trackers[id(prey)]
    assert tracker.target_id == id(first)


def test_approach_without_consumption():
    environment = make_environment()
    prey = environment.prey[0]
    prey.x, prey.y = 100, 100

    plant = environment.spawn_plant()
    plant.x, plant.y = 300, 100

    observer = FoodOrientationObserver(environment)

    prey.x = 200
    observer.after_movement()
    observer.finish()

    result = observer.results()[id(prey)]

    assert result["score"] > 0
    assert result["close_reason"] == "episode_end"


def test_disappeared_plant_stops_progress():
    environment = make_environment()
    prey = environment.prey[0]
    prey.x, prey.y = 100, 100

    plant = environment.spawn_plant()
    plant.x, plant.y = 300, 100

    observer = FoodOrientationObserver(environment)

    prey.x = 200
    observer.after_movement()

    environment.plants.remove(plant)
    observer.after_interactions([prey])

    score = observer.results()[id(prey)]["score"]

    prey.x = 290
    observer.after_movement()

    assert observer.results()[id(prey)]["score"] == score
    assert (
        observer.results()[id(prey)]["close_reason"]
        == "other_consumed"
    )


def test_own_consumption():
    environment = make_environment()
    prey = environment.prey[0]
    prey.x, prey.y = 100, 100

    plant = environment.spawn_plant()
    plant.x, plant.y = 200, 100

    observer = FoodOrientationObserver(environment)

    # Registrar referencias con la infraestructura existente.
    prey.synchronize_food_progress_references(
        environment.plants
    )

    prey.x, prey.y = 200, 100
    observer.after_movement()

    environment.resolve_interactions()
    observer.after_interactions([prey])

    result = observer.results()[id(prey)]

    assert result["close_reason"] == "self_consumed"
    assert result["score"] > 0


def test_no_initial_food():
    environment = make_environment()
    prey = environment.prey[0]

    observer = FoodOrientationObserver(environment)
    observer.finish()

    result = observer.results()[id(prey)]

    assert result["score"] == pytest.approx(0)
    assert result["target_id"] is None