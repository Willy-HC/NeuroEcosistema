
import pytest

from neuroecosistema.agents.prey import Prey


def test_first_target_observation_does_not_add_progress():
    prey = Prey(x=100, y=100)

    prey.register_food_target_progress(
        target_id=1,
        distance=100.0,
    )

    assert prey.food_net_progress == 0.0
    assert prey.food_target_initial_distance == 100.0
    assert prey.food_target_min_distance == 100.0


def test_approaching_target_adds_net_progress():
    prey = Prey(x=100, y=100)

    prey.register_food_target_progress(1, 100.0)
    prey.register_food_target_progress(1, 80.0)

    assert prey.food_net_progress == pytest.approx(20.0)


def test_moving_away_does_not_add_progress():
    prey = Prey(x=100, y=100)

    prey.register_food_target_progress(1, 100.0)
    prey.register_food_target_progress(1, 80.0)
    prey.register_food_target_progress(1, 95.0)

    assert prey.food_net_progress == pytest.approx(20.0)


def test_returning_to_previous_minimum_does_not_add_progress():
    prey = Prey(x=100, y=100)

    prey.register_food_target_progress(1, 100.0)
    prey.register_food_target_progress(1, 80.0)
    prey.register_food_target_progress(1, 95.0)
    prey.register_food_target_progress(1, 80.0)

    assert prey.food_net_progress == pytest.approx(20.0)


def test_new_distance_record_adds_only_additional_progress():
    prey = Prey(x=100, y=100)

    prey.register_food_target_progress(1, 100.0)
    prey.register_food_target_progress(1, 80.0)
    prey.register_food_target_progress(1, 95.0)
    prey.register_food_target_progress(1, 70.0)

    assert prey.food_net_progress == pytest.approx(30.0)


def test_changing_target_starts_new_tracking():
    prey = Prey(x=100, y=100)

    prey.register_food_target_progress(1, 100.0)
    prey.register_food_target_progress(1, 80.0)

    prey.register_food_target_progress(2, 150.0)

    assert prey.food_net_progress == pytest.approx(20.0)
    assert prey.food_target_id == 2
    assert prey.food_target_initial_distance == 150.0
    assert prey.food_target_min_distance == 150.0


def test_negative_distance_raises_error():
    prey = Prey(x=100, y=100)

    with pytest.raises(ValueError):
        prey.register_food_target_progress(1, -10.0)