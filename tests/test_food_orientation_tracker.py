
import pytest

from neuroecosistema.analysis.food_orientation_tracker import (
    FoodOrientationTracker,
)


def test_approach_without_consumption():
    tracker = FoodOrientationTracker(map_diagonal=1000)
    tracker.start(plant_id=1, distance=200)
    tracker.observe(plant_id=1, distance=100)
    tracker.close("episode_end")

    assert tracker.score == pytest.approx(0.1)


def test_recede():
    tracker = FoodOrientationTracker(map_diagonal=1000)
    tracker.start(plant_id=1, distance=100)
    tracker.observe(plant_id=1, distance=200)
    tracker.close("episode_end")

    assert tracker.score == pytest.approx(-0.1)


def test_oscillation_has_zero_net_progress():
    tracker = FoodOrientationTracker(map_diagonal=1000)
    tracker.start(plant_id=1, distance=200)

    for _ in range(10):
        tracker.observe(1, 100)
        tracker.observe(1, 200)

    tracker.close("episode_end")

    assert tracker.score == pytest.approx(0.0)


def test_other_target_does_not_change_reference():
    tracker = FoodOrientationTracker(map_diagonal=1000)
    tracker.start(plant_id=1, distance=200)
    tracker.observe(plant_id=2, distance=20)

    assert tracker.score == 0.0


def test_disappeared_target_stops_progress():
    tracker = FoodOrientationTracker(map_diagonal=1000)
    tracker.start(plant_id=1, distance=200)
    tracker.observe(plant_id=1, distance=100)
    tracker.close("disappeared")

    tracker.observe(plant_id=1, distance=10)

    assert tracker.score == pytest.approx(0.1)


def test_self_consumption_is_recorded():
    tracker = FoodOrientationTracker(map_diagonal=1000)
    tracker.start(plant_id=1, distance=200)
    tracker.observe(plant_id=1, distance=11)
    tracker.close("self_consumed")

    assert tracker.close_reason == "self_consumed"
    assert tracker.score == pytest.approx(0.189)


def test_cannot_start_twice():
    tracker = FoodOrientationTracker(map_diagonal=1000)
    tracker.start(1, 200)

    with pytest.raises(RuntimeError):
        tracker.start(2, 100)


def test_cannot_close_twice():
    tracker = FoodOrientationTracker(map_diagonal=1000)
    tracker.start(1, 200)
    tracker.close("episode_end")

    with pytest.raises(RuntimeError):
        tracker.close("episode_end")    