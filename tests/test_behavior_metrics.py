import random

import pytest

from neuroecosistema.agents.prey import Prey


def test_movement_accumulates_distance():
    prey = Prey(x=100, y=100)

    prey.update(
        width=500,
        height=500,
        rng=random.Random(42),
        neural_direction=(1.0, 0.0),
    )

    assert prey.distance_traveled == pytest.approx(1.8)


def test_stationary_agent_does_not_accumulate_distance():
    prey = Prey(x=100, y=100)

    prey.update(
        width=500,
        height=500,
        rng=random.Random(42),
        neural_direction=(0.0, 0.0),
    )

    assert prey.distance_traveled == 0.0


def test_agent_near_wall_is_registered():
    prey = Prey(x=5, y=100)

    prey.update(
        width=500,
        height=500,
        rng=random.Random(42),
        neural_direction=(-1.0, 0.0),
    )

    assert prey.steps_near_wall == 1


def test_agent_in_center_is_not_near_wall():
    prey = Prey(x=250, y=250)

    prey.update(
        width=500,
        height=500,
        rng=random.Random(42),
        neural_direction=(1.0, 0.0),
    )

    assert prey.steps_near_wall == 0


def test_metrics_accumulate_across_steps():
    prey = Prey(x=5, y=100)

    for _ in range(10):
        prey.update(
            width=500,
            height=500,
            rng=random.Random(42),
            neural_direction=(-1.0, 0.0),
        )

    assert prey.steps_near_wall == 10
    assert prey.steps_survived == 10