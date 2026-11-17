
import math

import pytest

from neuroecosistema.agents.neural_controller import (
    apply_wall_avoidance,
    normalize_direction,
)
from neuroecosistema.agents.prey import Prey


def test_normalize_direction():
    dx, dy = normalize_direction(3.0, 4.0)

    assert dx == pytest.approx(0.6)
    assert dy == pytest.approx(0.8)


def test_normalize_zero_direction():
    assert normalize_direction(0.0, 0.0) == (
        0.0,
        0.0,
    )


def test_interior_does_not_modify_direction():
    agent = Prey(x=250, y=200)

    result = apply_wall_avoidance(
        agent,
        1.0,
        0.0,
        width=500,
        height=400,
    )

    assert result == pytest.approx((1.0, 0.0))


def test_left_wall_pushes_right():
    agent = Prey(x=5, y=200)

    dx, dy = apply_wall_avoidance(
        agent,
        -1.0,
        0.0,
        width=500,
        height=400,
    )

    assert dx > 0
    assert dy == pytest.approx(0.0)


def test_right_wall_pushes_left():
    agent = Prey(x=495, y=200)

    dx, dy = apply_wall_avoidance(
        agent,
        1.0,
        0.0,
        width=500,
        height=400,
    )

    assert dx < 0
    assert dy == pytest.approx(0.0)


def test_top_wall_pushes_down():
    agent = Prey(x=250, y=5)

    dx, dy = apply_wall_avoidance(
        agent,
        0.0,
        -1.0,
        width=500,
        height=400,
    )

    assert dx == pytest.approx(0.0)
    assert dy > 0


def test_bottom_wall_pushes_up():
    agent = Prey(x=250, y=395)

    dx, dy = apply_wall_avoidance(
        agent,
        0.0,
        1.0,
        width=500,
        height=400,
    )

    assert dx == pytest.approx(0.0)
    assert dy < 0


def test_corner_pushes_towards_interior():
    agent = Prey(x=5, y=5)

    dx, dy = apply_wall_avoidance(
        agent,
        -0.7,
        -0.7,
        width=500,
        height=400,
    )

    assert dx > 0
    assert dy > 0
    assert math.hypot(dx, dy) == pytest.approx(1.0)


def test_invalid_margin():
    agent = Prey(x=250, y=200)

    with pytest.raises(ValueError):
        apply_wall_avoidance(
            agent,
            1.0,
            0.0,
            width=500,
            height=400,
            margin=0,
        )


def test_invalid_strength():
    agent = Prey(x=250, y=200)

    with pytest.raises(ValueError):
        apply_wall_avoidance(
            agent,
            1.0,
            0.0,
            width=500,
            height=400,
            strength=-1,
        )