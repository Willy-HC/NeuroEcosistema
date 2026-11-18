
import math

import pytest

from neuroecosistema.agents.neural_controller import (
    apply_wall_avoidance,
    normalize_direction,
)
from neuroecosistema.agents.prey import Prey


@pytest.mark.parametrize(
    ("x", "y", "dx", "dy", "axis", "expected_sign"),
    [
        (5, 250, -1.0, 0.0, "x", 1),
        (495, 250, 1.0, 0.0, "x", -1),
        (250, 5, 0.0, -1.0, "y", 1),
        (250, 495, 0.0, 1.0, "y", -1),
    ],
)
def test_agent_escapes_wall(
    x, y, dx, dy, axis, expected_sign
):
    agent = Prey(x=x, y=y)

    result_dx, result_dy = apply_wall_avoidance(
        agent,
        dx,
        dy,
        width=500,
        height=500,
    )

    component = (
        result_dx if axis == "x" else result_dy
    )

    assert component * expected_sign > 0


def test_agent_does_not_get_stuck_at_corner():
    agent = Prey(x=5, y=5)

    dx, dy = apply_wall_avoidance(
        agent,
        dx=-1.0,
        dy=-1.0,
        width=500,
        height=500,
    )

    assert dx > 0
    assert dy > 0
    assert math.hypot(dx, dy) == pytest.approx(1.0)


def test_wall_avoidance_preserves_direction_in_center():
    agent = Prey(x=250, y=250)

    dx, dy = apply_wall_avoidance(
        agent,
        dx=0.6,
        dy=0.8,
        width=500,
        height=500,
    )

    assert dx == pytest.approx(0.6)
    assert dy == pytest.approx(0.8)

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


def test_escape_activates_near_wall():
    agent = Prey(x=5, y=250)

    dx, dy = apply_wall_avoidance(
        agent, -1.0, 0.0, 500, 500
    )

    assert agent.wall_escape_active
    assert dx > 0
    assert dy == pytest.approx(0.0)


def test_escape_remains_active_between_thresholds():
    agent = Prey(
        x=60,
        y=250,
        wall_escape_active=True,
    )

    dx, _ = apply_wall_avoidance(
        agent, -1.0, 0.0, 500, 500
    )

    assert agent.wall_escape_active
    assert dx > 0


def test_escape_deactivates_in_safe_area():
    agent = Prey(
        x=150,
        y=250,
        wall_escape_active=True,
    )

    dx, dy = apply_wall_avoidance(
        agent, -1.0, 0.0, 500, 500
    )

    assert not agent.wall_escape_active
    assert dx == pytest.approx(-1.0)
    assert dy == pytest.approx(0.0)


def test_escape_from_corner_moves_inward():
    agent = Prey(x=5, y=5)

    dx, dy = apply_wall_avoidance(
        agent, -1.0, -1.0, 500, 500
    )

    assert agent.wall_escape_active
    assert dx > 0
    assert dy > 0