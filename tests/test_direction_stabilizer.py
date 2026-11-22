
import math

import pytest

from neuroecosistema.agents.direction_stabilizer import limit_turn


def direction_angle(direction):
    return math.degrees(math.atan2(direction[1], direction[0]))


def test_no_turn():
    result = limit_turn((1, 0), (1, 0), 45)
    assert result == pytest.approx((1, 0))


def test_turn_below_limit():
    requested = (
        math.cos(math.radians(30)),
        math.sin(math.radians(30)),
    )
    result = limit_turn((1, 0), requested, 45)
    assert direction_angle(result) == pytest.approx(30)


def test_turn_above_limit():
    result = limit_turn((1, 0), (0, 1), 45)
    assert direction_angle(result) == pytest.approx(45)


def test_opposite_direction():
    result = limit_turn((1, 0), (-1, 0), 45)
    assert direction_angle(result) == pytest.approx(45)


def test_negative_turn():
    result = limit_turn((1, 0), (0, -1), 45)
    assert direction_angle(result) == pytest.approx(-45)


def test_first_movement():
    result = limit_turn(None, (0, 1), 45)
    assert result == pytest.approx((0, 1))


def test_zero_requested_direction():
    result = limit_turn((1, 0), (0, 0), 45)
    assert result == (0.0, 0.0)


def test_zero_previous_direction():
    result = limit_turn((0, 0), (0, 1), 45)
    assert result == pytest.approx((0, 1))


def test_angle_wraparound():
    previous = (
        math.cos(math.radians(170)),
        math.sin(math.radians(170)),
    )
    requested = (
        math.cos(math.radians(-170)),
        math.sin(math.radians(-170)),
    )

    result = limit_turn(previous, requested, 45)

    assert direction_angle(result) == pytest.approx(-170)


@pytest.mark.parametrize("invalid_limit", [
    0,
    -10,
    181,
    math.inf,
    math.nan,
])
def test_invalid_turn_limit(invalid_limit):
    with pytest.raises(ValueError):
        limit_turn((1, 0), (0, 1), invalid_limit)


def test_non_finite_requested_direction():
    with pytest.raises(ValueError):
        limit_turn((1, 0), (math.nan, 1), 45)


def test_non_finite_previous_direction():
    with pytest.raises(ValueError):
        limit_turn((math.inf, 0), (0, 1), 45)