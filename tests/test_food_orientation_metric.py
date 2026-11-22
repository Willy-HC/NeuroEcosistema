
import math

import pytest

from neuroecosistema.analysis.food_orientation_metric import (
    normalized_net_progress,
)


def test_approach_is_positive():
    result = normalized_net_progress(200, 100, 1000)
    assert result == pytest.approx(0.1)


def test_recede_is_negative():
    result = normalized_net_progress(100, 200, 1000)
    assert result == pytest.approx(-0.1)


def test_return_to_start_is_zero():
    result = normalized_net_progress(200, 200, 1000)
    assert result == pytest.approx(0.0)


def test_stationary_is_zero():
    result = normalized_net_progress(100, 100, 1000)
    assert result == 0.0


def test_invalid_diagonal():
    with pytest.raises(ValueError):
        normalized_net_progress(100, 50, 0)


def test_invalid_distance():
    with pytest.raises(ValueError):
        normalized_net_progress(-10, 50, 1000)


def test_non_finite_value():
    with pytest.raises(ValueError):
        normalized_net_progress(math.nan, 50, 1000)