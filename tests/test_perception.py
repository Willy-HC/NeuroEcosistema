
import numpy as np

from neuroecosistema.agents.perception import (
    perceive_predator,
    perceive_prey,
)
from neuroecosistema.agents.predator import Predator
from neuroecosistema.agents.prey import Prey
from neuroecosistema.simulation.environment import Plant


def test_prey_perception_has_eight_inputs():
    prey = Prey(x=100, y=100)
    plant = Plant(x=150, y=100)
    predator = Predator(x=200, y=200)

    inputs = perceive_prey(
        prey, [plant], [predator], 1000, 700
    )

    assert inputs.shape == (8,)
    assert inputs.dtype == np.float32


def test_predator_perception_has_eight_inputs():
    predator = Predator(x=100, y=100)
    prey = Prey(x=200, y=100)

    inputs = perceive_predator(
        predator, [prey], 1000, 700
    )

    assert inputs.shape == (8,)
    assert inputs.dtype == np.float32


def test_perception_handles_missing_entities():
    prey = Prey(x=100, y=100)

    inputs = perceive_prey(
        prey, [], [], 1000, 700
    )

    assert inputs.shape == (8,)
    assert np.all(np.isfinite(inputs))


def test_predator_perception_handles_missing_prey():
    predator = Predator(x=100, y=100)

    inputs = perceive_predator(
        predator, [], 1000, 700
    )

    assert inputs.shape == (8,)
    assert np.all(np.isfinite(inputs))