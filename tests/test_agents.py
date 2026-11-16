
import random

from neuroecosistema.agents.predator import Predator
from neuroecosistema.agents.prey import Prey


def test_prey_has_expected_initial_attributes():
    prey = Prey(x=100, y=100)

    assert prey.energy == 100
    assert prey.speed == 1.8
    assert prey.alive


def test_predator_has_expected_initial_attributes():
    predator = Predator(x=100, y=100)

    assert predator.energy == 100
    assert predator.speed == 1.5
    assert predator.alive


def test_agent_consumes_energy():
    prey = Prey(x=100, y=100)

    prey.update(1000, 700, random.Random(42), movement_cost=0.5)

    assert prey.energy == 99.5


def test_agent_dies_when_energy_reaches_zero():
    prey = Prey(x=100, y=100, energy=0.05)

    prey.update(1000, 700, random.Random(42), movement_cost=0.1)

    assert not prey.alive


def test_agent_stays_inside_world():
    prey = Prey(x=998, y=698)

    for _ in range(100):
        prey.update(1000, 700, random.Random(42))

    assert prey.radius <= prey.x <= 1000 - prey.radius
    assert prey.radius <= prey.y <= 700 - prey.radius
