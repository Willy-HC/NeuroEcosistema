
import random

import pytest

from neuroecosistema.agents.predator import Predator
from neuroecosistema.agents.prey import Prey
from neuroecosistema.evolution.fitness import (
    calculate_predator_fitness,
    calculate_prey_fitness,
)


def test_region_is_registered():
    prey = Prey(x=50, y=50)

    prey.register_region(500, 500)

    assert prey.visited_regions == {(0, 0)}


def test_multiple_regions_are_registered():
    prey = Prey(x=50, y=50)

    prey.register_region(500, 500)

    prey.x = 250
    prey.y = 250

    prey.register_region(500, 500)

    assert len(prey.visited_regions) == 2


def test_repeated_region_is_not_duplicated():
    prey = Prey(x=50, y=50)

    for _ in range(10):
        prey.register_region(500, 500)

    assert len(prey.visited_regions) == 1


def test_agent_update_registers_region():
    prey = Prey(x=100, y=100)

    prey.update(
        width=500,
        height=500,
        rng=random.Random(42),
        movement_cost=0.1,
        neural_direction=(1.0, 0.0),
    )

    assert len(prey.visited_regions) == 1


def test_prey_exploration_increases_fitness():
    prey = Prey(x=100, y=100)
    prey.steps_survived = 100

    initial_fitness = calculate_prey_fitness(
        prey, max_steps=100
    )

    prey.visited_regions.update(
        {(0, 0), (1, 0), (2, 0)}
    )

    new_fitness = calculate_prey_fitness(
        prey, max_steps=100
    )

    assert new_fitness > initial_fitness


def test_prey_food_has_greater_weight():
    prey = Prey(x=100, y=100)
    prey.steps_survived = 100

    prey.food_eaten = 1

    food_fitness = calculate_prey_fitness(
        prey, max_steps=100
    )

    prey.food_eaten = 0
    prey.visited_regions.update(
        {(0, 0), (1, 0), (2, 0)}
    )

    exploration_fitness = calculate_prey_fitness(
        prey, max_steps=100
    )

    assert food_fitness > exploration_fitness


def test_predator_captures_increase_fitness():
    predator = Predator(x=100, y=100)
    predator.steps_survived = 100

    initial_fitness = calculate_predator_fitness(
        predator, max_steps=100
    )

    predator.captures = 1

    new_fitness = calculate_predator_fitness(
        predator, max_steps=100
    )

    assert new_fitness > initial_fitness


def test_exploration_does_not_exceed_grid():
    prey = Prey(x=500, y=500)

    prey.register_region(500, 500)

    assert prey.visited_regions == {(4, 4)}


def test_invalid_grid_size():
    prey = Prey(x=100, y=100)

    with pytest.raises(ValueError):
        prey.register_region(500, 500, grid_size=0)
