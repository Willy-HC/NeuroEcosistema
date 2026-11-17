
import pytest

from neuroecosistema.agents.predator import Predator
from neuroecosistema.agents.prey import Prey
from neuroecosistema.evolution.fitness import (
    calculate_predator_fitness,
    calculate_prey_fitness,
    normalize_value,
)
from neuroecosistema.simulation.environment import Environment


def test_normalize_value():
    assert normalize_value(50, 100) == 0.5
    assert normalize_value(150, 100) == 1.0
    assert normalize_value(-10, 100) == 0.0


def test_normalize_value_rejects_invalid_maximum():
    with pytest.raises(ValueError):
        normalize_value(10, 0)


def test_prey_fitness_increases_with_food():
    prey = Prey(x=100, y=100)
    prey.steps_survived = 1000

    fitness_without_food = calculate_prey_fitness(
        prey, max_steps=3000
    )

    prey.food_eaten = 3

    fitness_with_food = calculate_prey_fitness(
        prey, max_steps=3000
    )

    assert fitness_with_food > fitness_without_food


def test_prey_fitness_increases_with_survival():
    prey = Prey(x=100, y=100)

    prey.steps_survived = 500
    fitness_a = calculate_prey_fitness(prey, 3000)

    prey.steps_survived = 2000
    fitness_b = calculate_prey_fitness(prey, 3000)

    assert fitness_b > fitness_a


def test_predator_fitness_increases_with_captures():
    predator = Predator(x=100, y=100)
    predator.steps_survived = 1000

    fitness_without_captures = calculate_predator_fitness(
        predator, 3000
    )

    predator.captures = 2

    fitness_with_captures = calculate_predator_fitness(
        predator, 3000
    )

    assert fitness_with_captures > fitness_without_captures


def test_stagnation_reduces_fitness():
    prey = Prey(x=100, y=100)
    prey.steps_survived = 2000
    prey.food_eaten = 3

    fitness_without_stagnation = calculate_prey_fitness(
        prey, 3000
    )

    prey.stagnation_steps = 1000

    fitness_with_stagnation = calculate_prey_fitness(
        prey, 3000
    )

    assert fitness_with_stagnation < fitness_without_stagnation


def test_fitness_is_never_negative():
    prey = Prey(x=100, y=100)
    prey.stagnation_steps = 3000

    assert calculate_prey_fitness(prey, 3000) == 0.0


def test_food_consumption_updates_individual_statistics():
    environment = Environment(
        width=500,
        height=500,
        initial_food=0,
        seed=42,
    )

    prey = Prey(x=100, y=100)
    environment.prey = [prey]

    from neuroecosistema.simulation.environment import Plant

    environment.plants = [Plant(x=100, y=100)]

    environment.resolve_interactions()

    assert prey.food_eaten == 1
    assert environment.total_food_consumed == 1


def test_capture_updates_individual_statistics():
    environment = Environment(
        width=500,
        height=500,
        initial_food=0,
        seed=42,
    )

    prey = Prey(x=100, y=100)
    predator = Predator(x=100, y=100)

    environment.prey = [prey]
    environment.predators = [predator]

    environment.resolve_interactions()

    assert predator.captures == 1
    assert environment.total_captures == 1


def test_agent_survival_counter_increases():
    prey = Prey(x=100, y=100)

    import random

    prey.update(
        width=500,
        height=500,
        rng=random.Random(42),
    )

    assert prey.steps_survived == 1