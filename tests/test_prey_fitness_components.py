import pytest

from neuroecosistema.agents.prey import Prey
from neuroecosistema.evolution.fitness import (
    calculate_prey_fitness,
    calculate_prey_fitness_components,
)


def test_prey_fitness_components_match_known_metrics():
    prey = Prey(x=100, y=100)
    prey.steps_survived = 500
    prey.food_eaten = 2
    prey.visited_regions.update(
        {(column, 0) for column in range(10)}
    )
    prey.stagnation_steps = 250

    components = calculate_prey_fitness_components(
        prey,
        max_steps=1000,
    )

    assert components == pytest.approx(
        {
            "survival": 0.125,
            "food": 0.2,
            "exploration": 0.08,
            "stagnation": -0.025,
            "fitness": 0.38,
        }
    )


def test_prey_fitness_matches_component_fitness():
    prey = Prey(x=100, y=100)
    prey.steps_survived = 750
    prey.food_eaten = 3
    prey.visited_regions.update({(0, 0), (1, 0)})
    prey.stagnation_steps = 100

    components = calculate_prey_fitness_components(
        prey,
        max_steps=1000,
    )

    assert calculate_prey_fitness(prey, 1000) == components["fitness"]


@pytest.mark.parametrize(
    ("food_eaten", "expected_contribution"),
    [
        (0, 0.0),
        (1, 0.1),
        (2, 0.2),
        (3, 0.3),
        (4, 0.4),
        (5, 0.5),
        (6, 0.5),
    ],
)
def test_food_contribution_is_point_one_per_plant_up_to_five(
    food_eaten,
    expected_contribution,
):
    prey = Prey(x=100, y=100)
    prey.food_eaten = food_eaten

    components = calculate_prey_fitness_components(
        prey,
        max_steps=100,
    )

    assert components["food"] == pytest.approx(expected_contribution)


def test_survival_contribution_is_capped_at_point_two_five():
    prey = Prey(x=100, y=100)
    prey.steps_survived = 200

    components = calculate_prey_fitness_components(
        prey,
        max_steps=100,
    )

    assert components["survival"] == 0.25


def test_exploration_contribution_is_capped_at_point_two():
    prey = Prey(x=100, y=100)
    prey.visited_regions.update(
        {(column, 0) for column in range(30)}
    )

    components = calculate_prey_fitness_components(
        prey,
        max_steps=100,
    )

    assert components["exploration"] == 0.2


def test_stagnation_contribution_is_capped_at_negative_point_one():
    prey = Prey(x=100, y=100)
    prey.stagnation_steps = 200

    components = calculate_prey_fitness_components(
        prey,
        max_steps=100,
    )

    assert components["stagnation"] == -0.1


def test_stagnation_contribution_is_never_positive():
    prey = Prey(x=100, y=100)

    components = calculate_prey_fitness_components(
        prey,
        max_steps=100,
    )

    assert components["stagnation"] == 0.0


def test_fitness_has_a_zero_lower_bound():
    prey = Prey(x=100, y=100)
    prey.stagnation_steps = 100

    components = calculate_prey_fitness_components(
        prey,
        max_steps=100,
    )

    assert components["stagnation"] == -0.1
    assert components["fitness"] == 0.0
    assert calculate_prey_fitness(prey, 100) == 0.0


@pytest.mark.parametrize(
    ("max_steps", "food_target"),
    [
        (0, 5),
        (100, 0),
    ],
)
def test_invalid_normalization_maximum_still_raises(
    max_steps,
    food_target,
):
    prey = Prey(x=100, y=100)

    with pytest.raises(ValueError):
        calculate_prey_fitness_components(
            prey,
            max_steps,
            food_target,
        )

    with pytest.raises(ValueError):
        calculate_prey_fitness(
            prey,
            max_steps,
            food_target,
        )


def test_prey_without_food_can_outscore_prey_that_eats_one_plant():
    prey_without_food = Prey(x=100, y=100)
    prey_without_food.steps_survived = 100
    prey_without_food.visited_regions.update(
        {(column, 0) for column in range(25)}
    )

    prey_with_food = Prey(x=100, y=100)
    prey_with_food.steps_survived = 100
    prey_with_food.food_eaten = 1

    fitness_without_food = calculate_prey_fitness(
        prey_without_food,
        max_steps=100,
    )
    fitness_with_food = calculate_prey_fitness(
        prey_with_food,
        max_steps=100,
    )

    assert fitness_without_food == 0.45
    assert fitness_with_food == 0.35
    assert fitness_without_food > fitness_with_food
