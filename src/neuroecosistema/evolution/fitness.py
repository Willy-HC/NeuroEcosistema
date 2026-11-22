
import math
from numbers import Real

FITNESS_MODES = ("baseline", "food_progress")


def validate_fitness_options(fitness_mode, food_progress_weight):
    """Validate the optional prey fitness contribution."""

    if fitness_mode not in FITNESS_MODES:
        raise ValueError(
            f"Modo de fitness no válido: {fitness_mode}"
        )

    if (
        isinstance(food_progress_weight, bool)
        or not isinstance(food_progress_weight, Real)
        or not math.isfinite(float(food_progress_weight))
        or food_progress_weight < 0
    ):
        raise ValueError(
            "El peso del progreso alimentario debe ser finito "
            "y no negativo."
        )

    effective_weight = (
        float(food_progress_weight)
        if fitness_mode == "food_progress"
        else 0.0
    )
    return fitness_mode, effective_weight


def normalize_value(value, maximum):
    """Normaliza un valor entre 0 y 1."""

    if maximum <= 0:
        raise ValueError(
            "El valor máximo debe ser mayor que cero."
        )

    return max(0.0, min(1.0, value / maximum))



def calculate_prey_fitness_components(
    prey,
    max_steps,
    food_target=5,
    *,
    fitness_mode="baseline",
    food_progress_weight=0.05,
):
    """Calcula las contribuciones ponderadas del fitness de una presa."""

    fitness_mode, food_progress_weight = validate_fitness_options(
        fitness_mode,
        food_progress_weight,
    )

    survival_score = normalize_value(
        prey.steps_survived,
        max_steps,
    )

    food_score = normalize_value(
        prey.food_eaten,
        food_target,
    )

    exploration_score = normalize_value(
        len(prey.visited_regions),
        25,
    )

    stagnation_score = normalize_value(
        prey.stagnation_steps,
        max_steps,
    )

    fitness = (
        0.25 * survival_score
        + 0.50 * food_score
        + 0.20 * exploration_score
        - 0.10 * stagnation_score
    )

    components = {
        "survival": 0.25 * survival_score,
        "food": 0.50 * food_score,
        "exploration": 0.20 * exploration_score,
        "stagnation": -0.10 * stagnation_score,
        "fitness": max(0.0, fitness),
    }

    if fitness_mode == "baseline":
        return components

    progress_score = prey.food_progress_score
    if (
        not math.isfinite(progress_score)
        or not 0.0 <= progress_score <= 1.0
    ):
        raise ValueError(
            "El progreso alimentario debe estar entre 0 y 1."
        )

    progress_contribution = food_progress_weight * progress_score
    components["food_progress"] = progress_contribution
    components["fitness"] = max(
        0.0,
        min(1.0, fitness + progress_contribution),
    )
    return components


def calculate_prey_fitness(
    prey,
    max_steps,
    food_target=5,
    *,
    fitness_mode="baseline",
    food_progress_weight=0.05,
):
    """Evalúa supervivencia, alimentación y exploración."""

    return calculate_prey_fitness_components(
        prey,
        max_steps,
        food_target,
        fitness_mode=fitness_mode,
        food_progress_weight=food_progress_weight,
    )["fitness"]


def calculate_predator_fitness(
    predator,
    max_steps,
    capture_target=3,
):
    """Evalúa capturas, supervivencia y exploración."""

    survival_score = normalize_value(
        predator.steps_survived,
        max_steps,
    )

    capture_score = normalize_value(
        predator.captures,
        capture_target,
    )

    exploration_score = normalize_value(
        len(predator.visited_regions),
        25,
    )

    stagnation_score = normalize_value(
        predator.stagnation_steps,
        max_steps,
    )

    fitness = (
        0.65 * capture_score
        + 0.20 * survival_score
        + 0.15 * exploration_score
        - 0.10 * stagnation_score
    )

    return max(0.0, fitness)