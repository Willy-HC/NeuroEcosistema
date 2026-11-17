
def normalize_value(value, maximum):
    """Normaliza un valor entre 0 y 1."""

    if maximum <= 0:
        raise ValueError(
            "El valor máximo debe ser mayor que cero."
        )

    return max(0.0, min(1.0, value / maximum))


def calculate_prey_fitness(
    prey,
    max_steps,
    food_target=5,
):
    """Calcula el fitness individual de una presa."""

    survival_score = normalize_value(
        prey.steps_survived,
        max_steps,
    )

    food_score = normalize_value(
        prey.food_eaten,
        food_target,
    )

    stagnation_score = normalize_value(
        prey.stagnation_steps,
        max_steps,
    )

    fitness = (
        0.6 * survival_score
        + 0.4 * food_score
        - 0.1 * stagnation_score
    )

    return max(0.0, fitness)


def calculate_predator_fitness(
    predator,
    max_steps,
    capture_target=3,
):
    """Calcula el fitness individual de un depredador."""

    survival_score = normalize_value(
        predator.steps_survived,
        max_steps,
    )

    capture_score = normalize_value(
        predator.captures,
        capture_target,
    )

    stagnation_score = normalize_value(
        predator.stagnation_steps,
        max_steps,
    )

    fitness = (
        0.7 * capture_score
        + 0.3 * survival_score
        - 0.1 * stagnation_score
    )

    return max(0.0, fitness)