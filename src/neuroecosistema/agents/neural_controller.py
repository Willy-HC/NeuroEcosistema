
import math

from neuroecosistema.agents.perception import (
    perceive_predator,
    perceive_prey,
)


def normalize_direction(dx, dy):
    """Normaliza un vector de movimiento."""

    magnitude = math.hypot(dx, dy)

    if magnitude <= 1e-8:
        return 0.0, 0.0

    return dx / magnitude, dy / magnitude


def apply_wall_avoidance(
    agent,
    dx,
    dy,
    width,
    height,
    margin=40.0,
    strength=1.5,
):
    """Corrige gradualmente el movimiento cerca de los bordes."""

    if width <= 0 or height <= 0:
        raise ValueError(
            "Las dimensiones del entorno deben ser positivas."
        )

    if margin <= 0:
        raise ValueError(
            "El margen de seguridad debe ser positivo."
        )

    if strength < 0:
        raise ValueError(
            "La intensidad de corrección no puede ser negativa."
        )

    left_distance = agent.x
    right_distance = width - agent.x
    top_distance = agent.y
    bottom_distance = height - agent.y

    correction_x = 0.0
    correction_y = 0.0

    if left_distance < margin:
        correction_x += (
            1.0 - max(0.0, left_distance) / margin
        )

    if right_distance < margin:
        correction_x -= (
            1.0 - max(0.0, right_distance) / margin
        )

    if top_distance < margin:
        correction_y += (
            1.0 - max(0.0, top_distance) / margin
        )

    if bottom_distance < margin:
        correction_y -= (
            1.0 - max(0.0, bottom_distance) / margin
        )

    corrected_dx = dx + strength * correction_x
    corrected_dy = dy + strength * correction_y

    return normalize_direction(
        corrected_dx,
        corrected_dy,
    )


def calculate_neural_direction(agent, network, environment):
    """Calcula la dirección neuronal con protección espacial."""

    if any(
        candidate is agent
        for candidate in environment.prey
    ):
        inputs = perceive_prey(
            agent,
            environment.plants,
            environment.predators,
            environment.width,
            environment.height,
        )

    elif any(
        candidate is agent
        for candidate in environment.predators
    ):
        inputs = perceive_predator(
            agent,
            environment.prey,
            environment.width,
            environment.height,
        )

    else:
        raise ValueError(
            "El agente no pertenece al entorno."
        )

    outputs = network.forward(inputs)

    dx = float(outputs[0])
    dy = float(outputs[1])

    if not math.isfinite(dx) or not math.isfinite(dy):
        raise ValueError(
            "La red neuronal produjo una dirección no finita."
        )

    dx, dy = normalize_direction(dx, dy)

    return apply_wall_avoidance(
        agent,
        dx,
        dy,
        environment.width,
        environment.height,
    )