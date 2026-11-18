
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
    """Evita paredes mediante recuperación con histéresis."""

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

    left = agent.x - agent.radius
    right = width - agent.radius - agent.x
    top = agent.y - agent.radius
    bottom = height - agent.radius - agent.y

    nearest_distance = min(
        left, right, top, bottom
    )

    # Activar recuperación cerca de los límites.
    if nearest_distance <= margin:
        agent.wall_escape_active = True

    # Desactivar solo al alcanzar una zona más segura.
    if nearest_distance >= margin * 2:
        agent.wall_escape_active = False

    if agent.wall_escape_active:
        center_x = width / 2
        center_y = height / 2

        escape_dx = center_x - agent.x
        escape_dy = center_y - agent.y

        return normalize_direction(
            escape_dx,
            escape_dy,
        )

    return normalize_direction(dx, dy)


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