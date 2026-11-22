
"""Estabilizacion experimental de direcciones de movimiento."""

import math


def limit_turn(previous_direction, requested_direction, max_turn_degrees):
    """Limita el giro entre dos direcciones.

    Devuelve una direccion unitaria. Si no existe una direccion
    anterior valida, utiliza directamente la solicitada.
    """
    if not math.isfinite(max_turn_degrees):
        raise ValueError("El limite de giro debe ser finito.")

    if not 0 < max_turn_degrees <= 180:
        raise ValueError(
            "El limite de giro debe estar entre 0 y 180 grados."
        )

    dx, dy = requested_direction

    if not math.isfinite(dx) or not math.isfinite(dy):
        raise ValueError("La direccion solicitada debe ser finita.")

    requested_magnitude = math.hypot(dx, dy)

    if requested_magnitude <= 1e-8:
        return 0.0, 0.0

    requested_angle = math.atan2(dy, dx)

    if previous_direction is None:
        return math.cos(requested_angle), math.sin(requested_angle)

    px, py = previous_direction

    if not math.isfinite(px) or not math.isfinite(py):
        raise ValueError("La direccion anterior debe ser finita.")

    if math.hypot(px, py) <= 1e-8:
        return math.cos(requested_angle), math.sin(requested_angle)

    previous_angle = math.atan2(py, px)

    angle_difference = math.atan2(
        math.sin(requested_angle - previous_angle),
        math.cos(requested_angle - previous_angle),
    )

    max_turn_radians = math.radians(max_turn_degrees)

    limited_difference = max(
        -max_turn_radians,
        min(max_turn_radians, angle_difference),
    )

    final_angle = previous_angle + limited_difference

    return math.cos(final_angle), math.sin(final_angle)