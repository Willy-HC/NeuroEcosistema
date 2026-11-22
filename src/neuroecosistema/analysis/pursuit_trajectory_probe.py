
"""Herramientas para diagnosticar oscilaciones durante persecuciones."""

import math


def calculate_turn_angle(previous_direction, current_direction):
    """Calcula el giro absoluto entre dos vectores en grados."""
    px, py = previous_direction
    cx, cy = current_direction

    previous_magnitude = math.hypot(px, py)
    current_magnitude = math.hypot(cx, cy)

    if previous_magnitude <= 1e-8 or current_magnitude <= 1e-8:
        return None

    cosine = (
        px * cx + py * cy
    ) / (previous_magnitude * current_magnitude)

    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def calculate_path_efficiency(positions):
    """Calcula desplazamiento neto dividido por distancia recorrida."""
    if len(positions) < 2:
        return None

    total_distance = sum(
        math.dist(start, end)
        for start, end in zip(positions, positions[1:])
    )

    if total_distance <= 1e-8:
        return None

    net_distance = math.dist(positions[0], positions[-1])

    return net_distance / total_distance


def detect_oscillation(
    positions,
    angle_threshold=150.0,
    min_reversals=3,
    max_efficiency=0.25,
):
    """Detecta posibles oscilaciones en una ventana de posiciones."""
    if len(positions) < 3:
        return False

    directions = [
        (end[0] - start[0], end[1] - start[1])
        for start, end in zip(positions, positions[1:])
    ]

    reversals = 0

    for previous, current in zip(directions, directions[1:]):
        angle = calculate_turn_angle(previous, current)

        if angle is not None and angle >= angle_threshold:
            reversals += 1

    efficiency = calculate_path_efficiency(positions)

    return (
        efficiency is not None
        and reversals >= min_reversals
        and efficiency <= max_efficiency
    )