
import math


def normalized_net_progress(
    initial_distance: float,
    final_distance: float,
    map_diagonal: float,
) -> float:
    """Calcula el progreso neto firmado hacia una referencia fija."""

    values = (initial_distance, final_distance, map_diagonal)

    if not all(math.isfinite(value) for value in values):
        raise ValueError("Las distancias deben ser finitas.")

    if initial_distance < 0 or final_distance < 0:
        raise ValueError("Las distancias no pueden ser negativas.")

    if map_diagonal <= 0:
        raise ValueError("La diagonal debe ser positiva.")

    progress = (initial_distance - final_distance) / map_diagonal

    return max(-1.0, min(1.0, progress))