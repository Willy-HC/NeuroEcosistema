
import numpy as np


def uniform_crossover(
    parent_a,
    parent_b,
    crossover_rate=0.70,
    rng=None,
):
    """Combina dos genomas mediante cruce uniforme."""

    parent_a = np.asarray(parent_a, dtype=np.float64)
    parent_b = np.asarray(parent_b, dtype=np.float64)

    if (
        parent_a.ndim != 1
        or parent_b.ndim != 1
        or parent_a.shape != parent_b.shape
        or parent_a.size == 0
    ):
        raise ValueError(
            "Los progenitores deben tener genomas "
            "unidimensionales del mismo tamaño."
        )

    if not np.all(np.isfinite(parent_a)) or not np.all(
        np.isfinite(parent_b)
    ):
        raise ValueError("Los genomas deben ser finitos.")

    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("La tasa de cruce debe estar entre 0 y 1.")

    if rng is None:
        rng = np.random.default_rng()

    if rng.random() >= crossover_rate:
        return parent_a.copy()

    mask = rng.random(parent_a.size) < 0.5

    child = np.where(mask, parent_a, parent_b)

    return child