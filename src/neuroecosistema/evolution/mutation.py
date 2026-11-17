
import numpy as np


def gaussian_mutation(
    genome,
    mutation_rate=0.10,
    mutation_strength=0.15,
    rng=None,
):
    """Aplica mutación gaussiana a un genoma."""

    genome = np.asarray(genome, dtype=np.float64)

    if genome.ndim != 1 or genome.size == 0:
        raise ValueError("El genoma debe ser un vector no vacío.")

    if not np.all(np.isfinite(genome)):
        raise ValueError("El genoma debe contener valores finitos.")

    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(
            "La tasa de mutación debe estar entre 0 y 1."
        )

    if mutation_strength < 0 or not np.isfinite(
        mutation_strength
    ):
        raise ValueError(
            "La intensidad de mutación debe ser finita y no negativa."
        )

    if rng is None:
        rng = np.random.default_rng()

    mutated_genome = genome.copy()

    mutation_mask = (
        rng.random(genome.size) < mutation_rate
    )

    mutations = rng.normal(
        loc=0.0,
        scale=mutation_strength,
        size=genome.size,
    )

    mutated_genome[mutation_mask] += mutations[mutation_mask]

    return mutated_genome