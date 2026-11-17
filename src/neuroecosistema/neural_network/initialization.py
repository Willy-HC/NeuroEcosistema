
import numpy as np


def initialize_weights(input_size, output_size, method, rng):
    """Inicializa los pesos de una capa neuronal."""

    if method == "he":
        standard_deviation = np.sqrt(2.0 / input_size)

    elif method == "xavier":
        standard_deviation = np.sqrt(
            2.0 / (input_size + output_size)
        )

    else:
        raise ValueError(
            f"Método de inicialización no soportado: {method}"
        )

    return rng.normal(
        0.0,
        standard_deviation,
        size=(output_size, input_size),
    )


def initialize_biases(output_size):
    """Inicializa los sesgos de una capa en cero."""
    return np.zeros(output_size, dtype=np.float32)