
import numpy as np


def relu(values):
    """Función de activación ReLU."""
    return np.maximum(0, values)


def tanh(values):
    """Función de activación tangente hiperbólica."""
    return np.tanh(values)


def get_activation(name):
    """Obtiene una función de activación por su nombre."""
    activations = {
        "relu": relu,
        "tanh": tanh,
    }

    if name not in activations:
        raise ValueError(
            f"Función de activación no soportada: {name}"
        )

    return activations[name]