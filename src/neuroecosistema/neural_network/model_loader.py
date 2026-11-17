
from pathlib import Path

import numpy as np

from neuroecosistema.neural_network.network import NeuralNetwork


def load_trained_network(model_path, neural_config, seed=42):
    """Carga una red neuronal desde un genoma entrenado."""

    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el modelo: {model_path}"
        )

    with np.load(model_path, allow_pickle=False) as data:
        if "genome" not in data.files:
            raise ValueError(
                "El archivo no contiene un genoma válido."
            )

        genome = data["genome"].copy()

    network = NeuralNetwork(
        input_size=neural_config["input_size"],
        hidden_layers=neural_config["hidden_layers"],
        output_size=neural_config["output_size"],
        hidden_activation=neural_config["activation_hidden"],
        output_activation=neural_config["activation_output"],
        seed=seed,
    )

    network.set_genome(genome)

    return network