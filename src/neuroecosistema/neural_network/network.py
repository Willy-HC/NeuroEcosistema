
import numpy as np

from neuroecosistema.neural_network.activations import get_activation
from neuroecosistema.neural_network.initialization import (
    initialize_biases,
    initialize_weights,
)


class NeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        hidden_activation="relu",
        output_activation="tanh",
        seed=None,
    ):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        self.hidden_activation_name = hidden_activation
        self.output_activation_name = output_activation

        self.hidden_activation = get_activation(hidden_activation)
        self.output_activation = get_activation(output_activation)

        self.rng = np.random.default_rng(seed)

        layer_sizes = [
            input_size,
            *hidden_layers,
            output_size,
        ]

        self.weights = []
        self.biases = []

        for index in range(len(layer_sizes) - 1):
            current_size = layer_sizes[index]
            next_size = layer_sizes[index + 1]

            if index == len(layer_sizes) - 2:
                method = "xavier"
            else:
                method = "he"

            weights = initialize_weights(
                current_size,
                next_size,
                method,
                self.rng,
            )

            biases = initialize_biases(next_size)

            self.weights.append(weights)
            self.biases.append(biases)

    def forward(self, inputs):
        """Calcula la salida de la red neuronal."""

        values = np.asarray(inputs, dtype=np.float32)

        if values.shape[-1] != self.input_size:
            raise ValueError(
                f"La red espera {self.input_size} entradas, "
                f"pero recibió {values.shape[-1]}."
            )

        for index, (weights, biases) in enumerate(
            zip(self.weights, self.biases)
        ):
            values = np.dot(values, weights.T) + biases

            if index == len(self.weights) - 1:
                values = self.output_activation(values)
            else:
                values = self.hidden_activation(values)

        return values

    
    def get_genome(self):
        """Convierte todos los pesos y sesgos en un vector."""

        parameters = []

        for weights, biases in zip(self.weights, self.biases):
            parameters.extend(weights.flatten())
            parameters.extend(biases.flatten())

        return np.asarray(parameters, dtype=np.float64)

    def set_genome(self, genome):
        """Reconstruye los pesos y sesgos desde un genoma."""

        genome = np.asarray(genome, dtype=np.float64)

        expected_size = sum(
            weights.size + biases.size
            for weights, biases in zip(self.weights, self.biases)
        )

        if genome.ndim != 1 or genome.size != expected_size:
            raise ValueError(
                f"Se esperaban {expected_size} parámetros, "
                f"pero se recibió un genoma de forma {genome.shape}."
            )

        if not np.all(np.isfinite(genome)):
            raise ValueError(
                "El genoma contiene valores no finitos."
            )

        position = 0

        for index in range(len(self.weights)):
            weights = self.weights[index]
            biases = self.biases[index]

            weights_size = weights.size
            biases_size = biases.size

            self.weights[index] = genome[
                position:position + weights_size
            ].reshape(weights.shape).copy()

            position += weights_size

            self.biases[index] = genome[
                position:position + biases_size
            ].reshape(biases.shape).copy()

            position += biases_size

    def count_parameters(self):
        """Devuelve la cantidad total de parámetros de la red."""

        return sum(
            weights.size + biases.size
            for weights, biases in zip(self.weights, self.biases)
        )