
import math

from neuroecosistema.agents.perception import (
    perceive_predator,
    perceive_prey,
)


def calculate_neural_direction(agent, network, environment):
    """Calcula la dirección de movimiento mediante una red neuronal."""

    if agent in environment.prey:
        inputs = perceive_prey(
            agent,
            environment.plants,
            environment.predators,
            environment.width,
            environment.height,
        )

    elif agent in environment.predators:
        inputs = perceive_predator(
            agent,
            environment.prey,
            environment.width,
            environment.height,
        )

    else:
        raise ValueError("El agente no pertenece al entorno.")

    outputs = network.forward(inputs)

    dx = float(outputs[0])
    dy = float(outputs[1])

    magnitude = math.hypot(dx, dy)

    if magnitude > 1e-8:
        dx /= magnitude
        dy /= magnitude
    else:
        dx = 0.0
        dy = 0.0

    return dx, dy