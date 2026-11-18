
import math

from neuroecosistema.agents.neural_controller import normalize_direction
from neuroecosistema.agents.predator import Predator
from neuroecosistema.agents.prey import Prey


def calculate_heuristic_direction(agent, environment):
    """Calcula una dirección hacia el objetivo más cercano.

    Las presas buscan plantas y los depredadores buscan presas.
    Si no existe un objetivo, devuelve None.
    """

    if not agent.alive:
        return None

    if isinstance(agent, Prey):
        targets = environment.plants

    elif isinstance(agent, Predator):
        targets = environment.prey

    else:
        raise TypeError(
            "El controlador solo admite presas y depredadores."
        )

    if not targets:
        return None

    nearest_target = min(
        targets,
        key=lambda target: math.dist(
            (agent.x, agent.y),
            (target.x, target.y),
        ),
    )

    dx = nearest_target.x - agent.x
    dy = nearest_target.y - agent.y

    return normalize_direction(dx, dy)
