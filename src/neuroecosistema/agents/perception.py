
import math

import numpy as np


def normalize_position(observer, target, width, height):
    """Calcula la posición relativa normalizada entre dos entidades."""
    dx = (target.x - observer.x) / width
    dy = (target.y - observer.y) / height

    return dx, dy


def get_nearest_entities(observer, entities, quantity=1):
    """Obtiene las entidades más cercanas a un agente."""
    ordered_entities = sorted(
        entities,
        key=lambda entity: math.dist(
            (observer.x, observer.y),
            (entity.x, entity.y),
        ),
    )

    return ordered_entities[:quantity]


def get_entity_information(observer, entities, width, height):
    """Obtiene dirección y distancia normalizadas de la entidad más cercana."""
    nearest = get_nearest_entities(observer, entities)

    if not nearest:
        return 0.0, 0.0, 1.0

    target = nearest[0]

    dx, dy = normalize_position(
        observer,
        target,
        width,
        height,
    )

    distance = math.dist(
        (observer.x, observer.y),
        (target.x, target.y),
    )

    max_distance = math.hypot(width, height)
    normalized_distance = distance / max_distance

    return dx, dy, normalized_distance


def get_wall_distance(observer, width, height):
    """Calcula la distancia normalizada al borde más cercano."""
    distance = min(
        observer.x,
        width - observer.x,
        observer.y,
        height - observer.y,
    )

    max_distance = min(width, height) / 2

    return max(0.0, min(1.0, distance / max_distance))


def perceive_prey(prey, plants, predators, width, height):
    """Genera las ocho entradas sensoriales de una presa."""
    plant_info = get_entity_information(
        prey, plants, width, height
    )

    predator_info = get_entity_information(
        prey, predators, width, height
    )

    energy = max(0.0, min(1.0, prey.energy / 100.0))
    wall_distance = get_wall_distance(prey, width, height)

    return np.array(
        [
            *plant_info,
            *predator_info,
            energy,
            wall_distance,
        ],
        dtype=np.float32,
    )


def perceive_predator(predator, prey_agents, width, height):
    """Genera las ocho entradas sensoriales de un depredador."""
    nearest_prey = get_nearest_entities(
        predator,
        prey_agents,
        quantity=2,
    )

    prey_information = []

    for prey in nearest_prey:
        dx, dy = normalize_position(
            predator,
            prey,
            width,
            height,
        )

        distance = math.dist(
            (predator.x, predator.y),
            (prey.x, prey.y),
        )

        normalized_distance = distance / math.hypot(width, height)

        prey_information.extend(
            [dx, dy, normalized_distance]
        )

    while len(prey_information) < 6:
        prey_information.extend([0.0, 0.0, 1.0])

    energy = max(0.0, min(1.0, predator.energy / 100.0))
    wall_distance = get_wall_distance(
        predator,
        width,
        height,
    )

    return np.array(
        [
            *prey_information,
            energy,
            wall_distance,
        ],
        dtype=np.float32,
    )
