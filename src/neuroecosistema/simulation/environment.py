import math
import random
from dataclasses import dataclass

from neuroecosistema.agents.predator import Predator
from neuroecosistema.agents.prey import Prey


@dataclass
class Plant:
    """Representa una fuente de alimento dentro del ecosistema."""

    x: float
    y: float
    energy: float = 20.0
    radius: int = 6


class Environment:
    """Administra el mundo, sus recursos y sus organismos."""

    def __init__(
        self,
        width: int,
        height: int,
        initial_food: int,
        seed: int | None = None,
        food_energy: float = 20.0,
    ) -> None:
        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.food_energy = food_energy

        self.plants: list[Plant] = []
        self.prey: list[Prey] = []
        self.predators: list[Predator] = []

        self.total_food_consumed = 0
        self.total_captures = 0

        self.generate_plants(initial_food)

    def generate_plants(self, quantity: int) -> None:
        """Distribuye plantas aleatoriamente en el entorno."""

        margin = 20

        for _ in range(quantity):
            x = self.random.randint(margin, self.width - margin)
            y = self.random.randint(margin, self.height - margin)

            self.plants.append(
                Plant(x=x, y=y, energy=self.food_energy)
            )

    def generate_agents(
        self,
        prey_quantity: int,
        predator_quantity: int,
    ) -> None:
        """Genera las poblaciones iniciales de organismos."""

        margin = 30

        for _ in range(prey_quantity):
            x = self.random.randint(margin, self.width - margin)
            y = self.random.randint(margin, self.height - margin)

            self.prey.append(Prey(x=x, y=y))

        for _ in range(predator_quantity):
            x = self.random.randint(margin, self.width - margin)
            y = self.random.randint(margin, self.height - margin)

            self.predators.append(Predator(x=x, y=y))

    def update_agents(self, movement_cost: float = 0.1) -> None:
        """Actualiza el movimiento de todos los organismos."""

        for agent in self.prey + self.predators:
            agent.update(
                width=self.width,
                height=self.height,
                rng=self.random,
                movement_cost=movement_cost,
            )

        self.prey = [agent for agent in self.prey if agent.alive]
        self.predators = [
            agent for agent in self.predators if agent.alive
        ]

    def resolve_interactions(self, capture_reward: float = 30.0) -> None:
        """Resuelve la alimentación y las capturas del ciclo."""

        # Alimentación de las presas.
        for prey in self.prey:
            for plant in self.plants[:]:
                distance = math.dist(
                    (prey.x, prey.y),
                    (plant.x, plant.y),
                )

                collision_distance = prey.radius + plant.radius

                if distance <= collision_distance:
                    prey.energy = min(100.0, prey.energy + plant.energy)
                    self.plants.remove(plant)
                    self.total_food_consumed += 1
                    break

        # Cacería de los depredadores.
        remaining_prey = self.prey[:]

        for predator in self.predators:
            for prey in remaining_prey:
                distance = math.dist(
                    (predator.x, predator.y),
                    (prey.x, prey.y),
                )

                collision_distance = predator.radius + prey.radius

                if distance <= collision_distance:
                    prey.alive = False
                    predator.energy = min(
                        100.0,
                        predator.energy + capture_reward,
                    )

                    remaining_prey.remove(prey)
                    self.total_captures += 1
                    break

        self.prey = remaining_prey