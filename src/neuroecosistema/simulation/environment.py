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
    """Administra las dimensiones y los recursos del entorno."""

    def __init__(
        self,
        width: int,
        height: int,
        initial_food: int,
        seed: int | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.plants: list[Plant] = []

        self.prey: list[Prey] = []
        self.predators: list[Predator] = []

        self.generate_plants(initial_food)

    def generate_plants(self, quantity: int) -> None:
        """Distribuye las plantas aleatoriamente en el entorno."""

        margin = 20

        for _ in range(quantity):
            x = self.random.randint(margin, self.width - margin)
            y = self.random.randint(margin, self.height - margin)

            self.plants.append(Plant(x=x, y=y))

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