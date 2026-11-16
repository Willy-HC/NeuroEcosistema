import random
from dataclasses import dataclass


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

        self.generate_plants(initial_food)

    def generate_plants(self, quantity: int) -> None:
        """Distribuye las plantas aleatoriamente en el entorno."""

        margin = 20

        for _ in range(quantity):
            x = self.random.randint(margin, self.width - margin)
            y = self.random.randint(margin, self.height - margin)

            self.plants.append(Plant(x=x, y=y))