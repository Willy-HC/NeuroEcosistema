
import math
import random
from dataclasses import dataclass


@dataclass
class BaseAgent:
    """Clase base para los organismos del ecosistema."""

    x: float
    y: float
    energy: float = 100.0
    speed: float = 1.5
    radius: int = 6
    color: tuple[int, int, int] = (255, 255, 255)

    direction: float = 0.0
    direction_timer: int = 0
    direction_change_interval: int = 45
    alive: bool = True

    def update(
        self,
        width: int,
        height: int,
        rng: random.Random,
        movement_cost: float = 0.1,
    ) -> None:
        """Actualiza el movimiento y la energía del agente."""

        if not self.alive:
            return

        # Cambiar de dirección periódicamente.
        if self.direction_timer <= 0:
            self.direction = rng.uniform(0, 2 * math.pi)
            self.direction_timer = self.direction_change_interval

        self.direction_timer -= 1

        # Calcular desplazamiento.
        self.x += math.cos(self.direction) * self.speed
        self.y += math.sin(self.direction) * self.speed

        # Mantener al agente dentro del mundo.
        min_x = self.radius
        max_x = width - self.radius
        min_y = self.radius
        max_y = height - self.radius

        if self.x < min_x or self.x > max_x:
            self.direction = math.pi - self.direction
            self.x = max(min_x, min(self.x, max_x))

        if self.y < min_y or self.y > max_y:
            self.direction = -self.direction
            self.y = max(min_y, min(self.y, max_y))

        # Consumo de energía por movimiento.
        self.energy -= movement_cost

        if self.energy <= 0:
            self.energy = 0
            self.alive = False