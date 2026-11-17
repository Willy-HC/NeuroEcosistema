
import math
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
        width,
        height,
        rng,
        movement_cost=0.1,
        neural_direction=None,
    ):
        """Actualiza el movimiento y la energía del agente."""

        if not self.alive:
            return

        if neural_direction is not None:
            dx, dy = neural_direction
            magnitude = math.hypot(dx, dy)

            if magnitude > 1e-8:
                self.direction = math.atan2(dy, dx)

                movement_x = self.speed * dx / magnitude
                movement_y = self.speed * dy / magnitude
            else:
                movement_x = 0.0
                movement_y = 0.0

        else:
            self.direction_timer += 1

            if self.direction_timer >= self.direction_change_interval:
                self.direction = rng.uniform(0, 2 * math.pi)
                self.direction_timer = 0

            movement_x = math.cos(self.direction) * self.speed
            movement_y = math.sin(self.direction) * self.speed

        self.x += movement_x
        self.y += movement_y

        self.x = max(self.radius, min(width - self.radius, self.x))
        self.y = max(self.radius, min(height - self.radius, self.y))

        self.energy -= movement_cost

        if self.energy <= 0:
            self.energy = 0
            self.alive = False