import pygame

from neuroecosistema.simulation.environment import Environment


class Renderer:
    """Se encarga de representar visualmente el ecosistema."""

    BACKGROUND_COLOR = (30, 45, 35)
    PLANT_COLOR = (90, 190, 90)
    PLANT_CENTER_COLOR = (150, 220, 100)

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen

    def draw_environment(self, environment: Environment) -> None:
        """Dibuja el fondo y las plantas del entorno."""

        self.screen.fill(self.BACKGROUND_COLOR)

        for plant in environment.plants:
            position = (int(plant.x), int(plant.y))

            pygame.draw.circle(
                self.screen,
                self.PLANT_COLOR,
                position,
                plant.radius,
            )

            pygame.draw.circle(
                self.screen,
                self.PLANT_CENTER_COLOR,
                position,
                2,
            )