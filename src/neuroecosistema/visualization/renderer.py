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

    def draw_agents(self, environment: Environment) -> None:
        """Dibuja las presas y los depredadores."""

        for agent in environment.prey:
            pygame.draw.circle(
                self.screen,
                agent.color,
                (int(agent.x), int(agent.y)),
                agent.radius,
            )

        for agent in environment.predators:
            pygame.draw.circle(
                self.screen,
                agent.color,
                (int(agent.x), int(agent.y)),
                agent.radius,
            )

    def draw_statistics(self, environment: Environment) -> None:
        """Muestra estadísticas básicas del ecosistema."""

        font = pygame.font.Font(None, 26)

        statistics = [
            f"Presas: {len(environment.prey)}",
            f"Depredadores: {len(environment.predators)}",
            f"Plantas: {len(environment.plants)}",
            f"Alimento consumido: {environment.total_food_consumed}",
            f"Capturas: {environment.total_captures}",
        ]

        for index, text in enumerate(statistics):
            surface = font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (15, 15 + index * 25))