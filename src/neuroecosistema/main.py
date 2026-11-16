from pathlib import Path

import pygame
import yaml

from neuroecosistema.simulation.environment import Environment
from neuroecosistema.visualization.renderer import Renderer


def load_configuration() -> dict:
    """Carga la configuración principal del ecosistema."""

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / "simulation.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    pygame.init()

    config = load_configuration()

    simulation_config = config["simulation"]
    environment_config = config["environment"]

    width = simulation_config["width"]
    height = simulation_config["height"]
    fps = simulation_config["fps"]

    movement_cost = config["energy"]["movement_cost"]

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("NeuroEcosistema: Evolución artificial")

    clock = pygame.time.Clock()

    environment = Environment(
        width=width,
        height=height,
        initial_food=environment_config["initial_food"],
        seed=42,
    )

    population_config = config["population"]
    environment.generate_agents(
        prey_quantity=population_config["prey"],
        predator_quantity=population_config["predators"],
    )

    renderer = Renderer(screen)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        environment.update_agents(movement_cost)
        renderer.draw_environment(environment)
        renderer.draw_agents(environment)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()