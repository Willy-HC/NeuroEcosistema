from pathlib import Path

import pygame
import yaml

from neuroecosistema.simulation.environment import Environment
from neuroecosistema.visualization.renderer import Renderer


def load_configuration() -> tuple[dict, dict]:
    """Carga la configuración del ecosistema y de las redes neuronales."""

    project_root = Path(__file__).resolve().parents[2]

    simulation_path = project_root / "config" / "simulation.yaml"
    neural_path = project_root / "config" / "neural_network.yaml"

    with open(simulation_path, encoding="utf-8") as file:
        simulation_config = yaml.safe_load(file)

    with open(neural_path, encoding="utf-8") as file:
        neural_config = yaml.safe_load(file)["neural_network"]

    return simulation_config, neural_config


def main() -> None:
    pygame.init()

    # Cargar configuraciones
    config, neural_config = load_configuration()

    simulation_config = config["simulation"]
    environment_config = config["environment"]
    population_config = config["population"]
    energy_config = config["energy"]

    width = simulation_config["width"]
    height = simulation_config["height"]
    fps = simulation_config["fps"]

    movement_cost = energy_config["movement_cost"]

    # Inicializar ventana
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("NeuroEcosistema: Evolución artificial")

    clock = pygame.time.Clock()

    # Crear entorno
    environment = Environment(
        width=width,
        height=height,
        initial_food=environment_config["initial_food"],
        seed=42,
        food_energy=energy_config["food_reward"],
    )

    # Generar poblaciones
    environment.generate_agents(
        prey_quantity=population_config["prey"],
        predator_quantity=population_config["predators"],
    )

    # Inicializar redes neuronales independientes
    environment.initialize_neural_networks(neural_config)

    renderer = Renderer(screen)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Actualizar agentes mediante sus redes neuronales
        environment.update_agents(
            movement_cost=movement_cost
        )

        # Resolver alimentación y cacería
        environment.resolve_interactions(
            capture_reward=energy_config["capture_reward"]
        )

        # Renderizar ecosistema
        renderer.draw_environment(environment)
        renderer.draw_agents(environment)
        renderer.draw_statistics(environment)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()