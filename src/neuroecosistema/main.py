import argparse
from pathlib import Path

import pygame
import yaml

from neuroecosistema.neural_network.model_loader import (
    load_trained_network,
)
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


def assign_trained_models(
    environment,
    neural_config,
    model_dir="results",
):
    """Asigna una copia del modelo entrenado a cada agente."""

    model_dir = Path(model_dir)

    models = (
        ("prey", environment.prey, environment.prey_networks),
        (
            "predator",
            environment.predators,
            environment.predator_networks,
        ),
    )

    for species, agents, networks in models:
        model_path = model_dir / f"best_{species}.npz"

        trained_network = load_trained_network(
            model_path,
            neural_config,
        )

        genome = trained_network.get_genome()

        for agent in agents:
            # Las redes ya se inicializaron en main().
            # Solo sustituimos sus parámetros.
            network = networks[id(agent)]
            network.set_genome(genome)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulación visual de NeuroEcosistema"
    )

    parser.add_argument(
        "--trained",
        action="store_true",
        help="Utilizar los mejores modelos entrenados",
    )

    parser.add_argument(
        "--model-dir",
        default="results",
        help="Directorio de los modelos guardados",
    )

    args = parser.parse_args()

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
        food_respawn=environment_config.get(
            "food_respawn", 0
        ),
        max_food=environment_config["initial_food"],
    )

    # Generar poblaciones
    environment.generate_agents(
        prey_quantity=population_config["prey"],
        predator_quantity=population_config["predators"],
    )

    # Inicializar redes neuronales independientes
    environment.initialize_neural_networks(neural_config)

    if args.trained:
        assign_trained_models(
            environment,
            neural_config,
            model_dir=args.model_dir,
        )

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
        environment.update_agents(movement_cost = movement_cost)
        environment.resolve_interactions(
            capture_reward=energy_config["capture_reward"]
        )
        environment.update_food()

        # Renderizar ecosistema
        renderer.draw_environment(environment)
        renderer.draw_agents(environment)
        renderer.draw_statistics(environment)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    main()