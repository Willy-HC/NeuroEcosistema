
from pathlib import Path

import numpy as np

from neuroecosistema.neural_network.model_loader import (
    load_trained_network,
)
from neuroecosistema.simulation.environment import Environment


class BehaviorEvaluator:
    """Evalúa el comportamiento de agentes en episodios sin Pygame."""

    def __init__(self, simulation_config, neural_config):
        self.simulation_config = simulation_config
        self.neural_config = neural_config

    def evaluate(self, seed=42, model_dir=None):
        """Ejecuta un episodio y devuelve métricas por especie."""

        if model_dir is not None:
            model_dir = Path(model_dir)

        config = self.simulation_config
        simulation = config["simulation"]
        population = config["population"]
        environment_config = config["environment"]
        energy = config["energy"]

        environment = Environment(
            width=simulation["width"],
            height=simulation["height"],
            initial_food=environment_config["initial_food"],
            seed=seed,
            food_energy=energy["food_reward"],
            food_respawn=environment_config.get(
                "food_respawn", 0
            ),
            max_food=environment_config["initial_food"],
        )

        environment.generate_agents(
            prey_quantity=population["prey"],
            predator_quantity=population["predators"],
        )

        # Conservar referencias incluso si mueren o son capturados.
        original_prey = environment.prey[:]
        original_predators = environment.predators[:]

        if model_dir is not None:
            environment.initialize_neural_networks(
                self.neural_config
            )

            models = (
                (
                    "prey",
                    original_prey,
                    environment.prey_networks,
                ),
                (
                    "predator",
                    original_predators,
                    environment.predator_networks,
                ),
            )

            for species, agents, networks in models:
                model_path = (
                    model_dir / f"best_{species}.npz"
                )

                trained_network = load_trained_network(
                    model_path,
                    self.neural_config,
                )

                genome = trained_network.get_genome()

                for agent in agents:
                    networks[id(agent)].set_genome(genome)

        steps_executed = 0

        for _ in range(simulation["max_steps"]):
            if not environment.prey or not environment.predators:
                break

            environment.update_agents(
                movement_cost=energy["movement_cost"]
            )

            environment.resolve_interactions(
                capture_reward=energy["capture_reward"]
            )

            environment.update_food()
            steps_executed += 1

        return {
            "prey": self.summarize(
                original_prey,
                "prey",
                seed,
                steps_executed,
            ),
            "predator": self.summarize(
                original_predators,
                "predator",
                seed,
                steps_executed,
            ),
        }

    @staticmethod
    def summarize(agents, species, seed, steps_executed):
        """Resume las métricas conductuales de una especie."""

        if not agents:
            raise ValueError(
                "No se pueden evaluar poblaciones vacías."
            )

        def average(values):
            return float(np.mean(values))

        return {
            "seed": seed,
            "species": species,
            "agents": len(agents),
            "steps_executed": steps_executed,
            "mean_survival": average(
                [agent.steps_survived for agent in agents]
            ),
            "mean_food_eaten": average(
                [agent.food_eaten for agent in agents]
            ),
            "mean_captures": average(
                [agent.captures for agent in agents]
            ),
            "mean_exploration": average(
                [len(agent.visited_regions) for agent in agents]
            ),
            "mean_wall_ratio": average(
                [
                    agent.steps_near_wall
                    / max(1, agent.steps_survived)
                    for agent in agents
                ]
            ),
            "mean_distance": average(
                [agent.distance_traveled for agent in agents]
            ),
        }