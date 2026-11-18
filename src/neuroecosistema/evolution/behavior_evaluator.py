
from pathlib import Path

import numpy as np

from neuroecosistema.neural_network.model_loader import (
    load_trained_network,
)
from neuroecosistema.simulation.environment import Environment


class BehaviorEvaluator:
    """Evalúa el comportamiento de agentes sin Pygame."""

    def __init__(self, simulation_config, neural_config):
        self.simulation_config = simulation_config
        self.neural_config = neural_config

    def evaluate(
        self,
        seed=42,
        model_dir=None,
        strategy=None,
        prey_strategy=None,
        predator_strategy=None,
    ):
        """Evalúa un episodio con estrategias independientes."""

        if strategy is None:
            strategy = "neural" if model_dir is not None else "random"

        if prey_strategy is None:
            prey_strategy = strategy

        if predator_strategy is None:
            predator_strategy = strategy

        valid_strategies = {"random", "heuristic", "neural"}

        if strategy not in valid_strategies:
            raise ValueError("Estrategia general no válida.")

        if prey_strategy not in valid_strategies:
            raise ValueError("Estrategia de presas no válida.")

        if predator_strategy not in valid_strategies:
            raise ValueError("Estrategia de depredadores no válida.")

        neural_required = (
            prey_strategy == "neural"
            or predator_strategy == "neural"
        )

        if neural_required and model_dir is None:
            raise ValueError(
                "Las estrategias neuronales requieren model_dir."
            )

        if not neural_required and model_dir is not None:
            raise ValueError(
                "model_dir solo se utiliza con estrategias neuronales."
            )

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


        original_prey = environment.prey[:]
        original_predators = environment.predators[:]

        if neural_required:
            environment.initialize_neural_networks(
                self.neural_config
            )

            models = (
                (
                    "prey",
                    prey_strategy,
                    original_prey,
                    environment.prey_networks,
                ),
                (
                    "predator",
                    predator_strategy,
                    original_predators,
                    environment.predator_networks,
                ),
            )

            for species, selected_strategy, agents, networks in models:
                if selected_strategy != "neural":
                    networks.clear()
                    continue

                model_path = model_dir / f"best_{species}.npz"

                trained_network = load_trained_network(
                    model_path,
                    self.neural_config,
                )

                genome = trained_network.get_genome()

                for agent in agents:
                    networks[id(agent)].set_genome(genome)

        steps_executed = 0

        for _ in range(simulation["max_steps"]):
            if not environment.prey and not environment.predators:
                break

            environment.update_agents(
                movement_cost=energy["movement_cost"],
                prey_strategy=prey_strategy,
                predator_strategy=predator_strategy,
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