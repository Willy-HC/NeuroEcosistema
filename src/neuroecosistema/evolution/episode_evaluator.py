
import numpy as np

from neuroecosistema.evolution.fitness import (
    calculate_predator_fitness,
    calculate_prey_fitness,
)
from neuroecosistema.neural_network.network import NeuralNetwork
from neuroecosistema.simulation.environment import Environment


class EpisodeEvaluator:
    """Evalúa genomas neuronales mediante episodios de simulación."""

    def __init__(
        self,
        simulation_config,
        neural_config,
        seed=42,
    ):
        self.simulation_config = simulation_config
        self.neural_config = neural_config
        self.seed = seed

    def create_network(self, genome):
        """Reconstruye una red neuronal desde su genoma."""

        network = NeuralNetwork(
            input_size=self.neural_config["input_size"],
            hidden_layers=self.neural_config["hidden_layers"],
            output_size=self.neural_config["output_size"],
            hidden_activation=self.neural_config["activation_hidden"],
            output_activation=self.neural_config["activation_output"],
            seed=self.seed,
        )

        network.set_genome(genome)

        return network

    def evaluate(self, genomes, species="prey"):
        """Evalúa una población de genomas de una especie."""

        if species not in ("prey", "predator"):
            raise ValueError("Especie no válida.")

        if not genomes:
            raise ValueError("La población no puede estar vacía.")

        config = self.simulation_config

        simulation = config["simulation"]
        population = config["population"]
        environment_config = config["environment"]
        energy = config["energy"]

        environment = Environment(
            width=simulation["width"],
            height=simulation["height"],
            initial_food=environment_config["initial_food"],
            seed=self.seed,
            food_energy=energy["food_reward"],
            food_respawn=environment_config.get(
                "food_respawn", 0
            ),
            max_food=environment_config["initial_food"],
        )

        if species == "prey":
            environment.generate_agents(
                prey_quantity=len(genomes),
                predator_quantity=population["predators"],
            )
            evaluated_agents = environment.prey[:]

            environment.prey_networks = {
                id(agent): self.create_network(genome)
                for agent, genome in zip(evaluated_agents, genomes)
            }

        else:
            environment.generate_agents(
                prey_quantity=population["prey"],
                predator_quantity=len(genomes),
            )
            evaluated_agents = environment.predators[:]

            environment.predator_networks = {
                id(agent): self.create_network(genome)
                for agent, genome in zip(evaluated_agents, genomes)
            }

        # Los oponentes sin red utilizan movimiento aleatorio.
        max_steps = simulation["max_steps"]


        for _ in range(max_steps):
            if not environment.prey or not environment.predators:
                break

            environment.update_agents(
                movement_cost=energy["movement_cost"]
            )
            
            environment.resolve_interactions(
                capture_reward=energy["capture_reward"]
            )

            environment.update_food()

        # evaluated_agents conserva referencias a todos los
        # individuos originales, incluso si fueron eliminados
        # de las listas de agentes vivos.
        if species == "prey":
            fitness_values = [
                calculate_prey_fitness(agent, max_steps)
                for agent in evaluated_agents
            ]
        else:
            fitness_values = [
                calculate_predator_fitness(agent, max_steps)
                for agent in evaluated_agents
            ]

        return np.asarray(fitness_values, dtype=np.float64)