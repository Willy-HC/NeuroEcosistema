
import numpy as np

from neuroecosistema.evolution.fitness import (
    calculate_predator_fitness,
    calculate_prey_fitness,
    calculate_prey_fitness_components,
    validate_fitness_options,
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
        fitness_mode="baseline",
        food_progress_weight=0.05,
    ):
        self.simulation_config = simulation_config
        self.neural_config = neural_config
        self.seed = seed
        (
            self.fitness_mode,
            self.food_progress_weight,
        ) = validate_fitness_options(
            fitness_mode,
            food_progress_weight,
        )

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

    @staticmethod
    def _create_prey_diagnostic(
        prey,
        genome_index,
        max_steps,
        fitness_mode="baseline",
        food_progress_weight=0.05,
    ):
        """Resume el fitness y las métricas de una presa evaluada."""

        components = calculate_prey_fitness_components(
            prey,
            max_steps,
            fitness_mode=fitness_mode,
            food_progress_weight=food_progress_weight,
        )

        return {
            "genome_index": genome_index,
            **components,
            "food_eaten": prey.food_eaten,
            "steps_survived": prey.steps_survived,
            "visited_regions": len(prey.visited_regions),
            "stagnation_steps": prey.stagnation_steps,
            "food_progress_score": prey.food_progress_score,
            "food_progress": components.get("food_progress", 0.0),
        }

    def evaluate(
        self,
        genomes,
        species="prey",
        include_diagnostics=False,
    ):
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
            if not environment.prey and not environment.predators:
                break

            environment.update_agents(
                movement_cost=energy["movement_cost"]
            )
            
            environment.resolve_interactions(
                capture_reward=energy["capture_reward"]
            )
            if species == "prey":
                environment.synchronize_food_progress(evaluated_agents)

            environment.update_food()
            if species == "prey":
                environment.synchronize_food_progress(evaluated_agents)

        # evaluated_agents conserva referencias a todos los
        # individuos originales, incluso si fueron eliminados
        # de las listas de agentes vivos.
        diagnostics = []
        if species == "prey":
            if include_diagnostics:
                diagnostics = [
                    self._create_prey_diagnostic(
                        agent,
                        genome_index,
                        max_steps,
                        self.fitness_mode,
                        self.food_progress_weight,
                    )
                    for genome_index, agent in enumerate(
                        evaluated_agents
                    )
                ]
                fitness_values = [
                    diagnostic["fitness"]
                    for diagnostic in diagnostics
                ]
            else:
                fitness_values = [
                    calculate_prey_fitness(
                        agent,
                        max_steps,
                        fitness_mode=self.fitness_mode,
                        food_progress_weight=self.food_progress_weight,
                    )
                    for agent in evaluated_agents
                ]
        else:
            fitness_values = [
                calculate_predator_fitness(agent, max_steps)
                for agent in evaluated_agents
            ]

        fitness = np.asarray(fitness_values, dtype=np.float64)

        if include_diagnostics:
            return fitness, diagnostics

        return fitness

    def evaluate_both(
        self,
        prey_genomes,
        predator_genomes,
        rotation=0,
        include_diagnostics=False,
    ):
        """Evalúa simultáneamente presas y depredadores neuronales."""

        if not prey_genomes or not predator_genomes:
            raise ValueError(
                "Ambas poblaciones deben contener genomas."
            )

        if len(prey_genomes) != len(predator_genomes):
            raise ValueError(
                "Las poblaciones deben tener el mismo tamaño."
            )

        config = self.simulation_config
        simulation = config["simulation"]
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

        population_size = len(prey_genomes)
        environment.generate_agents(
            prey_quantity=population_size,
            predator_quantity=population_size,
        )

        evaluated_prey = environment.prey[:]
        evaluated_predators = environment.predators[:]

        # Cada rotación cambia qué genoma ocupa cada posición
        # de depredador dentro del entorno.
        predator_indices = [
            (index + rotation) % population_size
            for index in range(population_size)
        ]

        environment.prey_networks = {
            id(agent): self.create_network(prey_genomes[index])
            for index, agent in enumerate(evaluated_prey)
        }

        environment.predator_networks = {
            id(agent): self.create_network(
                predator_genomes[predator_indices[index]]
            )
            for index, agent in enumerate(evaluated_predators)
        }

        max_steps = simulation["max_steps"]

        for _ in range(max_steps):
            if not environment.prey and not environment.predators:
                break

            environment.update_agents(
                movement_cost=energy["movement_cost"]
            )

            environment.resolve_interactions(
                capture_reward=energy["capture_reward"]
            )
            environment.synchronize_food_progress(evaluated_prey)

            environment.update_food()
            environment.synchronize_food_progress(evaluated_prey)

        diagnostics = []
        if include_diagnostics:
            diagnostics = [
                self._create_prey_diagnostic(
                    agent,
                    genome_index,
                    max_steps,
                    self.fitness_mode,
                    self.food_progress_weight,
                )
                for genome_index, agent in enumerate(evaluated_prey)
            ]
            prey_fitness_values = [
                diagnostic["fitness"]
                for diagnostic in diagnostics
            ]
        else:
            prey_fitness_values = [
                calculate_prey_fitness(
                    agent,
                    max_steps,
                    fitness_mode=self.fitness_mode,
                    food_progress_weight=self.food_progress_weight,
                )
                for agent in evaluated_prey
            ]

        prey_fitness = np.asarray(
            prey_fitness_values,
            dtype=np.float64,
        )

        predator_fitness = np.zeros(
            population_size,
            dtype=np.float64,
        )

        # Devolver cada fitness a su genoma original,
        # independientemente de la posición asignada.
        for position, agent in enumerate(evaluated_predators):
            genome_index = predator_indices[position]
            predator_fitness[genome_index] = (
                calculate_predator_fitness(agent, max_steps)
            )

        if include_diagnostics:
            return prey_fitness, predator_fitness, diagnostics

        return prey_fitness, predator_fitness
