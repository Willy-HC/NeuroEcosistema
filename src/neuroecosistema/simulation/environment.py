import math
import random
from dataclasses import dataclass

from neuroecosistema.agents.heuristic_controller import (
    calculate_heuristic_direction,
)
from neuroecosistema.agents.neural_controller import (
    calculate_neural_direction,
)
from neuroecosistema.agents.predator import Predator
from neuroecosistema.agents.prey import Prey
from neuroecosistema.neural_network.network import NeuralNetwork


@dataclass
class Plant:
    """Representa una fuente de alimento dentro del ecosistema."""

    x: float
    y: float
    energy: float = 20.0
    radius: int = 6


class Environment:
    """Administra el mundo, sus recursos y sus organismos."""


    def __init__(
        self,
        width: int,
        height: int,
        initial_food: int,
        seed: int | None = None,
        food_energy: float = 20.0,
        food_respawn: int = 0,
        max_food: int | None = None,
    ) -> None:
        if initial_food < 0:
            raise ValueError(
                "La cantidad inicial de plantas no puede ser negativa."
            )
        if food_respawn < 0:
            raise ValueError(
                "El intervalo de reposición no puede ser negativo."
            )
        if max_food is None:
            max_food = initial_food
        if max_food < initial_food:
            raise ValueError(
                "El límite de plantas no puede ser menor que la cantidad inicial."
            )
        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.food_energy = food_energy
        self.food_respawn = food_respawn
        self.max_food = max_food
        self.step_count = 0
        self.plants: list[Plant] = []
        self.prey: list[Prey] = []
        self.predators: list[Predator] = []

        self.prey_networks = {}
        self.predator_networks = {}

        self.total_food_consumed = 0
        self.total_captures = 0

        self.generate_plants(initial_food)


    def spawn_plant(self) -> Plant:
        """Genera una planta utilizando la semilla del entorno."""
        margin = 20
        x = self.random.randint(
            margin,
            self.width - margin,
        )

        y = self.random.randint(
            margin,
            self.height - margin,
        )

        plant = Plant(
            x=x,
            y=y,
            energy=self.food_energy,
        )

        self.plants.append(plant)

        return plant
    
    def generate_plants(self, quantity: int) -> None:
        """Distribuye plantas aleatoriamente en el entorno."""

        if quantity < 0:
            raise ValueError(
                "La cantidad de plantas no puede ser negativa."
            )

        for _ in range(quantity):
            if len(self.plants) >= self.max_food:
                break

            self.spawn_plant()

    def generate_agents(
        self,
        prey_quantity: int,
        predator_quantity: int,
    ) -> None:
        """Genera las poblaciones iniciales de organismos."""

        margin = 30

        for _ in range(prey_quantity):
            x = self.random.randint(margin, self.width - margin)
            y = self.random.randint(margin, self.height - margin)

            self.prey.append(Prey(x=x, y=y))

        for _ in range(predator_quantity):
            x = self.random.randint(margin, self.width - margin)
            y = self.random.randint(margin, self.height - margin)

            self.predators.append(Predator(x=x, y=y))

    @staticmethod
    def nearest_plant(prey, plants):
        """Devuelve la planta más cercana a una presa."""

        if not plants:
            return None

        return min(
            plants,
            key=lambda plant: math.dist(
                (prey.x, prey.y),
                (plant.x, plant.y),
            ),
        )

    def update_agents(
    self,
    movement_cost=0.1,
    strategy="auto",
    prey_strategy=None,
    predator_strategy=None,
    ) -> None:
        """Actualiza los agentes según la estrategia seleccionada.

        auto:
            Utiliza redes si están asignadas; de lo contrario,
            movimiento aleatorio.

        random:
            Utiliza movimiento aleatorio.

        heuristic:
            Busca el alimento o la presa más cercana.

        neural:
            Utiliza las redes neuronales asignadas.
        """

        valid_strategies = {
            "auto",
            "random",
            "heuristic",
            "neural",
        }

        if strategy not in valid_strategies:
            raise ValueError(
                f"Estrategia no válida: {strategy}"
            )

        if prey_strategy is None:
            prey_strategy = strategy

        if predator_strategy is None:
            predator_strategy = strategy

        if prey_strategy not in valid_strategies:
            raise ValueError("Estrategia de presas no válida.")

        if predator_strategy not in valid_strategies:
            raise ValueError("Estrategia de depredadores no válida.")

        decisions = []

        for prey in self.prey:
            network = self.prey_networks.get(id(prey))

            if prey_strategy == "heuristic":
                direction = calculate_heuristic_direction(
                    prey,
                    self,
                )
            elif prey_strategy == "random":
                direction = None
            else:
                direction = (
                    calculate_neural_direction(prey, network, self)
                    if network is not None
                    else None
                )

            decisions.append((prey, direction))

        for predator in self.predators:
            network = self.predator_networks.get(id(predator))

            if predator_strategy == "heuristic":
                direction = calculate_heuristic_direction(
                    predator,
                    self,
                )
            elif predator_strategy == "random":
                direction = None
            else:
                direction = (
                    calculate_neural_direction(
                        predator,
                        network,
                        self,
                    )
                    if network is not None
                    else None
                )

            decisions.append((predator, direction))

        # Registrar objetivos antes del movimiento.
        food_targets = {}

        for prey in self.prey:
            target = self.nearest_plant(prey, self.plants)

            if target is None:
                continue

            distance_before = math.dist(
                (prey.x, prey.y),
                (target.x, target.y),
            )

            food_targets[id(prey)] = (
                target,
                distance_before,
            )

        # Calcular todas las decisiones antes de mover agentes.
        for agent, direction in decisions:
            agent.update(
                self.width,
                self.height,
                self.random,
                movement_cost,
                neural_direction=direction,
            )

        # Medir acercamiento a la misma planta seleccionada.
        for prey in self.prey:
            target_info = food_targets.get(id(prey))

            if target_info is None:
                prey.food_target_id = None
                prey.food_target_initial_distance = 0.0
                prey.food_target_min_distance = 0.0
                continue

            target, distance_before = target_info

            distance_after = math.dist(
                (prey.x, prey.y),
                (target.x, target.y),
            )

            prey.register_food_target_progress(
                target_id=id(target),
                distance=distance_before,
            )

            prey.register_food_target_progress(
                target_id=id(target),
                distance=distance_after,
            )

            progress = max(
                0.0,
                distance_before - distance_after,
            )

            prey.food_approach_progress += progress

        self.prey = [
            agent for agent in self.prey
            if agent.alive
        ]

        self.predators = [
            agent for agent in self.predators
            if agent.alive
        ]

        # Descartar redes de agentes eliminados.
        active_prey_ids = {
            id(agent) for agent in self.prey
        }

        active_predator_ids = {
            id(agent) for agent in self.predators
        }

        self.prey_networks = {
            agent_id: network
            for agent_id, network in self.prey_networks.items()
            if agent_id in active_prey_ids
        }

        self.predator_networks = {
            agent_id: network
            for agent_id, network in self.predator_networks.items()
            if agent_id in active_predator_ids
        }


    def update_food(self) -> None:
        """Repone una planta cada cierto número de pasos."""
        self.step_count += 1
        if self.food_respawn == 0:
            return
        if self.step_count % self.food_respawn != 0:
            return
        if len(self.plants) >= self.max_food:
            return

        self.spawn_plant()

    def resolve_interactions(self, capture_reward: float = 30.0) -> None:
        """Resuelve la alimentación y las capturas del ciclo."""

        # Alimentación de las presas.
        for prey in self.prey:
            for plant in self.plants[:]:
                distance = math.dist(
                    (prey.x, prey.y),
                    (plant.x, plant.y),
                )

                collision_distance = prey.radius + plant.radius

                if distance <= collision_distance:
                    prey.energy = min(100.0, prey.energy + plant.energy)
                    self.plants.remove(plant)
                    self.total_food_consumed += 1
                    prey.food_eaten += 1
                    break

        # Cacería de los depredadores.
        remaining_prey = self.prey[:]

        for predator in self.predators:
            for prey in remaining_prey:
                distance = math.dist(
                    (predator.x, predator.y),
                    (prey.x, prey.y),
                )

                collision_distance = predator.radius + prey.radius

                if distance <= collision_distance:
                    prey.alive = False
                    predator.energy = min(
                        100.0,
                        predator.energy + capture_reward,
                    )

                    remaining_prey.remove(prey)
                    self.total_captures += 1
                    predator.captures += 1
                    break

        self.prey = remaining_prey
        
        active_prey_ids = {id(agent) for agent in self.prey}

        self.prey_networks = {
            agent_id: network
            for agent_id, network in self.prey_networks.items()
            if agent_id in active_prey_ids
        }

    def initialize_neural_networks(self, network_config):
        """Asigna una red neuronal independiente a cada agente."""

        self.prey_networks = {}
        self.predator_networks = {}

        for prey in self.prey:
            network = NeuralNetwork(
                input_size=network_config["input_size"],
                hidden_layers=network_config["hidden_layers"],
                output_size=network_config["output_size"],
                hidden_activation=network_config["activation_hidden"],
                output_activation=network_config["activation_output"],
                seed=self.random.randint(0, 2**32 - 1),
            )

            self.prey_networks[id(prey)] = network

        for predator in self.predators:
            network = NeuralNetwork(
                input_size=network_config["input_size"],
                hidden_layers=network_config["hidden_layers"],
                output_size=network_config["output_size"],
                hidden_activation=network_config["activation_hidden"],
                output_activation=network_config["activation_output"],
                seed=self.random.randint(0, 2**32 - 1),
            )

            self.predator_networks[id(predator)] = network