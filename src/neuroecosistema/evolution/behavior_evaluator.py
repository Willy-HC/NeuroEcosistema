
from pathlib import Path

import numpy as np

from neuroecosistema.analysis.food_orientation_observer import (
    FoodOrientationObserver,
)
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
        observe_food_orientation=False,
        observe_tactical_progress=False,
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
                if not agents:
                    continue

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
        prey_by_id = {id(prey): prey for prey in original_prey}
        predator_by_id = {
            id(predator): predator
            for predator in original_predators
        }
        evasion_distance = {
            id(prey): 0.0 for prey in original_prey
        }
        threat_steps = {
            id(prey): 0 for prey in original_prey
        }
        pursuit_distance = {
            id(predator): 0.0
            for predator in original_predators
        }
        pursuit_steps = {
            id(predator): 0
            for predator in original_predators
        }
        threat_radius = 180.0

        observer = (
            FoodOrientationObserver(environment)
            if observe_food_orientation
            else None
        )

        for _ in range(simulation["max_steps"]):
            if not environment.prey and not environment.predators:
                break

            prey_threats = {}
            predator_targets = {}
            if observe_tactical_progress:
                for prey in environment.prey:
                    if not environment.predators:
                        continue
                    threat = min(
                        environment.predators,
                        key=lambda predator: np.hypot(
                            prey.x - predator.x,
                            prey.y - predator.y,
                        ),
                    )
                    distance = float(
                        np.hypot(prey.x - threat.x, prey.y - threat.y)
                    )
                    if distance <= threat_radius:
                        prey_threats[id(prey)] = (
                            threat.x,
                            threat.y,
                            distance,
                        )

                for predator in environment.predators:
                    if not environment.prey:
                        continue
                    target = min(
                        environment.prey,
                        key=lambda prey: np.hypot(
                            predator.x - prey.x,
                            predator.y - prey.y,
                        ),
                    )
                    distance = float(
                        np.hypot(
                            predator.x - target.x,
                            predator.y - target.y,
                        )
                    )
                    predator_targets[id(predator)] = (
                        target.x,
                        target.y,
                        distance,
                    )

            environment.update_agents(
                movement_cost=energy["movement_cost"],
                prey_strategy=prey_strategy,
                predator_strategy=predator_strategy,
            )

            for prey_id, (target_x, target_y, before) in (
                prey_threats.items()
            ):
                prey = prey_by_id[prey_id]
                after = float(
                    np.hypot(prey.x - target_x, prey.y - target_y)
                )
                evasion_distance[prey_id] += max(
                    0.0,
                    after - before,
                )
                threat_steps[prey_id] += 1

            for predator_id, (target_x, target_y, before) in (
                predator_targets.items()
            ):
                predator = predator_by_id[predator_id]
                after = float(
                    np.hypot(
                        predator.x - target_x,
                        predator.y - target_y,
                    )
                )
                pursuit_distance[predator_id] += max(
                    0.0,
                    before - after,
                )
                pursuit_steps[predator_id] += 1

            if observer is not None:
                observer.after_movement(original_prey)

            environment.resolve_interactions(
                capture_reward=energy["capture_reward"]
            )

            if observer is not None:
                observer.after_interactions(original_prey)

            environment.update_food()
            steps_executed += 1

        prey_summary = self.summarize(
            original_prey,
            "prey",
            seed,
            steps_executed,
        )
        results = {"prey": prey_summary}
        if observe_tactical_progress:
            prey_summary.update({
                "mean_food_progress_score": float(
                    np.mean(
                        [prey.food_progress_score for prey in original_prey]
                    )
                ),
                "mean_evasion_distance": float(
                    np.mean(list(evasion_distance.values()))
                ),
                "mean_threat_steps": float(
                    np.mean(list(threat_steps.values()))
                ),
            })

        if original_predators:
            predator_summary = self.summarize(
                original_predators,
                "predator",
                seed,
                steps_executed,
            )
            if observe_tactical_progress:
                predator_summary.update({
                    "mean_pursuit_distance": float(
                        np.mean(list(pursuit_distance.values()))
                    ),
                    "mean_pursuit_steps": float(
                        np.mean(list(pursuit_steps.values()))
                    ),
                })
            results["predator"] = predator_summary

        if observer is not None:
            observer.finish()
            results["food_orientation"] = observer.results()

        return results

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
            "mean_food_approach_progress": average(
                [agent.food_approach_progress for agent in agents]
            ),
            "mean_food_net_progress": average(
                [agent.food_net_progress for agent in agents]
            ),
            "mean_food_target_switches": average(
                [agent.food_target_switches for agent in agents]
            ),
        }