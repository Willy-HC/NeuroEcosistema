
"""Prueba de equivalencia del controlador neuronal instrumentado."""

import math
import unittest

import neuroecosistema.simulation.environment as environment_module
from neuroecosistema.analysis.population_condition_probe import (
    PopulationCondition,
    _attach_networks,
    _create_environment,
    _create_initial_layout,
    _load_models,
)
from neuroecosistema.analysis.pursuit_trajectory_probe import (
    record_neural_decisions,
)
from neuroecosistema.train import load_configs
from unittest.mock import Mock

import neuroecosistema.agents.neural_controller as controller_module


class PursuitInstrumentationTests(unittest.TestCase):

    
    def test_controller_restored_after_exception(self):
        original_controller = (
            environment_module.calculate_neural_direction
        )
        original_wall_avoidance = (
            controller_module.apply_wall_avoidance
        )
        records = []
        with self.assertRaisesRegex(RuntimeError, "error de prueba"):
            with record_neural_decisions(records, 0, "5G"):
                self.assertIsNot(
                    environment_module.calculate_neural_direction,
                    original_controller,
                )
                raise RuntimeError("error de prueba")
        self.assertIs(
            environment_module.calculate_neural_direction,
            original_controller,
        )
        self.assertIs(
            controller_module.apply_wall_avoidance,
            original_wall_avoidance,
        )
        self.assertEqual(records, [])

    def test_full_simulation_equivalence(self):
        simulation_config, neural_config, _ = load_configs()

        models = _load_models(
            {
                "5G": {
                    "prey": "models/coevolution_seed/best_prey.npz",
                    "predator": "models/coevolution_seed/best_predator.npz",
                },
                "30G": {
                    "prey": "results/dagger_coevolution/best_prey.npz",
                    "predator": "results/dagger_coevolution/best_predator.npz",
                },
            },
            neural_config,
        )

        simulation = simulation_config["simulation"]

        layout = _create_initial_layout(
            42,
            simulation["width"],
            simulation["height"],
            30,
        )

        condition = PopulationCondition("test", 10, 2)

        def create_world(model_id):
            environment = _create_environment(
                simulation_config,
                simulation_config["energy"],
                condition,
                42,
                layout,
            )
            environment.prey_networks, _ = _attach_networks(
                environment.prey,
                models[model_id]["prey"],
            )
            environment.predator_networks, _ = _attach_networks(
                environment.predators,
                models[model_id]["predator"],
            )
            return environment

        
        def snapshot(environment):
            return {
                "prey": [
                    (
                        agent.x,
                        agent.y,
                        agent.energy,
                        agent.wall_escape_active,
                    )
                    for agent in environment.prey
                ],
                "predators": [
                    (
                        agent.x,
                        agent.y,
                        agent.energy,
                        agent.wall_escape_active,
                        agent.captures,
                    )
                    for agent in environment.predators
                ],
                "plants": [
                    (plant.x, plant.y, plant.energy)
                    for plant in environment.plants
                ],
                "total_captures": environment.total_captures,
                "rng_state": environment.random.getstate(),
            }
        
        def run(instrumented, model_id):
            environment = create_world(model_id)
            records = []
            states = []
            for step in range(200):
                if instrumented:
                    with record_neural_decisions(
                        records,
                        step,
                        model_id,
                    ):
                        environment.update_agents(
                            movement_cost=simulation_config[
                                "energy"
                            ]["movement_cost"],
                            strategy="neural",
                        )
                else:
                    environment.update_agents(
                        movement_cost=simulation_config[
                            "energy"
                        ]["movement_cost"],
                        strategy="neural",
                    )
                environment.resolve_interactions(
                    capture_reward=simulation_config[
                        "energy"
                    ]["capture_reward"]
                )
                environment.update_food()
                states.append(snapshot(environment))
                if not environment.prey:
                    break
            return states, records

        original_controller = (
            environment_module.calculate_neural_direction
        )

        for model_id in ("5G", "30G"):
            with self.subTest(model=model_id):
                baseline_states, _ = run(False, model_id)
                observed_states, records = run(True, model_id)

                self.assertEqual(baseline_states, observed_states)
                self.assertGreater(len(records), 0)
        self.assertIs(
            environment_module.calculate_neural_direction,
            original_controller,
        )


if __name__ == "__main__":
    unittest.main()
