from neuroecosistema.main import create_environment

CONFIG = {
    "simulation": {"width": 500, "height": 400, "max_steps": 300},
    "population": {"prey": 2, "predators": 1},
    "environment": {"initial_food": 4},
    "energy": {
        "food_reward": 20,
        "movement_cost": 0.1,
        "capture_reward": 30,
    },
}
NEURAL_CONFIG = {
    "input_size": 8,
    "hidden_layers": [16, 12],
    "output_size": 2,
    "activation_hidden": "relu",
    "activation_output": "tanh",
}
SETTINGS = {
    "prey": 2,
    "predators": 1,
    "initial_food": 4,
    "food_respawn": 5,
    "max_food": 8,
    "seed": 73,
    "predator_max_turn_degrees": 45,
}


def create_run():
    return create_environment(
        CONFIG,
        NEURAL_CONFIG,
        SETTINGS,
        world_height=340,
    )


def snapshot(environment):
    return {
        "plants": [(plant.x, plant.y) for plant in environment.plants],
        "prey": [(agent.x, agent.y) for agent in environment.prey],
        "predators": [
            (agent.x, agent.y) for agent in environment.predators
        ],
        "prey_genomes": [
            environment.prey_networks[id(agent)].get_genome().tolist()
            for agent in environment.prey
        ],
        "predator_genomes": [
            environment.predator_networks[id(agent)].get_genome().tolist()
            for agent in environment.predators
        ],
    }


def test_repeated_runs_are_clean_and_reproducible():
    first = create_run()
    expected = snapshot(first)
    first.update_agents(movement_cost=0.0)
    first.update_food()
    first.total_captures = 5
    first.total_food_consumed = 7
    first.plants.clear()

    second = create_run()
    third = create_run()

    assert snapshot(second) == expected
    assert snapshot(third) == expected
    for restarted in (second, third):
        assert restarted.step_count == 0
        assert restarted.total_captures == 0
        assert restarted.total_food_consumed == 0
        assert restarted.predator_previous_directions == {}
        assert len(restarted.prey_networks) == len(restarted.prey) == 2
        assert len(restarted.predator_networks) == len(restarted.predators) == 1
        assert not set(restarted.prey_networks).intersection(
            id(agent) for agent in first.prey
        )
