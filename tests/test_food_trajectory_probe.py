import csv
import json
import math
from copy import deepcopy

import pytest

from neuroecosistema.agents.prey import Prey
from neuroecosistema.analysis.food_trajectory_probe import (
    PlantRegistry,
    _trace_episode,
    calculate_angular_error,
    classify_distance_change,
    classify_target_change,
    default_model_paths,
    match_consumed_plants,
    run_food_trajectory_probe,
)
from neuroecosistema.analysis.population_condition_probe import (
    CONDITIONS,
    _attach_networks,
    _create_environment,
    _create_initial_layout,
    _load_models,
)
from neuroecosistema.simulation.environment import Plant
from neuroecosistema.train import load_configs


def _loaded_models():
    _, neural_config, _ = load_configs()
    return _load_models(default_model_paths(), neural_config)


def _no_predator_condition():
    return next(
        condition
        for condition in CONDITIONS
        if condition.prey_count == 10 and condition.predator_count == 0
    )


def _environment_snapshot(environment):
    return {
        "prey": [
            (
                agent.x,
                agent.y,
                agent.energy,
                agent.direction,
                agent.direction_timer,
                agent.steps_survived,
                agent.food_eaten,
                agent.stagnation_steps,
                agent.distance_traveled,
                agent.steps_near_wall,
                agent.food_approach_progress,
                agent.food_net_progress,
                agent.food_target_switches,
                agent.food_target_initial_distance,
                agent.food_target_min_distance,
                agent.wall_escape_active,
                agent.visited_regions.copy(),
                agent.alive,
            )
            for agent in environment.prey
        ],
        "plants": [
            (plant.x, plant.y, plant.energy, plant.radius)
            for plant in environment.plants
        ],
        "rng_state": environment.random.getstate(),
        "step_count": environment.step_count,
        "total_food_consumed": environment.total_food_consumed,
        "total_captures": environment.total_captures,
    }


def test_angular_error_and_distance_change_handle_boundaries():
    assert calculate_angular_error((1, 0), (1, 0)) == pytest.approx(0)
    assert calculate_angular_error((1, 0), (0, 1)) == pytest.approx(90)
    assert calculate_angular_error((1, 0), (-1, 0)) == pytest.approx(180)
    assert calculate_angular_error((0, 0), (1, 0)) is None
    assert calculate_angular_error((1, 0), (0, 0)) is None

    assert classify_distance_change(10, 9.5) == "approaching"
    assert classify_distance_change(10, 10.5) == "receding"
    assert classify_distance_change(10, 10) == "unchanged"
    assert classify_distance_change(None, None) == "not_applicable"


def test_target_change_reasons_distinguish_consumption_and_reselection():
    active = {1, 2}
    removed = {1: "consumed_by_other"}

    assert (
        classify_target_change(None, 1, 1, removed, active)
        == "initial_acquisition"
    )
    assert (
        classify_target_change(1, 2, 2, removed, active)
        == "consumed_by_other"
    )
    assert (
        classify_target_change(2, 1, 2, removed, active)
        == "nearest_changed"
    )
    assert (
        classify_target_change(2, None, 2, removed, active)
        == "target_lost"
    )
    assert classify_target_change(None, None, 2, removed, active) == "no_target"


def test_consumption_is_associated_with_the_removed_collision_plant():
    prey = Prey(x=20, y=20)
    plant = Plant(x=25, y=20)
    registry = PlantRegistry()
    registry.ensure([plant])
    prey.food_eaten = 1

    eaten, consumers = match_consumed_plants(
        prey_agents=[prey],
        plants_before=[plant],
        plants_after=[],
        food_eaten_before={id(prey): 0},
        agent_indices={id(prey): 4},
        plant_registry=registry,
    )

    assert eaten == {id(prey): registry.plant_id(plant)}
    assert consumers == {registry.plant_id(plant): 4}


def test_trace_records_actual_nearest_target_and_all_network_inputs():
    simulation, _, _ = load_configs()
    models = _loaded_models()
    layout = _create_initial_layout(
        seed=1001,
        width=simulation["simulation"]["width"],
        height=simulation["simulation"]["height"],
        food_count=simulation["environment"]["initial_food"],
    )
    rows, summary, _ = _trace_episode(
        condition=_no_predator_condition(),
        seed=1001,
        model_id="5G",
        model=models["5G"]["prey"],
        simulation_config=simulation,
        max_steps=1,
        layout=layout,
    )

    assert len(rows) == 10
    assert summary["steps_executed"] == 1
    row = rows[0]
    prey_position = layout["prey_positions"][0]
    nearest_index, nearest_position = min(
        enumerate(layout["plants"], start=1),
        key=lambda item: math.dist(prey_position, item[1]),
    )
    assert row["target_plant_id"] == nearest_index
    assert row["target_plant_x"] == nearest_position[0]
    assert row["target_plant_y"] == nearest_position[1]
    assert len(
        [
            key
            for key in row
            if key.startswith("perception_")
        ]
    ) == 8
    assert math.isfinite(row["network_output_0"])
    assert math.isfinite(row["network_output_1"])
    assert row["target_distance_after_move"] == pytest.approx(
        math.dist(
            (row["x_after_move"], row["y_after_move"]),
            nearest_position,
        )
    )
    assert row["angular_error_deg"] is not None


def test_no_plant_steps_explicitly_mark_direction_metrics_not_applicable():
    simulation, _, _ = load_configs()
    simulation = deepcopy(simulation)
    simulation["environment"]["initial_food"] = 0
    simulation["environment"]["food_respawn"] = 0
    models = _loaded_models()
    layout = _create_initial_layout(
        seed=1001,
        width=simulation["simulation"]["width"],
        height=simulation["simulation"]["height"],
        food_count=0,
    )
    rows, summary, _ = _trace_episode(
        condition=_no_predator_condition(),
        seed=1001,
        model_id="30G",
        model=models["30G"]["prey"],
        simulation_config=simulation,
        max_steps=1,
        layout=layout,
    )

    assert all(row["target_plant_id"] is None for row in rows)
    assert all(row["angular_error_deg"] is None for row in rows)
    assert {
        row["angular_error_na_reason"] for row in rows
    } == {"no_plant"}
    assert {
        row["distance_change_na_reason"] for row in rows
    } == {"no_plant"}
    assert summary["angular_error_steps"] == 0
    assert summary["approaching_percent"] is None
    assert summary["distance_per_plant_na_reason"] == "no_plant_consumed"


def test_instrumentation_preserves_the_uninstrumented_environment_state():
    simulation, neural_config, _ = load_configs()
    models = _load_models(default_model_paths(), neural_config)
    layout = _create_initial_layout(
        seed=1002,
        width=simulation["simulation"]["width"],
        height=simulation["simulation"]["height"],
        food_count=simulation["environment"]["initial_food"],
    )
    condition = _no_predator_condition()
    model = models["30G"]["prey"]

    _, _, traced_environment = _trace_episode(
        condition=condition,
        seed=1002,
        model_id="30G",
        model=model,
        simulation_config=simulation,
        max_steps=4,
        layout=layout,
    )

    baseline_environment = _create_environment(
        simulation,
        simulation["energy"],
        condition,
        1002,
        layout,
    )
    networks, _ = _attach_networks(baseline_environment.prey, model)
    baseline_environment.prey_networks = networks
    for _ in range(4):
        baseline_environment.update_agents(
            movement_cost=simulation["energy"]["movement_cost"],
            prey_strategy="neural",
            predator_strategy="neural",
        )
        baseline_environment.resolve_interactions(
            capture_reward=simulation["energy"]["capture_reward"],
        )
        baseline_environment.update_food()

    assert _environment_snapshot(traced_environment) == _environment_snapshot(
        baseline_environment
    )


def test_probe_is_reproducible_and_limits_writes_to_output_directories(
    tmp_path,
):
    first = run_food_trajectory_probe(
        tmp_path / "first",
        seeds=(1003,),
        max_steps=2,
    )
    second = run_food_trajectory_probe(
        tmp_path / "second",
        seeds=(1003,),
        max_steps=2,
    )

    assert first["episode_count"] == 2
    assert first["trajectory_rows"] == 40
    assert first["summary_rows"] == 2
    for key in ("trajectory", "summary", "manifest"):
        assert first[key].read_bytes() == second[key].read_bytes()

    with first["trajectory"].open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 40
    manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
    assert manifest["seeds"] == [1003]
    assert manifest["condition"] == {
        "condition_id": "prey10_pred0",
        "prey_agents": 10,
        "predator_agents": 0,
    }
    assert manifest["comparability"]["additional_random_calls"] == 0
    assert {path.name for path in (tmp_path / "first").iterdir()} == {
        "food_trajectory.csv",
        "episode_summary.csv",
        "manifest.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == {"first", "second"}
