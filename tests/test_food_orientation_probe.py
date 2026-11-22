import csv
import math
from pathlib import Path

import numpy as np
import pytest

from neuroecosistema.agents.heuristic_controller import (
    calculate_heuristic_direction,
)
from neuroecosistema.agents.perception import perceive_prey
from neuroecosistema.analysis.food_orientation_probe import (
    FOOD_DISTANCES,
    INITIAL_ENERGY,
    MODEL_IDS,
    PREY_POSITIONS,
    SCENARIO_DIRECTIONS,
    TANH_SATURATION_LOGIT_THRESHOLD,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    _create_environment,
    _measure_scenario,
    angular_error_degrees,
    build_scenarios,
    circular_angle_span_degrees,
    default_model_paths,
    observe_network_outputs,
    run_probe,
    sha256_file,
)
from neuroecosistema.neural_network.model_loader import (
    load_trained_network,
)
from neuroecosistema.neural_network.network import NeuralNetwork
from neuroecosistema.train import load_configs


def food_scenario(start_position_id, distance, direction_id):
    return next(
        scenario
        for scenario in build_scenarios()
        if scenario.start_position_id == start_position_id
        and scenario.food_distance == distance
        and scenario.food_direction == direction_id
    )


def test_scenarios_cover_distances_directions_and_safe_positions():
    scenarios = build_scenarios()

    assert len(scenarios) == 3 * (3 * 8 + 1)
    for start_position_id, prey_position in PREY_POSITIONS:
        for distance in FOOD_DISTANCES:
            group = [
                scenario
                for scenario in scenarios
                if scenario.start_position_id == start_position_id
                and scenario.food_distance == distance
            ]
            assert [scenario.food_direction for scenario in group] == [
                direction for direction, _ in SCENARIO_DIRECTIONS
            ]
            assert [
                math.dist(scenario.prey_position, scenario.plant_position)
                for scenario in group
            ] == pytest.approx([distance] * 8, abs=1e-10)

            for scenario in group:
                plant_x, plant_y = scenario.plant_position
                assert 0.0 < plant_x < WORLD_WIDTH
                assert 0.0 < plant_y < WORLD_HEIGHT
                assert scenario.prey_position == prey_position

        no_food = [
            scenario
            for scenario in scenarios
            if scenario.start_position_id == start_position_id
            and scenario.food_distance is None
        ]
        assert len(no_food) == 1
        assert no_food[0].plant_position is None


def test_all_start_positions_have_same_wall_perception():
    wall_inputs = []
    for (
        start_position_id,
        _,
    ) in PREY_POSITIONS:
        scenario = food_scenario(start_position_id, 100.0, "east")
        environment, prey = _create_environment(scenario)
        inputs = perceive_prey(
            prey,
            environment.plants,
            environment.predators,
            WORLD_WIDTH,
            WORLD_HEIGHT,
        )
        wall_inputs.append(float(inputs[7]))

    assert wall_inputs == pytest.approx([300.0 / 350.0] * 3)


def test_perception_normalization_and_energy_are_controlled():
    scenario = food_scenario("center", 100.0, "northeast")
    environment, prey = _create_environment(scenario)
    inputs = perceive_prey(
        prey,
        environment.plants,
        environment.predators,
        WORLD_WIDTH,
        WORLD_HEIGHT,
    )
    plant = environment.plants[0]
    dx = plant.x - prey.x
    dy = plant.y - prey.y

    assert prey.energy == INITIAL_ENERGY
    np.testing.assert_allclose(
        inputs,
        [
            dx / WORLD_WIDTH,
            dy / WORLD_HEIGHT,
            100.0 / math.hypot(WORLD_WIDTH, WORLD_HEIGHT),
            0.0,
            0.0,
            1.0,
            1.0,
            300.0 / 350.0,
        ],
        rtol=0.0,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    ("direction", "ideal", "expected"),
    [
        ((1.0, 0.0), (1.0, 0.0), 0.0),
        ((1.0, 0.0), (0.0, 1.0), 90.0),
        ((1.0, 0.0), (-1.0, 0.0), 180.0),
    ],
)
def test_angular_error(direction, ideal, expected):
    assert angular_error_degrees(direction, ideal) == pytest.approx(expected)


def test_null_vectors_and_missing_food_have_no_angular_error():
    zero_network = create_constant_network()
    scenario = food_scenario("center", 50.0, "north")
    row = _measure_scenario(
        model_id="test",
        model_path=Path("test-model.npz"),
        model_hash="test-hash",
        network=zero_network,
        scenario=scenario,
    )
    assert row["vector_is_practically_null"]
    assert row["angular_error_degrees"] is None
    assert angular_error_degrees((1.0, 0.0), None) is None

    no_food = next(
        scenario
        for scenario in build_scenarios()
        if scenario.start_position_id == "center" and scenario.food_distance is None
    )
    no_food_row = _measure_scenario(
        model_id="test",
        model_path=Path("test-model.npz"),
        model_hash="test-hash",
        network=zero_network,
        scenario=no_food,
    )
    assert no_food_row["angular_error_degrees"] is None


def create_constant_network(output_biases=(0.0, 0.0)):
    network = NeuralNetwork(
        input_size=8,
        hidden_layers=[16, 12],
        output_size=2,
        hidden_activation="relu",
        output_activation="tanh",
        seed=42,
    )
    genome = np.zeros(network.count_parameters(), dtype=np.float64)
    genome[-2:] = output_biases
    network.set_genome(genome)
    return network


def test_tanh_saturation_uses_documented_logit_threshold():
    scenario = food_scenario("center", 50.0, "north")
    row = _measure_scenario(
        model_id="test",
        model_path=Path("test-model.npz"),
        model_hash="test-hash",
        network=create_constant_network((TANH_SATURATION_LOGIT_THRESHOLD, -3.1)),
        scenario=scenario,
    )

    assert row["tanh_saturated"]
    assert row["tanh_saturated_components"] == 2


def test_circular_heading_span_handles_wraparound():
    assert circular_angle_span_degrees([350.0, 10.0]) == pytest.approx(20.0)


def test_observer_outputs_match_network_forward_for_both_models():
    _, neural_config, _ = load_configs()
    inputs = np.asarray(
        [0.1, -0.07, 0.08, 0.0, 0.0, 1.0, 1.0, 300 / 350],
        dtype=np.float32,
    )

    for model_path in default_model_paths().values():
        network = load_trained_network(model_path, neural_config)
        logits, observed_outputs = observe_network_outputs(
            network,
            inputs,
        )
        assert logits.shape == (2,)
        np.testing.assert_allclose(
            observed_outputs,
            network.forward(inputs),
            rtol=1e-7,
            atol=1e-8,
        )


def test_observer_rejects_wrong_input_shape():
    _, neural_config, _ = load_configs()
    network = load_trained_network(
        default_model_paths()["5G"],
        neural_config,
    )

    with pytest.raises(ValueError):
        observe_network_outputs(network, np.zeros(7))


def test_heuristic_is_a_positive_control_across_all_food_scenarios():
    for scenario in build_scenarios():
        if scenario.plant_position is None:
            continue

        environment, prey = _create_environment(scenario)
        direction = calculate_heuristic_direction(prey, environment)
        ideal = (
            scenario.plant_position[0] - prey.x,
            scenario.plant_position[1] - prey.y,
        )

        assert direction is not None
        assert angular_error_degrees(direction, ideal) == pytest.approx(
            0.0,
            abs=1e-6,
        )


def test_probe_outputs_all_policies_and_preserves_models(tmp_path):
    model_paths = default_model_paths()
    hashes_before = {
        model_id: sha256_file(path) for model_id, path in model_paths.items()
    }

    result = run_probe(model_paths, tmp_path / "probe")
    rows_by_policy = {
        policy_id: [row for row in result["rows"] if row["model_id"] == policy_id]
        for policy_id in (*MODEL_IDS, "heuristic")
    }

    assert len(result["rows"]) == 225
    assert all(len(rows) == 75 for rows in rows_by_policy.values())
    assert all(
        row["model_sha256"] == hashes_before[row["model_id"]]
        for model_id in MODEL_IDS
        for row in rows_by_policy[model_id]
    )
    assert all(not row["wall_avoidance_active"] for row in result["rows"])
    assert {
        model_id: sha256_file(path) for model_id, path in model_paths.items()
    } == hashes_before

    heuristic_rows = [
        row for row in rows_by_policy["heuristic"] if row["food_distance"] is not None
    ]
    assert len(heuristic_rows) == 72
    assert all(
        row["angular_error_degrees"] == pytest.approx(0.0, abs=1e-6)
        for row in heuristic_rows
    )

    for model_id in MODEL_IDS:
        for distance in FOOD_DISTANCES:
            for direction, _ in SCENARIO_DIRECTIONS:
                translated_rows = [
                    row
                    for row in rows_by_policy[model_id]
                    if row["food_distance"] == distance
                    and row["food_direction"] == direction
                ]
                assert len(translated_rows) == 3
                np.testing.assert_allclose(
                    [
                        [
                            row["input_plant_dx"],
                            row["input_plant_dy"],
                            row["input_plant_distance"],
                            row["input_wall_distance"],
                            row["effective_direction_x"],
                            row["effective_direction_y"],
                        ]
                        for row in translated_rows
                    ],
                    [
                        [
                            translated_rows[0]["input_plant_dx"],
                            translated_rows[0]["input_plant_dy"],
                            translated_rows[0]["input_plant_distance"],
                            translated_rows[0]["input_wall_distance"],
                            translated_rows[0]["effective_direction_x"],
                            translated_rows[0]["effective_direction_y"],
                        ]
                    ]
                    * 3,
                    rtol=0.0,
                    atol=1e-7,
                )

    assert result["diagnostics"].is_file()
    assert result["summary"].is_file()
    assert result["diagnostics"].parent == tmp_path / "probe"
    assert result["summary"].parent == tmp_path / "probe"

    with result["diagnostics"].open(
        newline="",
        encoding="utf-8",
    ) as file:
        csv_rows = list(csv.DictReader(file))
    no_food_rows = [row for row in csv_rows if row["food_distance"] == ""]
    assert len(no_food_rows) == 9
    assert all(row["angular_error_degrees"] == "N/A" for row in no_food_rows)

    with result["summary"].open(
        newline="",
        encoding="utf-8",
    ) as file:
        summaries = list(csv.DictReader(file))
    assert len(summaries) == 30
    assert all(
        len(
            [
                row
                for row in summaries
                if row["model_id"] == model_id
                and row["group_type"] == "distance"
                and row["scenario_count"] == "24"
            ]
        )
        == 3
        for model_id in (*MODEL_IDS, "heuristic")
    )
    for model_id in MODEL_IDS:
        overall = next(
            row
            for row in summaries
            if row["model_id"] == model_id and row["group_type"] == "overall"
        )
        assert float(overall["food_output_heading_span_degrees"]) > 0.0
    for axis in ("horizontal", "vertical"):
        assert all(
            any(
                row["model_id"] == model_id
                and row["group_type"] == "axis"
                and row["group_value"] == axis
                and row["scenario_count"] == "18"
                for row in summaries
            )
            for model_id in (*MODEL_IDS, "heuristic")
        )


def test_probe_is_reproducible(tmp_path):
    model_paths = default_model_paths()
    first = run_probe(model_paths, tmp_path / "first")
    second = run_probe(model_paths, tmp_path / "second")

    assert first["diagnostics"].read_bytes() == (second["diagnostics"].read_bytes())
    assert first["summary"].read_bytes() == second["summary"].read_bytes()


def test_probe_writes_only_into_requested_output_directory(
    tmp_path,
    monkeypatch,
):
    written_paths = []
    original_open = Path.open

    def tracked_open(path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            written_paths.append(path.resolve())
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    output_dir = tmp_path / "contained-output"

    result = run_probe(default_model_paths(), output_dir)

    assert set(written_paths) == {
        result["diagnostics"].resolve(),
        result["summary"].resolve(),
    }
    assert all(path.parent == output_dir.resolve() for path in written_paths)


def test_both_repository_models_load_with_expected_architecture():
    _, neural_config, _ = load_configs()

    assert set(default_model_paths()) == set(MODEL_IDS)
    for model_path in default_model_paths().values():
        network = load_trained_network(model_path, neural_config)
        assert (
            network.input_size,
            *network.hidden_layers,
            network.output_size,
        ) == (8, 16, 12, 2)
        assert network.count_parameters() == 374
        assert all(
            np.all(np.isfinite(values))
            for values in [*network.weights, *network.biases]
        )
