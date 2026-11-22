import csv
import math
from pathlib import Path

import numpy as np
import pytest

from neuroecosistema.agents.perception import perceive_prey
from neuroecosistema.agents.prey import Prey
from neuroecosistema.analysis.food_orientation_probe import (
    INITIAL_ENERGY,
    MODEL_IDS,
    NULL_VECTOR_EPSILON,
    PLANT_DISTANCE,
    PREY_POSITION,
    WORLD_HEIGHT,
    WORLD_WIDTH,
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
from neuroecosistema.simulation.environment import Plant
from neuroecosistema.train import load_configs


def test_scenarios_cover_eight_directions_at_equal_distance():
    scenarios = build_scenarios()
    direction_scenarios = scenarios[:8]

    assert [scenario.scenario_id for scenario in direction_scenarios] == [
        "north",
        "northeast",
        "east",
        "southeast",
        "south",
        "southwest",
        "west",
        "northwest",
    ]
    assert scenarios[-1].scenario_id == "no_plant"
    assert scenarios[-1].plant_position is None

    distances = [
        math.dist(scenario.prey_position, scenario.plant_position)
        for scenario in direction_scenarios
    ]
    assert distances == pytest.approx(
        [PLANT_DISTANCE] * 8,
        abs=1e-10,
    )

    for scenario in direction_scenarios:
        assert scenario.prey_position == PREY_POSITION
        plant_x, plant_y = scenario.plant_position
        assert 40 < plant_x < WORLD_WIDTH - 40
        assert 40 < plant_y < WORLD_HEIGHT - 40


def test_scenario_perception_is_normalized_and_uses_controlled_energy():
    scenario = build_scenarios()[1]
    prey = Prey(
        x=scenario.prey_position[0],
        y=scenario.prey_position[1],
        energy=INITIAL_ENERGY,
    )
    plant = Plant(
        x=scenario.plant_position[0],
        y=scenario.plant_position[1],
    )

    inputs = perceive_prey(
        prey,
        [plant],
        [],
        WORLD_WIDTH,
        WORLD_HEIGHT,
    )
    dx = plant.x - prey.x
    dy = plant.y - prey.y

    np.testing.assert_allclose(
        inputs,
        [
            dx / WORLD_WIDTH,
            dy / WORLD_HEIGHT,
            PLANT_DISTANCE / math.hypot(WORLD_WIDTH, WORLD_HEIGHT),
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
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
    assert angular_error_degrees(direction, ideal) == pytest.approx(
        expected
    )


def test_angular_error_is_not_applicable_for_null_vectors():
    assert (
        angular_error_degrees(
            (NULL_VECTOR_EPSILON / 2, 0.0),
            (1.0, 0.0),
        )
        is None
    )
    assert angular_error_degrees((1.0, 0.0), None) is None


def test_circular_heading_span_handles_wraparound():
    assert circular_angle_span_degrees([350.0, 10.0]) == pytest.approx(
        20.0
    )


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


def test_practically_null_network_vector_is_recorded_without_angle():
    scenario = build_scenarios()[0]

    row = _measure_scenario(
        model_id="test",
        model_path=Path("test-model.npz"),
        model_hash="test-hash",
        network=create_constant_network(),
        scenario=scenario,
    )

    assert row["vector_is_practically_null"]
    assert row["output_magnitude"] == 0.0
    assert row["angular_error_degrees"] is None


def test_tanh_saturation_is_counted_at_documented_logit_threshold():
    scenario = build_scenarios()[0]

    row = _measure_scenario(
        model_id="test",
        model_path=Path("test-model.npz"),
        model_hash="test-hash",
        network=create_constant_network((3.0, -3.0)),
        scenario=scenario,
    )

    assert row["tanh_saturated"]
    assert row["tanh_saturated_components"] == 2


def test_observer_outputs_match_network_forward_for_both_models():
    _, neural_config, _ = load_configs()

    for model_path in default_model_paths().values():
        network = load_trained_network(model_path, neural_config)
        inputs = np.asarray(
            [0.1, -0.07, 0.08, 0.0, 0.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )

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


def test_probe_writes_nine_rows_per_model_and_preserves_models(
    tmp_path,
):
    model_paths = default_model_paths()
    hashes_before = {
        model_id: sha256_file(path)
        for model_id, path in model_paths.items()
    }

    result = run_probe(
        model_paths=model_paths,
        output_dir=tmp_path / "probe",
    )

    assert len(result["rows"]) == 18
    assert [row["model_id"] for row in result["rows"][:9]] == [
        "5G"
    ] * 9
    assert [row["model_id"] for row in result["rows"][9:]] == [
        "30G"
    ] * 9
    assert all(
        row["model_sha256"] == hashes_before[row["model_id"]]
        for row in result["rows"]
    )
    assert not any(
        row["wall_avoidance_active"] for row in result["rows"]
    )
    assert {
        model_id: sha256_file(path)
        for model_id, path in model_paths.items()
    } == hashes_before
    assert result["diagnostics"].is_file()
    assert result["summary"].is_file()
    assert result["diagnostics"].parent == tmp_path / "probe"
    assert result["summary"].parent == tmp_path / "probe"

    with result["diagnostics"].open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 18

    no_plant_rows = [
        row for row in rows if row["scenario_id"] == "no_plant"
    ]
    assert len(no_plant_rows) == 2
    assert all(
        row["angular_error_degrees"] == "N/A"
        for row in no_plant_rows
    )

    with result["summary"].open(
        newline="",
        encoding="utf-8",
    ) as file:
        summaries = list(csv.DictReader(file))
    assert len(summaries) == 2
    assert all(
        int(row["food_scenarios"]) == 8
        and int(row["angular_error_scenarios"]) == 8
        for row in summaries
    )


def test_probe_is_reproducible(tmp_path):
    model_paths = default_model_paths()
    first = run_probe(model_paths, tmp_path / "first")
    second = run_probe(model_paths, tmp_path / "second")

    assert first["diagnostics"].read_bytes() == (
        second["diagnostics"].read_bytes()
    )
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


