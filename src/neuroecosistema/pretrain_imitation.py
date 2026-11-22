"""Generate heuristic imitation data and pretrain both species."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from neuroecosistema.agents.heuristic_controller import (
    calculate_heuristic_direction,
)
from neuroecosistema.agents.neural_controller import (
    apply_wall_avoidance,
    normalize_direction,
)
from neuroecosistema.agents.perception import (
    perceive_predator,
    perceive_prey,
)
from neuroecosistema.agents.predator import Predator
from neuroecosistema.agents.prey import Prey
from neuroecosistema.neural_network.network import NeuralNetwork
from neuroecosistema.simulation.environment import Plant
from neuroecosistema.train import load_configs

DEFAULT_SEED = 8121901
ESCAPE_RADIUS = 180.0
DISTANT_FOOD_THREAT_SCENARIO = "food_with_distant_threat"
SCENARIO_TYPES = (
    "objective_only",
    "threat_only",
    "objective_and_threat",
    "no_objective",
    "near_wall_objective",
    "near_wall_threat",
    "multiple_targets",
)
PREY_SCENARIO_TYPES = (
    *SCENARIO_TYPES,
    DISTANT_FOOD_THREAT_SCENARIO,
)
CURRICULUM_PHASES = {
    "goal_seeking": ("objective_only",),
    "safety_and_mixed": (
        "threat_only",
        "objective_and_threat",
        "no_objective",
        "near_wall_objective",
        "near_wall_threat",
        "multiple_targets",
    ),
}
PREY_CURRICULUM_PHASES = {
    "goal_seeking": CURRICULUM_PHASES["goal_seeking"],
    "safety_and_mixed": (
        "threat_only",
        "objective_and_threat",
        "no_objective",
        "near_wall_objective",
        "near_wall_threat",
        "multiple_targets",
        DISTANT_FOOD_THREAT_SCENARIO,
        DISTANT_FOOD_THREAT_SCENARIO,
        DISTANT_FOOD_THREAT_SCENARIO,
    ),
}
GOAL_REPLAY_FRACTION = 0.5
OUTPUT_NORM_TARGET = 0.8
OUTPUT_NORM_WEIGHT = 0.1
OUTPUT_NORM_EPSILON = 1e-8


def _unit(vector):
    return normalize_direction(float(vector[0]), float(vector[1]))


def imitation_teacher_direction(agent, species, environment):
    """Combine the existing goal heuristic with reactive safety behavior."""

    direction = calculate_heuristic_direction(agent, environment)

    if species == "prey":
        nearest_predator = min(
            environment.predators,
            key=lambda predator: math.dist(
                (agent.x, agent.y),
                (predator.x, predator.y),
            ),
            default=None,
        )
        if nearest_predator is not None:
            distance = math.dist(
                (agent.x, agent.y),
                (nearest_predator.x, nearest_predator.y),
            )
            threat_weight = max(
                0.0,
                min(1.0, 1.0 - distance / ESCAPE_RADIUS),
            )
            if threat_weight > 0.0:
                away = _unit((
                    agent.x - nearest_predator.x,
                    agent.y - nearest_predator.y,
                ))
                if away == (0.0, 0.0):
                    away = (-1.0, 0.0)
                toward_food = direction or (0.0, 0.0)
                direction = _unit((
                    (1.0 - threat_weight) * toward_food[0]
                    + threat_weight * away[0],
                    (1.0 - threat_weight) * toward_food[1]
                    + threat_weight * away[1],
                ))
                if direction == (0.0, 0.0):
                    direction = away

    effective_direction = apply_wall_avoidance(
        agent,
        direction[0] if direction is not None else 0.0,
        direction[1] if direction is not None else 0.0,
        environment.width,
        environment.height,
    )
    return _unit(effective_direction)


def _place_relative(rng, origin, distance_range, width, height, radius):
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    distance = float(rng.uniform(*distance_range))
    return (
        min(
            width - radius,
            max(radius, origin[0] + distance * math.cos(angle)),
        ),
        min(
            height - radius,
            max(radius, origin[1] + distance * math.sin(angle)),
        ),
    )


def _place_distant_threat(rng, origin, width, height, radius):
    for _ in range(32):
        x, y = _place_relative(
            rng,
            origin,
            (220.0, 400.0),
            width,
            height,
            radius,
        )
        if math.dist(origin, (x, y)) > ESCAPE_RADIUS:
            return x, y

    corners = (
        (radius, radius),
        (width - radius, radius),
        (radius, height - radius),
        (width - radius, height - radius),
    )
    return max(corners, key=lambda point: math.dist(origin, point))


def _scenario_anchor(rng, width, height, near_wall):
    margin = 30.0
    x = float(rng.uniform(margin, width - margin))
    y = float(rng.uniform(margin, height - margin))
    if near_wall:
        side = int(rng.integers(0, 4))
        wall_offset = float(rng.uniform(6.0, 75.0))
        if side == 0:
            x = wall_offset
        elif side == 1:
            x = width - wall_offset
        elif side == 2:
            y = wall_offset
        else:
            y = height - wall_offset
    return x, y


def _build_synthetic_state(species, scenario_type, rng, width, height):
    near_wall = scenario_type.startswith("near_wall")
    anchor = _scenario_anchor(rng, width, height, near_wall)
    if species == "prey":
        agent = Prey(
            x=anchor[0],
            y=anchor[1],
            energy=float(rng.uniform(5.0, 100.0)),
        )
        plants = []
        predators = []
        if scenario_type in {
            "objective_only",
            "objective_and_threat",
            "near_wall_objective",
            "multiple_targets",
            DISTANT_FOOD_THREAT_SCENARIO,
        }:
            count = 3 if scenario_type == "multiple_targets" else 1
            for index in range(count):
                distance_range = (
                    (12.0, 600.0)
                    if scenario_type == "objective_only"
                    else (12.0, 250.0)
                    if index == 0
                    else (80.0, 600.0)
                )
                x, y = _place_relative(
                    rng,
                    anchor,
                    distance_range,
                    width,
                    height,
                    radius=6,
                )
                plants.append(Plant(x=x, y=y))
        if scenario_type in {
            "threat_only",
            "objective_and_threat",
            "near_wall_threat",
            "multiple_targets",
            DISTANT_FOOD_THREAT_SCENARIO,
        }:
            count = 3 if scenario_type == "multiple_targets" else 1
            for _ in range(count):
                if scenario_type == DISTANT_FOOD_THREAT_SCENARIO:
                    x, y = _place_distant_threat(
                        rng,
                        anchor,
                        width,
                        height,
                        radius=6,
                    )
                else:
                    threat_range = (
                        (12.0, 90.0)
                        if scenario_type in {
                            "threat_only",
                            "near_wall_threat",
                        }
                        else (35.0, 300.0)
                    )
                    x, y = _place_relative(
                        rng,
                        anchor,
                        threat_range,
                        width,
                        height,
                        radius=6,
                    )
                predators.append(Predator(x=x, y=y))
        environment = SimpleNamespace(
            plants=plants,
            predators=predators,
            prey=[],
            width=width,
            height=height,
        )
        inputs = perceive_prey(
            agent,
            plants,
            predators,
            width,
            height,
        )
    elif species == "predator":
        agent = Predator(
            x=anchor[0],
            y=anchor[1],
            energy=float(rng.uniform(5.0, 100.0)),
        )
        prey_agents = []
        if scenario_type != "no_objective":
            count = (
                5
                if scenario_type == "multiple_targets"
                else int(rng.integers(1, 4))
            )
            for index in range(count):
                distance_range = (
                    (12.0, 35.0)
                    if scenario_type == "objective_and_threat"
                    and index == 0
                    else (35.0, 600.0)
                    if index == 0
                    else (80.0, 600.0)
                )
                x, y = _place_relative(
                    rng,
                    anchor,
                    distance_range,
                    width,
                    height,
                    radius=5,
                )
                prey_agents.append(Prey(x=x, y=y))
        environment = SimpleNamespace(
            plants=[],
            predators=[],
            prey=prey_agents,
            width=width,
            height=height,
        )
        inputs = perceive_predator(
            agent,
            prey_agents,
            width,
            height,
        )
    else:
        raise ValueError(f"Unknown species: {species}")

    direction = imitation_teacher_direction(
        agent,
        species,
        environment,
    )
    label = direction or (0.0, 0.0)
    return inputs, np.asarray(label, dtype=np.float32)


def generate_teacher_dataset(
    species,
    sample_count,
    seed,
    width=1000,
    height=700,
    scenario_types=SCENARIO_TYPES,
):
    """Generate an independent, balanced set of synthetic scenarios."""

    if species not in {"prey", "predator"}:
        raise ValueError("species must be 'prey' or 'predator'.")
    scenario_types = tuple(scenario_types)
    allowed_scenarios = (
        PREY_SCENARIO_TYPES if species == "prey" else SCENARIO_TYPES
    )
    if not scenario_types or any(
        scenario_type not in allowed_scenarios
        for scenario_type in scenario_types
    ):
        raise ValueError("scenario_types must contain known scenarios.")
    if sample_count < len(scenario_types):
        raise ValueError(
            f"sample_count must be at least {len(scenario_types)}."
        )
    if width <= 0 or height <= 0:
        raise ValueError("Map dimensions must be positive.")

    rng = np.random.default_rng(seed)
    features = np.empty((sample_count, 8), dtype=np.float32)
    labels = np.empty((sample_count, 2), dtype=np.float32)
    scenario_ids = []

    for index in range(sample_count):
        scenario_type = scenario_types[index % len(scenario_types)]
        features[index], labels[index] = _build_synthetic_state(
            species,
            scenario_type,
            rng,
            width,
            height,
        )
        scenario_ids.append(f"{seed}:{index}:{scenario_type}")

    if not np.all(np.isfinite(features)) or not np.all(
        np.isfinite(labels)
    ):
        raise ValueError("Synthetic data contains NaN or infinity.")

    return features, labels, scenario_ids


def _combine_datasets(*datasets):
    return (
        np.concatenate([dataset[0] for dataset in datasets]),
        np.concatenate([dataset[1] for dataset in datasets]),
        [
            scenario_id
            for dataset in datasets
            for scenario_id in dataset[2]
        ],
    )


def _minimum_training_samples(safety_scenario_count):
    sample_count = 1
    while True:
        goal_count = max(1, round(sample_count * 0.4))
        safety_count = sample_count - goal_count
        replay_count = max(1, round(safety_count * GOAL_REPLAY_FRACTION))
        if safety_count - replay_count >= safety_scenario_count:
            return sample_count
        sample_count += 1


def _forward_with_cache(network, inputs):
    activation = inputs.astype(np.float64, copy=False)
    activations = [activation]
    preactivations = []

    for index, (weights, biases) in enumerate(
        zip(network.weights, network.biases)
    ):
        preactivation = activation @ weights.T + biases
        preactivations.append(preactivation)
        if index == len(network.weights) - 1:
            activation = network.output_activation(preactivation)
        else:
            activation = network.hidden_activation(preactivation)
        activations.append(activation)

    return activation, activations, preactivations


def _activation_derivative(name, preactivation, activation):
    if name == "relu":
        return preactivation > 0.0
    if name == "tanh":
        return 1.0 - activation**2
    raise ValueError(f"Unsupported activation for training: {name}")


def _direction_loss_and_gradient(predictions, labels):
    label_norms = np.linalg.norm(labels, axis=1)
    valid = label_norms > OUTPUT_NORM_EPSILON
    gradient = np.zeros_like(predictions, dtype=np.float64)
    if not np.any(valid):
        return 0.0, gradient

    active_predictions = predictions[valid]
    active_labels = labels[valid] / label_norms[valid, None]
    squared_norms = np.sum(active_predictions**2, axis=1)
    norms = np.sqrt(squared_norms + OUTPUT_NORM_EPSILON)
    dot_products = np.sum(active_predictions * active_labels, axis=1)
    cosine = dot_products / norms
    norm_error = norms - OUTPUT_NORM_TARGET
    losses = (
        1.0
        - cosine
        + OUTPUT_NORM_WEIGHT * norm_error**2
    )

    active_gradient = (
        -active_labels / norms[:, None]
        + (
            dot_products / norms**3
        )[:, None] * active_predictions
        + (
            2.0
            * OUTPUT_NORM_WEIGHT
            * norm_error
            / norms
        )[:, None] * active_predictions
    ) / np.count_nonzero(valid)
    gradient[valid] = active_gradient
    return float(np.mean(losses)), gradient


def _loss(network, features, labels, l2):
    predictions = np.asarray(network.forward(features), dtype=np.float64)
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError("Network produced non-finite outputs.")
    labels = np.asarray(labels, dtype=np.float64)
    valid = np.linalg.norm(labels, axis=1) > OUTPUT_NORM_EPSILON
    if np.any(valid):
        mse = float(np.mean((predictions[valid] - labels[valid]) ** 2))
    else:
        mse = 0.0
    direction_loss, _ = _direction_loss_and_gradient(
        predictions,
        labels,
    )
    penalty = 0.5 * l2 * sum(
        float(np.sum(weights**2))
        for weights in network.weights
    )
    return direction_loss + penalty, mse


def _train_batch(
    network,
    features,
    labels,
    first_moments,
    second_moments,
    step,
    learning_rate,
    l2,
):
    predictions, activations, preactivations = _forward_with_cache(
        network,
        features,
    )
    _, gradient = _direction_loss_and_gradient(predictions, labels)
    if not np.any(np.linalg.norm(labels, axis=1) > OUTPUT_NORM_EPSILON):
        return

    gradients = []
    for layer_index in reversed(range(len(network.weights))):
        activation_name = (
            network.output_activation_name
            if layer_index == len(network.weights) - 1
            else network.hidden_activation_name
        )
        gradient = gradient * _activation_derivative(
            activation_name,
            preactivations[layer_index],
            activations[layer_index + 1],
        )

        weight_gradient = gradient.T @ activations[layer_index]
        weight_gradient += l2 * network.weights[layer_index]
        bias_gradient = np.sum(gradient, axis=0)
        gradients.append((weight_gradient, bias_gradient))

        if layer_index > 0:
            gradient = gradient @ network.weights[layer_index]

    gradients.reverse()
    beta_one = 0.9
    beta_two = 0.999
    epsilon = 1e-8

    for index, (weight_gradient, bias_gradient) in enumerate(gradients):
        if not (
            np.all(np.isfinite(weight_gradient))
            and np.all(np.isfinite(bias_gradient))
        ):
            raise FloatingPointError("Non-finite training gradient.")

        weight_first, bias_first = first_moments[index]
        weight_second, bias_second = second_moments[index]
        weight_first *= beta_one
        weight_first += (1.0 - beta_one) * weight_gradient
        bias_first *= beta_one
        bias_first += (1.0 - beta_one) * bias_gradient
        weight_second *= beta_two
        weight_second += (1.0 - beta_two) * weight_gradient**2
        bias_second *= beta_two
        bias_second += (1.0 - beta_two) * bias_gradient**2

        corrected_weight_first = weight_first / (1.0 - beta_one**step)
        corrected_bias_first = bias_first / (1.0 - beta_one**step)
        corrected_weight_second = weight_second / (
            1.0 - beta_two**step
        )
        corrected_bias_second = bias_second / (
            1.0 - beta_two**step
        )

        network.weights[index] -= (
            learning_rate
            * corrected_weight_first
            / (np.sqrt(corrected_weight_second) + epsilon)
        )
        network.biases[index] -= (
            learning_rate
            * corrected_bias_first
            / (np.sqrt(corrected_bias_second) + epsilon)
        )


def train_supervised_network(
    network,
    train_features,
    train_labels,
    validation_features,
    validation_labels,
    *,
    seed,
    max_epochs=100,
    batch_size=256,
    patience=12,
    learning_rate=0.001,
    l2=0.0001,
):
    """Fit by Adam and return the checkpoint with lowest validation loss."""

    if max_epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive.")
    if (
        not math.isfinite(learning_rate)
        or not math.isfinite(l2)
        or learning_rate <= 0
        or l2 < 0
    ):
        raise ValueError("learning_rate must be positive and L2 nonnegative.")
    if (
        train_features.ndim != 2
        or validation_features.ndim != 2
        or train_features.shape[1] != network.input_size
        or validation_features.shape[1] != network.input_size
        or train_labels.shape != (len(train_features), network.output_size)
        or validation_labels.shape
        != (len(validation_features), network.output_size)
    ):
        raise ValueError("Training data shapes do not match the network.")
    if network.output_activation_name != "tanh":
        raise ValueError(
            "Direction learning requires a signed tanh output activation."
        )
    if (
        not np.all(np.isfinite(train_features))
        or not np.all(np.isfinite(train_labels))
        or not np.all(np.isfinite(validation_features))
        or not np.all(np.isfinite(validation_labels))
        or not np.any(
            np.linalg.norm(validation_labels, axis=1)
            > OUTPUT_NORM_EPSILON
        )
        or not np.any(
            np.linalg.norm(train_labels, axis=1) > OUTPUT_NORM_EPSILON
        )
    ):
        raise ValueError(
            "Training data must be finite and both splits must contain "
            "labeled directions."
        )

    first_moments = [
        (np.zeros_like(weights), np.zeros_like(biases))
        for weights, biases in zip(network.weights, network.biases)
    ]
    second_moments = [
        (np.zeros_like(weights), np.zeros_like(biases))
        for weights, biases in zip(network.weights, network.biases)
    ]
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_genome = network.get_genome().copy()
    best_epoch = 0
    epochs_without_improvement = 0
    step = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(train_features))
        for start in range(0, len(order), batch_size):
            batch_indices = order[start:start + batch_size]
            batch_labels = train_labels[batch_indices]
            if not np.any(
                np.linalg.norm(batch_labels, axis=1)
                > OUTPUT_NORM_EPSILON
            ):
                continue
            step += 1
            _train_batch(
                network,
                train_features[batch_indices],
                batch_labels,
                first_moments,
                second_moments,
                step,
                learning_rate,
                l2,
            )

        train_loss, train_mse = _loss(
            network,
            train_features,
            train_labels,
            l2,
        )
        validation_loss, validation_mse = _loss(
            network,
            validation_features,
            validation_labels,
            l2,
        )
        if not all(
            math.isfinite(value)
            for value in (train_loss, train_mse, validation_loss)
        ):
            raise FloatingPointError("Non-finite training loss.")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mse": train_mse,
            "validation_loss": validation_loss,
            "validation_mse": validation_mse,
        })

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_genome = network.get_genome().copy()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    network.set_genome(best_genome)
    return network, history, best_epoch


def _direction_metrics(network, features, labels, l2):
    loss, mse = _loss(network, features, labels, l2)
    predicted = np.asarray(network.forward(features), dtype=np.float64)
    predicted_norm = np.linalg.norm(predicted, axis=1)
    labels_norm = np.linalg.norm(labels, axis=1)
    labeled = labels_norm > OUTPUT_NORM_EPSILON
    predicted_valid = labeled & (predicted_norm > 1e-8)
    angular_error = np.full(
        np.count_nonzero(labeled),
        180.0,
        dtype=np.float64,
    )
    if np.any(predicted_valid):
        cosine = np.sum(
            predicted[predicted_valid] * labels[predicted_valid],
            axis=1,
        )
        cosine /= (
            predicted_norm[predicted_valid]
            * labels_norm[predicted_valid]
        )
        angular_error[predicted_norm[labeled] > 1e-8] = (
            np.degrees(
                np.arccos(np.clip(cosine, -1.0, 1.0))
            )
        )
    predicted_directions = predicted[predicted_valid] / (
        predicted_norm[predicted_valid, None]
    )
    label_directions = labels[labeled] / labels_norm[labeled, None]
    active_output_norms = predicted_norm[labeled]
    return {
        "loss": float(loss),
        "mse": float(mse),
        "mean_angular_error_degrees": (
            float(np.mean(angular_error))
            if angular_error.size
            else None
        ),
        "evaluated_directions": int(np.count_nonzero(labeled)),
        "unlabeled_samples": int(len(labels) - np.count_nonzero(labeled)),
        "low_output_norm_fraction": (
            float(np.mean(predicted_norm[labeled] <= 1e-8))
            if np.any(labeled)
            else 0.0
        ),
        "mean_output_norm": (
            float(np.mean(active_output_norms))
            if active_output_norms.size
            else 0.0
        ),
        "output_saturation_fraction": (
            float(np.mean(np.abs(predicted[labeled]) >= 0.995))
            if np.any(labeled)
            else 0.0
        ),
        "predicted_direction_resultant": (
            float(np.linalg.norm(np.mean(predicted_directions, axis=0)))
            if predicted_directions.size
            else 0.0
        ),
        "label_direction_resultant": (
            float(np.linalg.norm(np.mean(label_directions, axis=0)))
            if label_directions.size
            else 0.0
        ),
    }


def _network_from_seed(neural_config, seed):
    return NeuralNetwork(
        input_size=neural_config["input_size"],
        hidden_layers=neural_config["hidden_layers"],
        output_size=neural_config["output_size"],
        hidden_activation=neural_config["activation_hidden"],
        output_activation=neural_config["activation_output"],
        seed=seed,
    )


def pretrain_species(
    species,
    output_dir,
    neural_config,
    map_width,
    map_height,
    *,
    seed,
    train_samples,
    validation_samples,
    test_samples,
    max_epochs,
    batch_size,
    patience,
    learning_rate,
    l2,
):
    curriculum_phases = (
        PREY_CURRICULUM_PHASES
        if species == "prey"
        else CURRICULUM_PHASES
    )
    scenario_types = (
        PREY_SCENARIO_TYPES if species == "prey" else SCENARIO_TYPES
    )
    safety_scenarios = tuple(
        dict.fromkeys(curriculum_phases["safety_and_mixed"])
    )
    minimum_train_samples = _minimum_training_samples(
        len(curriculum_phases["safety_and_mixed"])
    )
    minimum_validation_samples = 2 * len(safety_scenarios)

    if max_epochs < 2:
        raise ValueError("Curriculum training requires at least two epochs.")
    if train_samples < minimum_train_samples:
        raise ValueError(
            "train_samples must be at least "
            f"{minimum_train_samples} for this species' curriculum."
        )
    if (
        validation_samples < minimum_validation_samples
        or test_samples < minimum_validation_samples
    ):
        raise ValueError(
            "Validation and test splits must cover goal seeking and "
            "each safety scenario."
        )

    navigation_count = max(1, round(train_samples * 0.4))
    safety_count = train_samples - navigation_count
    split_seeds = {
        "train_goal_seeking": seed,
        "train_goal_replay": seed + 100_003,
        "train_safety_and_mixed": seed + 200_003,
        "validation_goal_seeking": seed + 1_000_003,
        "validation_goal_replay": seed + 1_100_033,
        "validation_safety": seed + 1_200_037,
        "test_goal": seed + 2_000_033,
        "test_safety": seed + 2_100_037,
    }
    replay_count = max(1, round(safety_count * GOAL_REPLAY_FRACTION))
    advanced_count = safety_count - replay_count
    validation_goal_count = validation_samples // 2
    validation_safety_count = validation_samples - validation_goal_count
    test_goal_count = test_samples // 2
    test_safety_count = test_samples - test_goal_count
    datasets = {
        "train_goal_seeking": generate_teacher_dataset(
            species,
            navigation_count,
            split_seeds["train_goal_seeking"],
            map_width,
            map_height,
            scenario_types=curriculum_phases["goal_seeking"],
        ),
        "train_goal_replay": generate_teacher_dataset(
            species,
            replay_count,
            split_seeds["train_goal_replay"],
            map_width,
            map_height,
            scenario_types=curriculum_phases["goal_seeking"],
        ),
        "train_safety_examples": generate_teacher_dataset(
            species,
            advanced_count,
            split_seeds["train_safety_and_mixed"],
            map_width,
            map_height,
            scenario_types=curriculum_phases["safety_and_mixed"],
        ),
        "validation_goal_seeking": generate_teacher_dataset(
            species,
            validation_goal_count,
            split_seeds["validation_goal_seeking"],
            map_width,
            map_height,
            scenario_types=curriculum_phases["goal_seeking"],
        ),
        "validation_goal_replay": generate_teacher_dataset(
            species,
            validation_goal_count,
            split_seeds["validation_goal_replay"],
            map_width,
            map_height,
            scenario_types=curriculum_phases["goal_seeking"],
        ),
        "validation_safety": generate_teacher_dataset(
            species,
            validation_safety_count,
            split_seeds["validation_safety"],
            map_width,
            map_height,
            scenario_types=safety_scenarios,
        ),
        "test_goal": generate_teacher_dataset(
            species,
            test_goal_count,
            split_seeds["test_goal"],
            map_width,
            map_height,
            scenario_types=curriculum_phases["goal_seeking"],
        ),
        "test_safety": generate_teacher_dataset(
            species,
            test_safety_count,
            split_seeds["test_safety"],
            map_width,
            map_height,
            scenario_types=safety_scenarios,
        ),
    }
    datasets["train_safety_and_mixed"] = _combine_datasets(
        datasets["train_goal_replay"],
        datasets["train_safety_examples"],
    )
    datasets["validation"] = _combine_datasets(
        datasets["validation_goal_replay"],
        datasets["validation_safety"],
    )
    datasets["test"] = _combine_datasets(
        datasets["test_goal"],
        datasets["test_safety"],
    )
    network = _network_from_seed(neural_config, seed + 17)
    goal_epochs = max(1, max_epochs // 3)
    safety_epochs = max_epochs - goal_epochs
    network, goal_history, goal_best_epoch = train_supervised_network(
        network,
        datasets["train_goal_seeking"][0],
        datasets["train_goal_seeking"][1],
        datasets["validation_goal_seeking"][0],
        datasets["validation_goal_seeking"][1],
        seed=seed + 29,
        max_epochs=goal_epochs,
        batch_size=batch_size,
        patience=patience,
        learning_rate=learning_rate,
        l2=l2,
    )
    network, safety_history, safety_best_epoch = (
        train_supervised_network(
            network,
            datasets["train_safety_and_mixed"][0],
            datasets["train_safety_and_mixed"][1],
            datasets["validation"][0],
            datasets["validation"][1],
            seed=seed + 31,
            max_epochs=safety_epochs,
            batch_size=batch_size,
            patience=patience,
            learning_rate=learning_rate,
            l2=l2,
        )
    )
    history = [
        {"phase": "goal_seeking", **row}
        for row in goal_history
    ] + [
        {"phase": "safety_and_mixed", **row}
        for row in safety_history
    ]
    best_epochs = {
        "goal_seeking": goal_best_epoch,
        "safety_and_mixed": safety_best_epoch,
    }
    metrics = {
        split: _direction_metrics(
            network,
            dataset[0],
            dataset[1],
            l2,
        )
        for split, dataset in datasets.items()
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"best_{species}.npz"
    if model_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing model: {model_path}"
        )
    np.savez_compressed(model_path, genome=network.get_genome())
    genome_hash = hashlib.sha256(
        np.ascontiguousarray(
            network.get_genome(),
            dtype="<f8",
        ).tobytes()
    ).hexdigest()

    return {
        "species": species,
        "model_path": str(model_path),
        "genome_sha256": genome_hash,
        "seed": seed,
        "split_seeds": split_seeds,
        "sample_counts": {
            "train": train_samples,
            "validation": validation_samples,
            "test": test_samples,
        },
        "best_epoch": safety_best_epoch,
        "best_epochs_by_phase": best_epochs,
        "epochs_run": len(history),
        "metrics": metrics,
        "history": history,
        "scenario_types": list(scenario_types),
        "scenario_types_by_phase": {
            phase: list(scenario_types)
            for phase, scenario_types in curriculum_phases.items()
        },
        "unlabeled_samples_by_split": {
            split: int(
                np.count_nonzero(
                    np.linalg.norm(dataset[1], axis=1)
                    <= OUTPUT_NORM_EPSILON
                )
            )
            for split, dataset in datasets.items()
        },
        "scenario_metrics": {
            split: {
                scenario_type: _direction_metrics(
                    network,
                    np.asarray([
                        feature
                        for feature, scenario_id in zip(
                            dataset[0],
                            dataset[2],
                        )
                        if scenario_id.rsplit(":", 1)[-1]
                        == scenario_type
                    ]),
                    np.asarray([
                        label
                        for label, scenario_id in zip(
                            dataset[1],
                            dataset[2],
                        )
                        if scenario_id.rsplit(":", 1)[-1]
                        == scenario_type
                    ]),
                    l2,
                )
                for scenario_type in scenario_types
                if any(
                    scenario_id.rsplit(":", 1)[-1] == scenario_type
                    for scenario_id in dataset[2]
                )
            }
            for split, dataset in (
                ("validation", datasets["validation"]),
                ("test", datasets["test"]),
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preentrena por imitación las redes de ambas especies."
    )
    parser.add_argument("--output-dir", default="models/imitation")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-samples", type=int, default=12000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--l2", type=float, default=0.0001)
    args = parser.parse_args()

    if args.epochs < 2:
        parser.error("--epochs must be at least 2 for the curriculum.")
    minimum_train_samples = max(
        _minimum_training_samples(
            len(PREY_CURRICULUM_PHASES["safety_and_mixed"])
        ),
        _minimum_training_samples(
            len(CURRICULUM_PHASES["safety_and_mixed"])
        ),
    )
    if args.train_samples < minimum_train_samples:
        parser.error(
            "--train-samples must be at least "
            f"{minimum_train_samples} for both species' curricula."
        )
    minimum_validation_samples = max(
        2 * len(set(PREY_CURRICULUM_PHASES["safety_and_mixed"])),
        2 * len(set(CURRICULUM_PHASES["safety_and_mixed"])),
    )
    if min(args.validation_samples, args.test_samples) < (
        minimum_validation_samples
    ):
        parser.error(
            "Validation and test sample counts must be at least "
            f"{minimum_validation_samples}."
        )

    simulation_config, neural_config, _ = load_configs()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = (
        "best_prey.npz",
        "best_predator.npz",
        "imitation_metrics.json",
        "imitation_history.csv",
    )
    existing_outputs = [
        output_dir / name
        for name in output_names
        if (output_dir / name).exists()
    ]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite existing pretraining outputs: "
            + ", ".join(str(path) for path in existing_outputs)
        )
    metrics = [
        pretrain_species(
            species,
            output_dir,
            neural_config,
            simulation_config["simulation"]["width"],
            simulation_config["simulation"]["height"],
            seed=args.seed + index * 10_000_019,
            train_samples=args.train_samples,
            validation_samples=args.validation_samples,
            test_samples=args.test_samples,
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
        for index, species in enumerate(("prey", "predator"))
    ]

    metrics_path = output_dir / "imitation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "method": "heuristic_imitation",
                "seed": args.seed,
                "neural_config": neural_config,
                "optimizer": {
                    "name": "Adam",
                    "learning_rate": args.learning_rate,
                    "l2": args.l2,
                    "max_epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "patience": args.patience,
                },
                "direction_objective": {
                    "loss": (
                        "one minus cosine similarity plus an output-norm "
                        "penalty"
                    ),
                    "target_output_norm": OUTPUT_NORM_TARGET,
                    "output_norm_weight": OUTPUT_NORM_WEIGHT,
                    "goal_replay_fraction": GOAL_REPLAY_FRACTION,
                },
                "splits": {
                    "train": (
                        "independent synthetic states; goal-seeking stage "
                        "followed by safety and mixed scenarios"
                    ),
                    "validation": (
                        "independent seed and generated states spanning all "
                        "scenario families; controls early stopping"
                    ),
                    "test": (
                        "independent seed and generated states spanning all "
                        "scenario families; evaluated after model "
                        "selection only"
                    ),
                },
                "curriculum_phases": {
                    phase: list(scenario_types)
                    for phase, scenario_types in CURRICULUM_PHASES.items()
                },
                "curriculum_phases_by_species": {
                    species: {
                        phase: list(scenario_types)
                        for phase, scenario_types in curriculum.items()
                    }
                    for species, curriculum in (
                        ("prey", PREY_CURRICULUM_PHASES),
                        ("predator", CURRICULUM_PHASES),
                    )
                },
                "species": metrics,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.write("\n")

    history_path = output_dir / "imitation_history.csv"
    with history_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fieldnames = (
            "species",
            "phase",
            "epoch",
            "train_loss",
            "train_mse",
            "validation_loss",
            "validation_mse",
        )
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in metrics:
            writer.writerows(
                {
                    "species": result["species"],
                    "phase": row["phase"],
                    **{
                        key: value
                        for key, value in row.items()
                        if key != "phase"
                    },
                }
                for row in result["history"]
            )

    print(f"Pretrained models and metrics saved to: {output_dir}")


if __name__ == "__main__":
    main()
