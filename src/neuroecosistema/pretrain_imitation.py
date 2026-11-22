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
SCENARIO_TYPES = (
    "objective_only",
    "threat_only",
    "objective_and_threat",
    "no_objective",
    "near_wall_objective",
    "near_wall_threat",
    "multiple_targets",
)
SCENARIO_SPLITS = {
    "train": (
        "objective_only",
        "threat_only",
        "objective_and_threat",
        "no_objective",
        "near_wall_objective",
    ),
    "validation": ("near_wall_threat",),
    "test": ("multiple_targets",),
}


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

    if direction is None:
        direction = (1.0, 0.0)

    effective_direction = apply_wall_avoidance(
        agent,
        direction[0],
        direction[1],
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
        }:
            count = 3 if scenario_type == "multiple_targets" else 1
            for index in range(count):
                distance_range = (
                    (12.0, 65.0)
                    if index == 0
                    else (80.0, 500.0)
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
        }:
            threat_range = (
                (12.0, 90.0)
                if scenario_type in {
                    "threat_only",
                    "near_wall_threat",
                }
                else (35.0, 300.0)
            )
            count = 3 if scenario_type == "multiple_targets" else 1
            for _ in range(count):
                x, y = _place_relative(
                    rng,
                    anchor,
                    threat_range,
                    width,
                    height,
                    radius=6,
                )
                predators.append(Predator(x=x, y=y))
        if near_wall and rng.random() < 0.5:
            agent.wall_escape_active = True
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
        if near_wall and rng.random() < 0.5:
            agent.wall_escape_active = True
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

    label = imitation_teacher_direction(
        agent,
        species,
        environment,
    )
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
    if not scenario_types or any(
        scenario_type not in SCENARIO_TYPES
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
            activation = np.tanh(preactivation)
        else:
            activation = np.maximum(0.0, preactivation)
        activations.append(activation)

    return activation, activations, preactivations


def _loss(network, features, labels, l2):
    predictions = network.forward(features)
    if not np.all(np.isfinite(predictions)):
        raise FloatingPointError("Network produced non-finite outputs.")
    mse = float(np.mean((predictions - labels) ** 2))
    penalty = 0.5 * l2 * sum(
        float(np.sum(weights**2))
        for weights in network.weights
    )
    return mse + penalty, mse


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
    batch_size, output_size = labels.shape
    gradient = 2.0 * (predictions - labels) / (batch_size * output_size)

    gradients = []
    for layer_index in reversed(range(len(network.weights))):
        if layer_index == len(network.weights) - 1:
            gradient = gradient * (1.0 - predictions**2)
        else:
            gradient = gradient * (
                preactivations[layer_index] > 0.0
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
            step += 1
            _train_batch(
                network,
                train_features[batch_indices],
                train_labels[batch_indices],
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
    valid = (predicted_norm > 1e-8) & (labels_norm > 1e-8)
    cosine = np.sum(predicted[valid] * labels[valid], axis=1)
    cosine /= predicted_norm[valid] * labels_norm[valid]
    angular_error = np.degrees(
        np.arccos(np.clip(cosine, -1.0, 1.0))
    )
    return {
        "loss": float(loss),
        "mse": float(mse),
        "mean_angular_error_degrees": (
            float(np.mean(angular_error))
            if angular_error.size
            else 0.0
        ),
        "evaluated_directions": int(angular_error.size),
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
    split_seeds = {
        "train": seed,
        "validation": seed + 1_000_003,
        "test": seed + 2_000_033,
    }
    datasets = {
        split: generate_teacher_dataset(
            species,
            count,
            split_seeds[split],
            map_width,
            map_height,
            scenario_types=SCENARIO_SPLITS[split],
        )
        for split, count in (
            ("train", train_samples),
            ("validation", validation_samples),
            ("test", test_samples),
        )
    }
    network = _network_from_seed(neural_config, seed + 17)
    network, history, best_epoch = train_supervised_network(
        network,
        datasets["train"][0],
        datasets["train"][1],
        datasets["validation"][0],
        datasets["validation"][1],
        seed=seed + 29,
        max_epochs=max_epochs,
        batch_size=batch_size,
        patience=patience,
        learning_rate=learning_rate,
        l2=l2,
    )
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
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "metrics": metrics,
        "history": history,
        "scenario_types": list(SCENARIO_TYPES),
        "scenario_types_by_split": {
            split: list(scenario_types)
            for split, scenario_types in SCENARIO_SPLITS.items()
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

    if min(
        args.train_samples,
        args.validation_samples,
        args.test_samples,
    ) < len(SCENARIO_TYPES):
        parser.error(
            f"Each split needs at least {len(SCENARIO_TYPES)} scenarios."
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
                "splits": {
                    "train": (
                        "scenario types reserved for optimization; "
                        "instances generated from the train seed"
                    ),
                    "validation": (
                        "held-out scenario types; controls early stopping"
                    ),
                    "test": (
                        "held-out scenario types; evaluated after model "
                        "selection only"
                    ),
                },
                "scenario_types_by_split": {
                    split: list(scenario_types)
                    for split, scenario_types in SCENARIO_SPLITS.items()
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
                    **row,
                }
                for row in result["history"]
            )

    print(f"Pretrained models and metrics saved to: {output_dir}")


if __name__ == "__main__":
    main()
