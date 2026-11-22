"""Run a bounded DAgger pilot for prey imitation."""

import argparse
import csv
import hashlib
import json
import math
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np

from neuroecosistema.agents.heuristic_controller import (
    calculate_heuristic_direction,
)
from neuroecosistema.agents.perception import perceive_prey
from neuroecosistema.evolution.behavior_evaluator import BehaviorEvaluator
from neuroecosistema.neural_network.model_loader import load_trained_network
from neuroecosistema.pretrain_imitation import (
    SCENARIO_TYPES,
    generate_teacher_dataset,
    train_supervised_network,
)
from neuroecosistema.simulation.environment import Environment
from neuroecosistema.train import load_configs

SCENARIOS = (
    ("low_density", (10, 2, 30)),
    ("high_density", (50, 10, 100)),
)
DEFAULT_EVAL_SEEDS = (8401201, 8401202, 8401203)
MAX_DAGGER_ROUNDS = 3
REPLAY_SAMPLES = 512
SURVIVAL_RETENTION_FLOOR = 0.9
UPDATE_EPOCHS = 8
UPDATE_BATCH_SIZE = 256
UPDATE_PATIENCE = 2
UPDATE_LEARNING_RATE = 0.0002
UPDATE_L2 = 0.0001


def collect_state_observations(environment):
    """Return exact prey perceptions and deterministic heuristic targets."""

    features = []
    labels = []
    for prey in environment.prey:
        direction = calculate_heuristic_direction(prey, environment)
        if direction is None:
            continue

        perception = perceive_prey(
            prey,
            environment.plants,
            environment.predators,
            environment.width,
            environment.height,
        )
        label = np.asarray(direction, dtype=np.float32)
        if (
            not np.all(np.isfinite(perception))
            or not np.all(np.isfinite(label))
            or np.linalg.norm(label) <= 1e-8
        ):
            continue
        features.append(perception)
        labels.append(label)

    return (
        np.asarray(features, dtype=np.float32).reshape(-1, 8),
        np.asarray(labels, dtype=np.float32).reshape(-1, 2),
    )


def _environment_for_episode(simulation_config, seed, scenario):
    simulation = simulation_config["simulation"]
    world = simulation_config["environment"]
    energy = simulation_config["energy"]
    prey_count, predator_count, plant_count = scenario
    environment = Environment(
        width=simulation["width"],
        height=simulation["height"],
        initial_food=plant_count,
        seed=seed,
        food_energy=energy["food_reward"],
        food_respawn=world.get("food_respawn", 0),
        max_food=plant_count,
    )
    environment.generate_agents(
        prey_quantity=prey_count,
        predator_quantity=predator_count,
    )
    return environment


def collect_dagger_episode(
    simulation_config,
    neural_config,
    genome,
    seed,
    scenario,
    max_steps,
):
    """Collect expert labels on states visited by the neural prey policy."""

    environment = _environment_for_episode(
        simulation_config,
        seed,
        scenario,
    )
    environment.initialize_neural_networks(neural_config)
    for network in environment.prey_networks.values():
        network.set_genome(genome)

    feature_batches = []
    label_batches = []
    energy = simulation_config["energy"]
    for _ in range(max_steps):
        if not environment.prey:
            break
        features, labels = collect_state_observations(environment)
        if len(features):
            feature_batches.append(features)
            label_batches.append(labels)

        environment.update_agents(
            movement_cost=energy["movement_cost"],
            prey_strategy="neural",
            predator_strategy="random",
        )
        environment.resolve_interactions(
            capture_reward=energy["capture_reward"],
        )
        environment.update_food()

    if not feature_batches:
        return (
            np.empty((0, 8), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )
    return (
        np.concatenate(feature_batches),
        np.concatenate(label_batches),
    )


def _episode_seed(seed, round_index, scenario_index, split_index):
    return seed + round_index * 10_000 + scenario_index * 100 + split_index


def _collect_round_data(
    simulation_config,
    neural_config,
    genome,
    seed,
    round_index,
    collection_steps,
):
    training = []
    validation = []
    training_seeds = []
    validation_seeds = []
    for scenario_index, (_, scenario) in enumerate(SCENARIOS):
        for split_index in (1, 2):
            episode_seed = _episode_seed(
                seed,
                round_index,
                scenario_index,
                split_index,
            )
            training_seeds.append(episode_seed)
            training.append(
                collect_dagger_episode(
                    simulation_config,
                    neural_config,
                    genome,
                    episode_seed,
                    scenario,
                    collection_steps,
                )
            )

        validation_seed = _episode_seed(
            seed,
            round_index,
            scenario_index,
            99,
        )
        validation_seeds.append(validation_seed)
        validation.append(
            collect_dagger_episode(
                simulation_config,
                neural_config,
                genome,
                validation_seed,
                scenario,
                collection_steps,
            )
        )

    return training, validation, training_seeds, validation_seeds


def _join_batches(batches, width):
    nonempty = [batch for batch in batches if len(batch)]
    if not nonempty:
        return np.empty((0, width), dtype=np.float32)
    return np.concatenate(nonempty)


def _summarize_evaluations(rows):
    summaries = {}
    for policy in dict.fromkeys(row["policy"] for row in rows):
        summaries[policy] = {}
        for scenario_name, _ in SCENARIOS:
            selected = [
                row
                for row in rows
                if row["policy"] == policy
                and row["scenario"] == scenario_name
            ]
            summaries[policy][scenario_name] = {
                key: float(np.mean([row[key] for row in selected]))
                for key in ("mean_food_eaten", "mean_survival_steps")
            }
    return summaries


def behavior_score(candidate, random_baseline, initial_policy):
    """Score the weakest feeding/survival ratio across both scenarios."""

    scenario_scores = []
    for scenario_name, _ in SCENARIOS:
        candidate_metrics = candidate[scenario_name]
        random_metrics = random_baseline[scenario_name]
        initial_metrics = initial_policy[scenario_name]
        food_ratio = candidate_metrics["mean_food_eaten"] / max(
            random_metrics["mean_food_eaten"],
            1e-8,
        )
        survival_ratio = candidate_metrics["mean_survival_steps"] / max(
            initial_metrics["mean_survival_steps"],
            1e-8,
        )
        scenario_scores.append(min(food_ratio, survival_ratio))
    return min(scenario_scores)


def meets_acceptance(candidate, random_baseline, initial_policy):
    for scenario_name, _ in SCENARIOS:
        candidate_metrics = candidate[scenario_name]
        if candidate_metrics["mean_food_eaten"] <= (
            random_baseline[scenario_name]["mean_food_eaten"]
        ):
            return False
        if candidate_metrics["mean_survival_steps"] < (
            SURVIVAL_RETENTION_FLOOR
            * initial_policy[scenario_name]["mean_survival_steps"]
        ):
            return False
    return True


def _evaluate_policy(
    simulation_config,
    neural_config,
    policy,
    model_dir,
    seeds,
    round_index,
    max_steps,
):
    rows = []
    for scenario_name, scenario in SCENARIOS:
        config = deepcopy(simulation_config)
        config["simulation"]["max_steps"] = max_steps
        config["population"]["prey"] = scenario[0]
        config["population"]["predators"] = scenario[1]
        config["environment"]["initial_food"] = scenario[2]
        evaluator = BehaviorEvaluator(config, neural_config)
        for seed in seeds:
            result = evaluator.evaluate(
                seed=seed,
                model_dir=model_dir,
                prey_strategy=(
                    "neural"
                    if policy in {"initial_neural"} or policy.startswith(
                        "dagger_round_"
                    )
                    else policy
                ),
                predator_strategy="random",
            )
            rows.append({
                "round": round_index,
                "policy": policy,
                "scenario": scenario_name,
                "seed": seed,
                "mean_food_eaten": result["prey"]["mean_food_eaten"],
                "mean_survival_steps": result["prey"]["mean_survival"],
                "steps_executed": result["prey"]["steps_executed"],
            })
    return rows


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_genome(path, genome):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, genome=genome)


def run_dagger(
    simulation_config,
    neural_config,
    *,
    initial_model_path,
    output_dir,
    seed=8122019,
    rounds=MAX_DAGGER_ROUNDS,
    collection_steps=100,
    evaluation_steps=500,
    evaluation_seeds=DEFAULT_EVAL_SEEDS,
    replay_samples=REPLAY_SAMPLES,
):
    if not 1 <= rounds <= MAX_DAGGER_ROUNDS:
        raise ValueError("DAgger rounds must be between 1 and 3.")
    if collection_steps < 1 or evaluation_steps < 1:
        raise ValueError("Episode lengths must be positive.")
    if not evaluation_seeds or len(set(evaluation_seeds)) != len(
        evaluation_seeds
    ):
        raise ValueError("Evaluation seeds must be nonempty and unique.")
    if replay_samples < len(SCENARIO_TYPES):
        raise ValueError(
            f"replay_samples must be at least {len(SCENARIO_TYPES)}."
        )

    initial_model_path = Path(initial_model_path)
    if not initial_model_path.is_file():
        raise FileNotFoundError(
            f"Initial prey model does not exist: {initial_model_path}"
        )
    initial_hash = _sha256_file(initial_model_path)
    collection_seeds = [
        _episode_seed(seed, round_index, scenario_index, split_index)
        for round_index in range(1, rounds + 1)
        for scenario_index in range(len(SCENARIOS))
        for split_index in (1, 2, 99)
    ]
    replay_seed = seed + 50_000_003
    if set(collection_seeds + [replay_seed]).intersection(evaluation_seeds):
        raise ValueError("Evaluation seeds must be distinct from DAgger seeds.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    initial_dir = output_dir / "initial"
    initial_dir.mkdir()
    initial_copy = initial_dir / "best_prey.npz"
    shutil.copy2(initial_model_path, initial_copy)
    network = load_trained_network(initial_copy, neural_config)

    baseline_rows = []
    baseline_rows.extend(_evaluate_policy(
        simulation_config,
        neural_config,
        "initial_neural",
        initial_dir,
        evaluation_seeds,
        0,
        evaluation_steps,
    ))
    for policy in ("random", "heuristic"):
        baseline_rows.extend(_evaluate_policy(
            simulation_config,
            neural_config,
            policy,
            None,
            evaluation_seeds,
            0,
            evaluation_steps,
        ))
    baseline_summary = _summarize_evaluations(baseline_rows)
    random_baseline = baseline_summary["random"]
    initial_summary = baseline_summary["initial_neural"]

    replay_features, replay_labels, _ = generate_teacher_dataset(
        "prey",
        replay_samples,
        replay_seed,
        simulation_config["simulation"]["width"],
        simulation_config["simulation"]["height"],
    )
    retained_features = []
    retained_labels = []
    round_summaries = []
    evaluation_rows = list(baseline_rows)
    best_candidate_score = -math.inf
    best_candidate_round = None
    best_candidate_genome = None

    for round_index in range(1, rounds + 1):
        round_training, round_validation, train_seeds, val_seeds = (
            _collect_round_data(
                simulation_config,
                neural_config,
                network.get_genome().copy(),
                seed,
                round_index,
                collection_steps,
            )
        )
        retained_features.extend(
            features for features, _ in round_training if len(features)
        )
        retained_labels.extend(
            labels for _, labels in round_training if len(labels)
        )
        train_features = np.concatenate(
            [replay_features, *retained_features],
        )
        train_labels = np.concatenate(
            [replay_labels, *retained_labels],
        )
        validation_features = _join_batches([
            features for features, _ in round_validation
        ], 8)
        validation_labels = _join_batches([
            labels for _, labels in round_validation
        ], 2)
        if not np.any(np.linalg.norm(train_labels, axis=1) > 1e-8):
            raise RuntimeError("DAgger collected no deterministic targets.")
        if (
            not len(validation_features)
            or not np.any(
                np.linalg.norm(validation_labels, axis=1) > 1e-8
            )
        ):
            raise RuntimeError(
                "DAgger validation episodes contained no labeled targets."
            )

        network, history, best_epoch = train_supervised_network(
            network,
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            seed=seed + round_index * 1_000_003,
            max_epochs=UPDATE_EPOCHS,
            batch_size=UPDATE_BATCH_SIZE,
            patience=UPDATE_PATIENCE,
            learning_rate=UPDATE_LEARNING_RATE,
            l2=UPDATE_L2,
        )
        round_dir = output_dir / f"round_{round_index}"
        _save_genome(round_dir / "best_prey.npz", network.get_genome())
        policy = f"dagger_round_{round_index}"
        rows = _evaluate_policy(
            simulation_config,
            neural_config,
            policy,
            round_dir,
            evaluation_seeds,
            round_index,
            evaluation_steps,
        )
        evaluation_rows.extend(rows)
        candidate_summary = _summarize_evaluations(rows)[policy]
        score = behavior_score(
            candidate_summary,
            random_baseline,
            initial_summary,
        )
        accepted = meets_acceptance(
            candidate_summary,
            random_baseline,
            initial_summary,
        )
        round_summaries.append({
            "round": round_index,
            "policy": policy,
            "training_seeds": train_seeds,
            "validation_seeds": val_seeds,
            "training_observations": int(
                sum(len(features) for features, _ in round_training)
            ),
            "retained_training_observations": int(
                sum(len(batch) for batch in retained_features)
            ),
            "validation_observations": len(validation_features),
            "optimizer_best_epoch": best_epoch,
            "optimizer_epochs_run": len(history),
            "validation_loss": history[best_epoch - 1][
                "validation_loss"
            ],
            "behavior_score": score,
            "meets_acceptance": accepted,
            "evaluation": candidate_summary,
        })
        if score > best_candidate_score:
            best_candidate_score = score
            best_candidate_round = round_index
            best_candidate_genome = network.get_genome().copy()

    if best_candidate_genome is None:
        raise RuntimeError("DAgger did not produce a candidate model.")
    best_model_path = output_dir / "best_prey.npz"
    _save_genome(best_model_path, best_candidate_genome)
    if _sha256_file(initial_model_path) != initial_hash:
        raise RuntimeError("The initial model changed during the DAgger run.")
    best_round = round_summaries[best_candidate_round - 1]
    metrics = {
        "method": "prey_dagger_pilot",
        "initial_model": str(initial_model_path),
        "initial_model_sha256": initial_hash,
        "initial_model_unchanged": (
            _sha256_file(initial_model_path) == initial_hash
        ),
        "best_model": str(best_model_path),
        "best_model_sha256": _sha256_file(best_model_path),
        "best_round": best_candidate_round,
        "best_behavior_score": best_candidate_score,
        "dagger_candidate_meets_acceptance": best_round["meets_acceptance"],
        "survival_retention_floor": SURVIVAL_RETENTION_FLOOR,
        "round_limit": MAX_DAGGER_ROUNDS,
        "rounds_run": rounds,
        "collection_steps": collection_steps,
        "evaluation_steps": evaluation_steps,
        "evaluation_seeds": list(evaluation_seeds),
        "training_seed": seed,
        "scenarios": [
            {"name": name, "counts": list(counts)}
            for name, counts in SCENARIOS
        ],
        "optimizer": {
            "name": "existing supervised Adam",
            "max_epochs": UPDATE_EPOCHS,
            "batch_size": UPDATE_BATCH_SIZE,
            "patience": UPDATE_PATIENCE,
            "learning_rate": UPDATE_LEARNING_RATE,
            "l2": UPDATE_L2,
        },
        "replay_samples": replay_samples,
        "baseline": baseline_summary,
        "rounds": round_summaries,
        "evaluation_rows": evaluation_rows,
    }
    with (output_dir / "dagger_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
        file.write("\n")

    fields = (
        "round",
        "policy",
        "scenario",
        "seed",
        "mean_food_eaten",
        "mean_survival_steps",
        "steps_executed",
    )
    with (output_dir / "dagger_evaluations.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(evaluation_rows)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run up to three prey-only DAgger rounds."
    )
    parser.add_argument(
        "--initial-model",
        default="models/imitation/best_prey.npz",
    )
    parser.add_argument(
        "--output-dir",
        default="results/prey_dagger_pilot",
    )
    parser.add_argument("--seed", type=int, default=8122019)
    parser.add_argument("--rounds", type=int, default=MAX_DAGGER_ROUNDS)
    parser.add_argument("--collection-steps", type=int, default=100)
    parser.add_argument("--evaluation-steps", type=int, default=500)
    parser.add_argument(
        "--evaluation-seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_EVAL_SEEDS),
    )
    args = parser.parse_args()

    simulation_config, neural_config, _ = load_configs()
    metrics = run_dagger(
        simulation_config,
        neural_config,
        initial_model_path=args.initial_model,
        output_dir=args.output_dir,
        seed=args.seed,
        rounds=args.rounds,
        collection_steps=args.collection_steps,
        evaluation_steps=args.evaluation_steps,
        evaluation_seeds=args.evaluation_seeds,
    )
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "best_round": metrics["best_round"],
                "dagger_candidate_meets_acceptance": (
                    metrics["dagger_candidate_meets_acceptance"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
