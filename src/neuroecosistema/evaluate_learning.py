"""Compare controller families on reproducible held-out scenarios."""

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from neuroecosistema.evolution.behavior_evaluator import (
    BehaviorEvaluator,
)
from neuroecosistema.neural_network.model_loader import (
    load_trained_network,
)
from neuroecosistema.neural_network.network import NeuralNetwork
from neuroecosistema.train import load_configs

DEFAULT_SEEDS = (789, 1013, 2027)
DEFAULT_SCENARIOS = (
    (10, 2, 30),
    (30, 5, 60),
    (50, 10, 100),
)
POLICIES = ("random", "heuristic", "untrained", "pretrained", "coevolved")
METRICS = (
    "mean_survival",
    "mean_food_eaten",
    "mean_captures",
    "mean_exploration",
    "mean_wall_ratio",
    "mean_distance",
    "mean_food_approach_progress",
    "mean_food_net_progress",
    "mean_food_target_switches",
    "mean_food_progress_score",
    "mean_food_orientation_score",
    "mean_evasion_distance",
    "mean_threat_steps",
    "mean_pursuit_distance",
    "mean_pursuit_steps",
)


def parse_scenario(value):
    try:
        prey_count, predator_count, plant_count = (
            int(part) for part in value.split(",")
        )
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "Scenario format is PREY,PREDATORS,PLANTS."
        ) from error
    if prey_count < 1 or predator_count < 0 or plant_count < 0:
        raise argparse.ArgumentTypeError(
            "A scenario needs prey >= 1 and nonnegative other counts."
        )
    return prey_count, predator_count, plant_count


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _create_untrained_models(output_dir, neural_config, seed):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_paths = [
        output_dir / f"best_{species}.npz"
        for species in ("prey", "predator")
    ]
    conflicts = [path for path in model_paths if path.exists()]
    if conflicts:
        raise FileExistsError(
            "Refusing to overwrite untrained models: "
            + ", ".join(str(path) for path in conflicts)
        )

    hashes = {}
    initialization_seeds = {}
    rng = np.random.default_rng(seed)
    for species in ("prey", "predator"):
        network_seed = int(rng.integers(0, 2**32))
        network = NeuralNetwork(
            input_size=neural_config["input_size"],
            hidden_layers=neural_config["hidden_layers"],
            output_size=neural_config["output_size"],
            hidden_activation=neural_config["activation_hidden"],
            output_activation=neural_config["activation_output"],
            seed=network_seed,
        )
        path = output_dir / f"best_{species}.npz"
        np.savez_compressed(path, genome=network.get_genome())
        hashes[species] = _sha256(path)
        initialization_seeds[species] = network_seed
    return hashes, initialization_seeds


def _policy_model_dirs(output_dir, pretrained_dir, coevolved_dir):
    untrained_dir = output_dir / "untrained_models"
    return {
        "random": None,
        "heuristic": None,
        "untrained": untrained_dir,
        "pretrained": Path(pretrained_dir),
        "coevolved": Path(coevolved_dir),
    }


def _validate_model_dir(model_dir, neural_config):
    for species in ("prey", "predator"):
        model_path = model_dir / f"best_{species}.npz"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Missing {species} model: {model_path}"
            )
        load_trained_network(model_path, neural_config)


def _episode_rows(
    evaluator,
    scenario_id,
    scenario,
    seed,
    policy,
    strategy,
    model_dir,
):
    prey_count, predator_count, plant_count = scenario
    result = evaluator.evaluate(
        seed=seed,
        model_dir=model_dir,
        strategy=strategy,
        observe_food_orientation=True,
        observe_tactical_progress=True,
    )
    orientation = result.get("food_orientation", {})
    orientation_scores = [
        item["score"]
        for item in orientation.values()
        if item["score"] is not None
    ]
    orientation_mean = (
        float(np.mean(orientation_scores))
        if orientation_scores
        else 0.0
    )

    rows = []
    for species in ("prey", "predator"):
        if species not in result:
            continue
        metrics = result[species]
        row = {
            "scenario_id": scenario_id,
            "episode_seed": seed,
            "policy": policy,
            "species": species,
            "prey_count": prey_count,
            "predator_count": predator_count,
            "plant_count": plant_count,
            "agents": metrics["agents"],
            "steps_executed": metrics["steps_executed"],
        }
        for metric in METRICS:
            row[metric] = metrics.get(metric, "")
        if species == "prey":
            row["mean_food_orientation_score"] = orientation_mean
        rows.append(row)
    return rows


def _summarize(rows):
    group_fields = (
        "scenario_id",
        "policy",
        "species",
        "prey_count",
        "predator_count",
        "plant_count",
    )
    groups = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups.setdefault(key, []).append(row)

    summary = []
    for key, group in sorted(groups.items()):
        record = dict(zip(group_fields, key))
        record["episodes"] = len(group)
        for metric in METRICS:
            values = [
                float(row[metric])
                for row in group
                if row[metric] != ""
            ]
            record[f"{metric}_mean"] = (
                float(np.mean(values)) if values else ""
            )
            record[f"{metric}_std"] = (
                float(np.std(values)) if values else ""
            )
        summary.append(record)
    return summary


def run_learning_comparison(
    simulation_config,
    neural_config,
    *,
    pretrained_model_dir,
    coevolved_model_dir,
    output_dir,
    seeds=DEFAULT_SEEDS,
    scenarios=DEFAULT_SCENARIOS,
    max_steps=500,
    untrained_seed=812198,
):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be nonempty and unique.")
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = (
        "learning_episodes.csv",
        "learning_summary.csv",
        "learning_manifest.json",
    )
    conflicts = [
        output_dir / name
        for name in output_names
        if (output_dir / name).exists()
    ]
    if conflicts:
        raise FileExistsError(
            "Refusing to overwrite evaluation results: "
            + ", ".join(str(path) for path in conflicts)
        )

    pretrained_dir = Path(pretrained_model_dir)
    coevolved_dir = Path(coevolved_model_dir)
    _validate_model_dir(pretrained_dir, neural_config)
    _validate_model_dir(coevolved_dir, neural_config)

    untrained_dir = output_dir / "untrained_models"
    untrained_hashes, untrained_initialization_seeds = (
        _create_untrained_models(
            untrained_dir,
            neural_config,
            untrained_seed,
        )
    )
    model_dirs = _policy_model_dirs(
        output_dir,
        pretrained_dir,
        coevolved_dir,
    )
    rows = []

    for scenario_index, scenario in enumerate(scenarios):
        scenario_config = deepcopy(simulation_config)
        scenario_config["simulation"]["max_steps"] = max_steps
        scenario_config["population"]["prey"] = scenario[0]
        scenario_config["population"]["predators"] = scenario[1]
        scenario_config["environment"]["initial_food"] = scenario[2]
        scenario_id = f"scenario_{scenario_index + 1}"

        for seed in seeds:
            for policy in POLICIES:
                evaluator = BehaviorEvaluator(
                    scenario_config,
                    neural_config,
                )
                rows.extend(
                    _episode_rows(
                        evaluator,
                        scenario_id,
                        scenario,
                        seed,
                        policy,
                        (
                            "neural"
                            if policy in {
                                "untrained",
                                "pretrained",
                                "coevolved",
                            }
                            else policy
                        ),
                        model_dirs[policy],
                    )
                )

    episode_fields = [
        "scenario_id",
        "episode_seed",
        "policy",
        "species",
        "prey_count",
        "predator_count",
        "plant_count",
        "agents",
        "steps_executed",
        *METRICS,
    ]
    _write_csv(
        output_dir / "learning_episodes.csv",
        rows,
        episode_fields,
    )
    summary = _summarize(rows)
    _write_csv(
        output_dir / "learning_summary.csv",
        summary,
        list(summary[0]),
    )

    model_hashes = {
        policy: (
            {
                species: _sha256(
                    model_dirs[policy] / f"best_{species}.npz"
                )
                for species in ("prey", "predator")
            }
            if model_dirs[policy] is not None
            else None
        )
        for policy in POLICIES
    }
    model_hashes["untrained"] = untrained_hashes
    manifest = {
        "evaluation": "held-out controller comparison",
        "policies": list(POLICIES),
        "episode_seeds": list(seeds),
        "scenarios": [
            {
                "scenario_id": f"scenario_{index + 1}",
                "prey": scenario[0],
                "predators": scenario[1],
                "plants": scenario[2],
            }
            for index, scenario in enumerate(scenarios)
        ],
        "max_steps": max_steps,
        "untrained_initialization_seed": untrained_seed,
        "untrained_initialization_seeds": (
            untrained_initialization_seeds
        ),
        "model_directories": {
            "pretrained": str(pretrained_dir.resolve()),
            "coevolved": str(coevolved_dir.resolve()),
            "untrained": str(untrained_dir.resolve()),
        },
        "model_sha256": model_hashes,
        "neural_config": neural_config,
        "simulation_configuration": {
            "width": simulation_config["simulation"]["width"],
            "height": simulation_config["simulation"]["height"],
            "food_respawn": simulation_config["environment"].get(
                "food_respawn",
                0,
            ),
            "energy": simulation_config["energy"],
        },
        "metric_definitions": {
            "mean_evasion_distance": (
                "Mean positive prey movement away from its nearest "
                "predator frozen at the start of a step, only within "
                "180 map units."
            ),
            "mean_pursuit_distance": (
                "Mean positive predator closure toward its nearest prey "
                "frozen at the start of a step."
            ),
            "mean_food_progress_score": (
                "Per-prey realized food-consumption progress score."
            ),
        },
        "results": summary,
    }
    with (output_dir / "learning_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")

    return {
        "episodes": output_dir / "learning_episodes.csv",
        "summary": output_dir / "learning_summary.csv",
        "manifest": output_dir / "learning_manifest.json",
    }


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compara políticas en escenarios reproducibles."
    )
    parser.add_argument("--pretrained-model-dir", required=True)
    parser.add_argument("--coevolved-model-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/learning_evaluation",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        type=parse_scenario,
        dest="scenarios",
        help="PREY,PREDATORS,PLANTS; repetir para añadir escenarios.",
    )
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--untrained-seed",
        type=int,
        default=812198,
    )
    args = parser.parse_args()

    simulation_config, neural_config, _ = load_configs()
    paths = run_learning_comparison(
        simulation_config,
        neural_config,
        pretrained_model_dir=args.pretrained_model_dir,
        coevolved_model_dir=args.coevolved_model_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        scenarios=args.scenarios or DEFAULT_SCENARIOS,
        max_steps=args.max_steps,
        untrained_seed=args.untrained_seed,
    )
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
