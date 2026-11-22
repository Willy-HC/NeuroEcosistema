"""Calibrate hypothetical food-progress weights on frozen genotypes."""

import argparse
import csv
import hashlib
import json
import math
import os
from itertools import combinations
from pathlib import Path

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import numpy as np

from neuroecosistema.evolution.episode_evaluator import EpisodeEvaluator
from neuroecosistema.neural_network.network import NeuralNetwork
from neuroecosistema.train import load_configs

POPULATION_SIZE = 10
POPULATION_SEED = 812198
EPISODE_SEEDS = (42, 123, 456)
MAX_STEPS = 500
WEIGHTS = (0.0, 0.05, 0.10, 0.20, 0.30)
ELITE_FRACTION = 0.10
TOURNAMENT_SIZE = 3


def _weight_key(weight):
    return f"{weight:.2f}".replace(".", "_")


def _genome_hash(genome):
    canonical = np.ascontiguousarray(genome, dtype="<f8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _generate_population(neural_config, species_seed):
    rng = np.random.default_rng(species_seed)
    genomes = []
    records = []
    seen_hashes = set()
    attempts = 0

    while len(genomes) < POPULATION_SIZE:
        attempts += 1
        if attempts > POPULATION_SIZE * 100:
            raise RuntimeError(
                "No fue posible generar diez genomas distintos."
            )

        initialization_seed = int(rng.integers(0, 2**32))
        network = NeuralNetwork(
            input_size=neural_config["input_size"],
            hidden_layers=neural_config["hidden_layers"],
            output_size=neural_config["output_size"],
            hidden_activation=neural_config["activation_hidden"],
            output_activation=neural_config["activation_output"],
            seed=initialization_seed,
        )
        genome = network.get_genome()
        digest = _genome_hash(genome)
        if digest in seen_hashes:
            continue

        seen_hashes.add(digest)
        genomes.append(genome)
        records.append({
            "genome_index": len(genomes) - 1,
            "initialization_seed": initialization_seed,
            "genome_hash": digest,
        })

    return genomes, records


def _strict_inversions(baseline, hypothetical):
    inversions = 0
    comparable_pairs = 0

    for left, right in combinations(range(len(baseline)), 2):
        baseline_delta = baseline[left] - baseline[right]
        hypothetical_delta = (
            hypothetical[left] - hypothetical[right]
        )
        if baseline_delta == 0 or hypothetical_delta == 0:
            continue

        comparable_pairs += 1
        if baseline_delta * hypothetical_delta < 0:
            inversions += 1

    return inversions, comparable_pairs


def _winner(candidates, fitness):
    return max(candidates, key=lambda index: (fitness[index], -index))


def _summarize_weight(seed, weight, baseline, scores):
    nominal_bonus = weight * scores
    hypothetical = np.clip(baseline + nominal_bonus, 0.0, 1.0)
    baseline_elite_count = max(
        1,
        math.ceil(POPULATION_SIZE * ELITE_FRACTION),
    )
    baseline_elites = sorted(
        range(POPULATION_SIZE),
        key=lambda index: (-baseline[index], index),
    )[:baseline_elite_count]
    hypothetical_elites = sorted(
        range(POPULATION_SIZE),
        key=lambda index: (-hypothetical[index], index),
    )[:baseline_elite_count]

    tournament_sets = tuple(
        combinations(range(POPULATION_SIZE), TOURNAMENT_SIZE)
    )
    changed_tournaments = sum(
        _winner(candidates, baseline)
        != _winner(candidates, hypothetical)
        for candidates in tournament_sets
    )
    inversions, comparable_pairs = _strict_inversions(
        baseline,
        hypothetical,
    )

    return {
        "episode_seed": seed,
        "weight": weight,
        "strict_ranking_inversions": inversions,
        "comparable_pairs": comparable_pairs,
        "elite_count": baseline_elite_count,
        "baseline_elite_indices": baseline_elites,
        "hypothetical_elite_indices": hypothetical_elites,
        "elite_changed": baseline_elites != hypothetical_elites,
        "elite_changes": len(
            set(hypothetical_elites) - set(baseline_elites)
        ),
        "tournament_size": TOURNAMENT_SIZE,
        "tournament_sets_evaluated": len(tournament_sets),
        "tournament_winner_changes": changed_tournaments,
        "nominal_bonus_mean": float(np.mean(nominal_bonus)),
        "nominal_bonus_max": float(np.max(nominal_bonus)),
        "effective_fitness_delta_mean": float(
            np.mean(hypothetical - baseline)
        ),
        "effective_fitness_delta_max": float(
            np.max(hypothetical - baseline)
        ),
        "saturation_count": int(np.count_nonzero(hypothetical >= 1.0)),
        "saturation_fraction": float(
            np.mean(hypothetical >= 1.0)
        ),
    }


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_calibration(output_dir, population_seed=POPULATION_SEED):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    simulation_config, neural_config, _ = load_configs()
    simulation_config["simulation"]["max_steps"] = MAX_STEPS

    species_sequences = np.random.SeedSequence(population_seed).spawn(2)
    species_names = ("prey", "predator")
    species_seeds = {
        species: int(sequence.generate_state(1)[0])
        for species, sequence in zip(species_names, species_sequences)
    }

    populations = {}
    genotype_records = []
    for species in species_names:
        genomes, records = _generate_population(
            neural_config,
            species_seeds[species],
        )
        populations[species] = genomes
        genotype_records.extend(
            {"species": species, **record}
            for record in records
        )

    genome_results = []
    weight_summaries = []

    for episode_seed in EPISODE_SEEDS:
        evaluator = EpisodeEvaluator(
            simulation_config=simulation_config,
            neural_config=neural_config,
            seed=episode_seed,
            fitness_mode="baseline",
        )
        prey_fitness, predator_fitness, prey_diagnostics = (
            evaluator.evaluate_both(
                populations["prey"],
                populations["predator"],
                include_diagnostics=True,
            )
        )

        if not np.array_equal(
            prey_fitness,
            np.asarray(
                [item["fitness"] for item in prey_diagnostics],
                dtype=np.float64,
            ),
        ):
            raise RuntimeError(
                "El fitness base no coincide con los diagnósticos."
            )

        prey_scores = np.asarray(
            [
                item["food_progress_score"]
                for item in prey_diagnostics
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(prey_scores)) or np.any(
            (prey_scores < 0.0) | (prey_scores > 1.0)
        ):
            raise ValueError(
                "Se obtuvo un food_progress_score fuera de [0, 1]."
            )

        for item, genome_record, score in zip(
            prey_diagnostics,
            genotype_records[:POPULATION_SIZE],
            prey_scores,
        ):
            row = {
                "episode_seed": episode_seed,
                "species": "prey",
                "genome_index": item["genome_index"],
                "genome_hash": genome_record["genome_hash"],
                "initialization_seed": genome_record[
                    "initialization_seed"
                ],
                "baseline_fitness": float(item["fitness"]),
                "food_progress_score": float(score),
                "plants_consumed": int(item["food_eaten"]),
            }
            for weight in WEIGHTS:
                key = _weight_key(weight)
                bonus = weight * score
                row[f"nominal_bonus_w{key}"] = float(bonus)
                row[f"fitness_w{key}"] = float(
                    np.clip(item["fitness"] + bonus, 0.0, 1.0)
                )
            genome_results.append(row)

        predator_records = genotype_records[
            POPULATION_SIZE:POPULATION_SIZE * 2
        ]
        for index, (fitness, genome_record) in enumerate(
            zip(predator_fitness, predator_records)
        ):
            genome_results.append({
                "episode_seed": episode_seed,
                "species": "predator",
                "genome_index": index,
                "genome_hash": genome_record["genome_hash"],
                "initialization_seed": genome_record[
                    "initialization_seed"
                ],
                "baseline_fitness": float(fitness),
                "food_progress_score": "",
                "plants_consumed": "",
            })

        for weight in WEIGHTS:
            weight_summaries.append(
                _summarize_weight(
                    episode_seed,
                    weight,
                    prey_fitness,
                    prey_scores,
                )
            )

    individual_fields = [
        "episode_seed",
        "species",
        "genome_index",
        "genome_hash",
        "initialization_seed",
        "baseline_fitness",
        "food_progress_score",
        "plants_consumed",
    ]
    for weight in WEIGHTS:
        key = _weight_key(weight)
        individual_fields.extend(
            [f"nominal_bonus_w{key}", f"fitness_w{key}"]
        )

    _write_csv(
        output_dir / "genome_results.csv",
        genome_results,
        individual_fields,
    )
    _write_csv(
        output_dir / "weight_summary.csv",
        weight_summaries,
        list(weight_summaries[0]),
    )

    project_root = Path(__file__).resolve().parents[3]
    config_hashes = {
        name: hashlib.sha256(
            (project_root / "config" / name).read_bytes()
        ).hexdigest()
        for name in ("simulation.yaml", "neural_network.yaml")
    }
    manifest = {
        "diagnostic": "food_reward_weight_calibration",
        "evaluation_mode": "baseline",
        "population_size_per_species": POPULATION_SIZE,
        "population_seed": int(population_seed),
        "species_population_seeds": species_seeds,
        "episode_seeds": list(EPISODE_SEEDS),
        "steps_per_episode": MAX_STEPS,
        "episode_count": len(EPISODE_SEEDS),
        "episode_runs_per_seed": 1,
        "weights_analyzed": list(WEIGHTS),
        "elite_fraction": ELITE_FRACTION,
        "tournament_size": TOURNAMENT_SIZE,
        "tournament_method": (
            "Exhaustive enumeration of all unique 3-genome subsets; "
            "ties go to the lowest genome index."
        ),
        "ranking_tie_method": (
            "Strict inversions exclude pairs tied in either fitness vector."
        ),
        "fitness_formula": (
            "clip(baseline_fitness + weight * food_progress_score, 0, 1)"
        ),
        "food_progress_metric_scope": (
            "Prey only; predator CSV values are intentionally blank."
        ),
        "simulation_config": simulation_config,
        "neural_config": neural_config,
        "config_sha256": config_hashes,
        "genotypes": genotype_records,
        "genome_results": genome_results,
        "weight_summary": weight_summaries,
        "output_files": [
            "genome_results.csv",
            "weight_summary.csv",
            "manifest.json",
        ],
        "training_performed": False,
    }
    with (output_dir / "manifest.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa pesos hipotéticos de progreso alimentario sin entrenar."
        )
    )
    parser.add_argument(
        "--output",
        default="results/food_reward_weight_calibration",
        help="Directorio de salida para CSV y JSON.",
    )
    parser.add_argument(
        "--population-seed",
        type=int,
        default=POPULATION_SEED,
        help="Semilla reproducible de inicialización de genomas.",
    )
    args = parser.parse_args()

    output_dir = run_calibration(
        args.output,
        population_seed=args.population_seed,
    )
    print(f"Diagnostic results saved to: {output_dir}")


if __name__ == "__main__":
    main()
