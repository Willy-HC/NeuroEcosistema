"""Frozen evaluation of prey models across population conditions."""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np

from neuroecosistema.analysis.food_orientation_probe import (
    project_root,
    sha256_file,
)
from neuroecosistema.evolution.fitness import (
    calculate_predator_fitness,
    calculate_prey_fitness_components,
)
from neuroecosistema.neural_network.model_loader import (
    load_trained_network,
)
from neuroecosistema.simulation.environment import (
    Environment,
    Plant,
)
from neuroecosistema.train import load_configs

MODEL_IDS = ("5G", "30G")
TRAINING_SEEDS = (42, 123, 456)
MAX_STEPS = 500
OUTPUT_FILENAMES = (
    "agent_diagnostics.csv",
    "episode_summary.csv",
    "manifest.json",
)


@dataclass(frozen=True)
class PopulationCondition:
    condition_id: str
    prey_count: int
    predator_count: int


CONDITIONS = (
    PopulationCondition("prey10_pred10", 10, 10),
    PopulationCondition("prey50_pred10", 50, 10),
    PopulationCondition("prey10_pred0", 10, 0),
)

DETAIL_COLUMNS = (
    "run_id",
    "condition_id",
    "seed",
    "population_policy_type",
    "prey_model_id",
    "prey_model_sha256",
    "predator_model_id",
    "predator_model_sha256",
    "species",
    "agent_index",
    "policy_model_id",
    "policy_model_sha256",
    "prey_agents_initial",
    "predator_agents_initial",
    "prey_agents_alive",
    "predator_agents_alive",
    "layout_sha256",
    "steps_requested",
    "steps_executed",
    "initial_x",
    "initial_y",
    "final_x",
    "final_y",
    "initial_energy",
    "final_energy",
    "alive",
    "fitness",
    "fitness_survival",
    "fitness_food",
    "fitness_capture",
    "fitness_exploration",
    "fitness_stagnation",
    "food_eaten",
    "captures",
    "steps_survived",
    "regions_visited",
    "stagnation_steps",
    "distance_traveled",
    "food_approach_progress",
    "food_net_progress",
    "food_target_switches",
)

SUMMARY_METRICS = (
    "fitness",
    "fitness_survival",
    "fitness_food",
    "fitness_capture",
    "fitness_exploration",
    "fitness_stagnation",
    "food_eaten",
    "captures",
    "steps_survived",
    "regions_visited",
    "stagnation_steps",
    "distance_traveled",
    "food_approach_progress",
    "food_net_progress",
    "food_target_switches",
)


def default_model_paths():
    """Return both trained species' models for the 5G and 30G runs."""

    results_dir = project_root() / "results"
    return {
        model_id: {
            species: (
                results_dir
                / f"coevolution_diagnostics_{suffix}"
                / f"best_{species}.npz"
            )
            for species in ("prey", "predator")
        }
        for model_id, suffix in (("5G", "5gen"), ("30G", "30gen"))
    }


def _validate_seeds(seeds):
    if not seeds:
        raise ValueError("Se requiere al menos una semilla.")
    if any(
        isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0
        for seed in seeds
    ):
        raise ValueError("Las semillas deben ser enteros no negativos.")
    normalized = tuple(int(seed) for seed in seeds)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Las semillas deben ser diferentes.")
    return normalized


def _validate_max_steps(max_steps):
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, Integral)
        or max_steps < 1
    ):
        raise ValueError("max_steps debe ser un entero positivo.")
    return int(max_steps)


def _load_models(model_paths, neural_config):
    if set(model_paths) != set(MODEL_IDS):
        raise ValueError("model_paths debe contener exactamente las claves 5G y 30G.")

    expected_architecture = (
        neural_config["input_size"],
        *neural_config["hidden_layers"],
        neural_config["output_size"],
    )
    models = {}

    for model_id in MODEL_IDS:
        if set(model_paths[model_id]) != {"prey", "predator"}:
            raise ValueError(
                f"Se requieren modelos de presa y depredador para {model_id}."
            )
        models[model_id] = {}

        for species in ("prey", "predator"):
            path = Path(model_paths[model_id][species]).resolve()
            digest = sha256_file(path)
            network = load_trained_network(path, neural_config)
            architecture = (
                network.input_size,
                *network.hidden_layers,
                network.output_size,
            )
            if architecture != expected_architecture:
                raise ValueError(
                    f"Arquitectura inesperada para {model_id} {species}: "
                    f"{architecture}."
                )
            genome = network.get_genome().copy()
            if not np.all(np.isfinite(genome)):
                raise ValueError(f"El genoma de {model_id} {species} no es finito.")
            models[model_id][species] = {
                "path": path,
                "sha256": digest,
                "network": network,
                "genome": genome,
            }

    return models


def _create_initial_layout(seed, width, height, food_count):
    """Create one explicit layout shared by all conditions for a seed."""

    rng = random.Random(seed)
    plants = tuple(
        (
            rng.randint(20, width - 20),
            rng.randint(20, height - 20),
        )
        for _ in range(food_count)
    )
    prey_positions = tuple(
        (
            rng.randint(30, width - 30),
            rng.randint(30, height - 30),
        )
        for _ in range(max(condition.prey_count for condition in CONDITIONS))
    )
    predator_positions = tuple(
        (
            rng.randint(30, width - 30),
            rng.randint(30, height - 30),
        )
        for _ in range(max(condition.predator_count for condition in CONDITIONS))
    )
    payload = {
        "plants": plants,
        "prey_positions": prey_positions,
        "predator_positions": predator_positions,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return {
        **payload,
        "rng_state_after_layout": rng.getstate(),
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }


def _create_environment(
    simulation_config,
    energy_config,
    condition,
    seed,
    layout,
):
    simulation = simulation_config["simulation"]
    environment_config = simulation_config["environment"]
    food_count = environment_config["initial_food"]
    food_energy = energy_config["food_reward"]

    environment = Environment(
        width=simulation["width"],
        height=simulation["height"],
        initial_food=0,
        seed=seed,
        food_energy=food_energy,
        food_respawn=environment_config.get("food_respawn", 0),
        max_food=food_count,
    )
    environment.generate_agents(
        prey_quantity=condition.prey_count,
        predator_quantity=condition.predator_count,
    )

    environment.plants = [
        Plant(x=x, y=y, energy=food_energy) for x, y in layout["plants"]
    ]

    for agent, (x, y) in zip(
        environment.prey,
        layout["prey_positions"],
    ):
        agent.x = float(x)
        agent.y = float(y)
        agent.energy = energy_config["prey_initial"]

    for agent, (x, y) in zip(
        environment.predators,
        layout["predator_positions"],
    ):
        agent.x = float(x)
        agent.y = float(y)
        agent.energy = energy_config["predator_initial"]

    # Use the RNG state after the canonical 50-prey/10-predator layout.
    # Later respawn calls can diverge if conditions consume different amounts.
    environment.random.setstate(layout["rng_state_after_layout"])
    return environment


def _attach_networks(agents, model):
    networks = {}
    clones = []
    for agent in agents:
        network = deepcopy(model["network"])
        if not np.array_equal(network.get_genome(), model["genome"]):
            raise RuntimeError("Una copia de la red alteró el genoma.")
        networks[id(agent)] = network
        clones.append(network)
    return networks, clones


def _predator_fitness_components(predator, max_steps):
    """Decompose the existing predator fitness formula for diagnostics."""

    survival_score = min(
        1.0,
        max(0.0, predator.steps_survived / max_steps),
    )
    capture_score = min(1.0, max(0.0, predator.captures / 3))
    exploration_score = min(
        1.0,
        max(0.0, len(predator.visited_regions) / 25),
    )
    stagnation_score = min(
        1.0,
        max(0.0, predator.stagnation_steps / max_steps),
    )
    components = {
        "survival": 0.20 * survival_score,
        "capture": 0.65 * capture_score,
        "exploration": 0.15 * exploration_score,
        "stagnation": -0.10 * stagnation_score,
    }
    components["fitness"] = max(0.0, sum(components.values()))

    actual_fitness = calculate_predator_fitness(
        predator,
        max_steps,
    )
    if not np.isclose(
        components["fitness"],
        actual_fitness,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "La descomposición diagnóstica no coincide con el fitness real."
        )
    return components


def _make_agent_row(
    agent,
    species,
    index,
    policy_model_id,
    policy_model,
    run_metadata,
    initial_position,
    initial_energy,
    max_steps,
):
    if species == "prey":
        components = calculate_prey_fitness_components(
            agent,
            max_steps,
        )
        fitness_components = {
            "fitness_survival": components["survival"],
            "fitness_food": components["food"],
            "fitness_capture": None,
            "fitness_exploration": components["exploration"],
            "fitness_stagnation": components["stagnation"],
        }
        food_eaten = agent.food_eaten
        captures = None
        food_approach_progress = agent.food_approach_progress
        food_net_progress = agent.food_net_progress
        food_target_switches = agent.food_target_switches
    else:
        components = _predator_fitness_components(agent, max_steps)
        fitness_components = {
            "fitness_survival": components["survival"],
            "fitness_food": None,
            "fitness_capture": components["capture"],
            "fitness_exploration": components["exploration"],
            "fitness_stagnation": components["stagnation"],
        }
        food_eaten = None
        captures = agent.captures
        food_approach_progress = None
        food_net_progress = None
        food_target_switches = None

    return {
        **run_metadata,
        "species": species,
        "agent_index": index,
        "policy_model_id": policy_model_id,
        "policy_model_sha256": policy_model["sha256"],
        "initial_x": initial_position[0],
        "initial_y": initial_position[1],
        "final_x": agent.x,
        "final_y": agent.y,
        "initial_energy": initial_energy,
        "final_energy": agent.energy,
        "alive": agent.alive,
        "fitness": components["fitness"],
        **fitness_components,
        "food_eaten": food_eaten,
        "captures": captures,
        "steps_survived": agent.steps_survived,
        "regions_visited": len(agent.visited_regions),
        "stagnation_steps": agent.stagnation_steps,
        "distance_traveled": agent.distance_traveled,
        "food_approach_progress": food_approach_progress,
        "food_net_progress": food_net_progress,
        "food_target_switches": food_target_switches,
    }


def _run_episode(
    condition,
    seed,
    prey_model_id,
    predator_model_id,
    models,
    simulation_config,
    max_steps,
    layout,
):
    environment = _create_environment(
        simulation_config,
        simulation_config["energy"],
        condition,
        seed,
        layout,
    )
    prey_model = models[prey_model_id]["prey"]
    prey_networks, prey_clones = _attach_networks(
        environment.prey,
        prey_model,
    )
    environment.prey_networks = prey_networks

    predator_clones = []
    predator_model = None
    if condition.predator_count:
        predator_model = models[predator_model_id]["predator"]
        predator_networks, predator_clones = _attach_networks(
            environment.predators,
            predator_model,
        )
        environment.predator_networks = predator_networks

    original_prey = environment.prey[:]
    original_predators = environment.predators[:]
    initial_positions = {
        "prey": [(agent.x, agent.y) for agent in original_prey],
        "predator": [(agent.x, agent.y) for agent in original_predators],
    }
    initial_energies = {
        "prey": [agent.energy for agent in original_prey],
        "predator": [agent.energy for agent in original_predators],
    }

    steps_executed = 0
    for _ in range(max_steps):
        if not environment.prey and not environment.predators:
            break
        environment.update_agents(
            movement_cost=simulation_config["energy"]["movement_cost"],
            prey_strategy="neural",
            predator_strategy="neural",
        )
        environment.resolve_interactions(
            capture_reward=simulation_config["energy"]["capture_reward"],
        )
        environment.update_food()
        steps_executed += 1

    prey_alive = sum(agent.alive for agent in original_prey)
    predator_alive = sum(agent.alive for agent in original_predators)
    run_id = (
        f"{condition.condition_id}_seed{seed}_"
        f"prey{prey_model_id}_pred{predator_model_id or 'none'}"
    )
    common = {
        "run_id": run_id,
        "condition_id": condition.condition_id,
        "seed": seed,
        "population_policy_type": "clones",
        "prey_model_id": prey_model_id,
        "prey_model_sha256": prey_model["sha256"],
        "predator_model_id": predator_model_id,
        "predator_model_sha256": (
            predator_model["sha256"] if predator_model is not None else None
        ),
        "prey_agents_initial": len(original_prey),
        "predator_agents_initial": len(original_predators),
        "prey_agents_alive": prey_alive,
        "predator_agents_alive": predator_alive,
        "steps_requested": max_steps,
        "steps_executed": steps_executed,
    }
    rows = []
    for species, agents, policy_model_id, policy_model in (
        ("prey", original_prey, prey_model_id, prey_model),
        (
            "predator",
            original_predators,
            predator_model_id,
            predator_model,
        ),
    ):
        if policy_model is None:
            continue
        positions = initial_positions[species]
        energies = initial_energies[species]
        species_metadata = {
            **common,
        }
        rows.extend(
            _make_agent_row(
                agent,
                species,
                index,
                policy_model_id,
                policy_model,
                species_metadata,
                positions[index],
                energies[index],
                max_steps,
            )
            for index, agent in enumerate(agents)
        )

    for clones, model in (
        (prey_clones, prey_model),
        (predator_clones, predator_model),
    ):
        for network in clones:
            if not np.array_equal(
                network.get_genome(),
                model["genome"],
            ):
                raise RuntimeError("Una red cambió durante la evaluación.")
    return run_id, rows


def _summarize_episodes(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["run_id"], row["species"])].append(row)

    summaries = []
    for (_, species), agents in grouped.items():
        first = agents[0]
        summary = {
            key: first[key]
            for key in (
                "run_id",
                "condition_id",
                "seed",
                "population_policy_type",
                "prey_model_id",
                "prey_model_sha256",
                "predator_model_id",
                "predator_model_sha256",
                "layout_sha256",
                "steps_requested",
                "steps_executed",
            )
        }
        summary.update(
            {
                "species": species,
                "agents_evaluated": len(agents),
                "agents_alive": sum(bool(row["alive"]) for row in agents),
            }
        )
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in agents if row[metric] is not None]
            summary[f"mean_{metric}"] = float(np.mean(values)) if values else None
        summaries.append(summary)
    return summaries


def _write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_population_condition_probe(
    output_dir,
    seeds=TRAINING_SEEDS,
    max_steps=MAX_STEPS,
    model_paths=None,
):
    """Run frozen clone populations and write diagnostics to output_dir."""

    seeds = _validate_seeds(seeds)
    max_steps = _validate_max_steps(max_steps)
    simulation_config, neural_config, _ = load_configs()
    model_paths = default_model_paths() if model_paths is None else model_paths
    models = _load_models(model_paths, neural_config)

    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"El destino no es un directorio: {output_dir}")
    conflicts = [
        output_dir / filename
        for filename in OUTPUT_FILENAMES
        if (output_dir / filename).exists()
    ]
    if conflicts:
        raise FileExistsError(
            "No se sobrescribirán diagnósticos existentes:\n"
            + "\n".join(str(path) for path in conflicts)
        )

    simulation_config["simulation"]["max_steps"] = max_steps
    initial_food = simulation_config["environment"]["initial_food"]
    width = simulation_config["simulation"]["width"]
    height = simulation_config["simulation"]["height"]
    layouts = {
        seed: _create_initial_layout(
            seed,
            width,
            height,
            initial_food,
        )
        for seed in seeds
    }

    detail_rows = []
    episode_count = 0
    for seed in seeds:
        for condition in CONDITIONS:
            for prey_model_id in MODEL_IDS:
                predator_model_ids = MODEL_IDS if condition.predator_count else (None,)
                for predator_model_id in predator_model_ids:
                    _, episode_rows = _run_episode(
                        condition=condition,
                        seed=seed,
                        prey_model_id=prey_model_id,
                        predator_model_id=predator_model_id,
                        models=models,
                        simulation_config=simulation_config,
                        max_steps=max_steps,
                        layout=layouts[seed],
                    )
                    common = {
                        "layout_sha256": layouts[seed]["sha256"],
                    }
                    detail_rows.extend({**row, **common} for row in episode_rows)
                    episode_count += 1

    for model_id in MODEL_IDS:
        for species in ("prey", "predator"):
            model = models[model_id][species]
            if not np.array_equal(
                model["network"].get_genome(),
                model["genome"],
            ):
                raise RuntimeError(f"El genoma de {model_id} {species} fue modificado.")
            if sha256_file(model["path"]) != model["sha256"]:
                raise RuntimeError(f"El archivo de modelo {model['path']} cambió.")

    summary_rows = _summarize_episodes(detail_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "agent_diagnostics.csv"
    summary_path = output_dir / "episode_summary.csv"
    manifest_path = output_dir / "manifest.json"
    _write_csv(details_path, DETAIL_COLUMNS, detail_rows)
    summary_columns = (
        "run_id",
        "condition_id",
        "seed",
        "population_policy_type",
        "prey_model_id",
        "prey_model_sha256",
        "predator_model_id",
        "predator_model_sha256",
        "layout_sha256",
        "steps_requested",
        "steps_executed",
        "species",
        "agents_evaluated",
        "agents_alive",
        *(f"mean_{metric}" for metric in SUMMARY_METRICS),
    )
    _write_csv(summary_path, summary_columns, summary_rows)

    manifest = {
        "tool": "population_condition_probe",
        "seeds": list(seeds),
        "training_seeds": list(TRAINING_SEEDS),
        "additional_seeds": [seed for seed in seeds if seed not in TRAINING_SEEDS],
        "max_steps": max_steps,
        "effective_config": {
            "simulation": {
                "width": width,
                "height": height,
                "max_steps": max_steps,
            },
            "environment": {
                "initial_food": initial_food,
                "max_food": initial_food,
                "food_respawn": simulation_config["environment"].get(
                    "food_respawn",
                    0,
                ),
                "food_energy": simulation_config["energy"]["food_reward"],
            },
            "energy": simulation_config["energy"],
            "neural_network": neural_config,
        },
        "configured_but_not_applied": {
            "environment.obstacle_count": (
                "The production Environment does not create obstacles."
            )
        },
        "conditions": [
            {
                "condition_id": condition.condition_id,
                "prey_agents": condition.prey_count,
                "predator_agents": condition.predator_count,
            }
            for condition in CONDITIONS
        ],
        "population_policy_type": "clones of one genome per species",
        "predator_model_design": (
            "Full 5G/30G predator-policy cross: prey and predator model "
            "effects are not conflated."
        ),
        "comparability": {
            "initial_layouts_are_shared": True,
            "shared_layout_fields": [
                "initial plants",
                "first 10 prey positions",
                "first 10 predator positions when present",
            ],
            "layout_generation": (
                "Seeded explicit layout independent of Environment RNG."
            ),
            "respawn_rng": (
                "Initialized to the same post-layout RNG state; future "
                "respawn sequences can diverge if conditions consume "
                "different numbers of plants."
            ),
            "conditions_change": (
                "Agent counts; predator presence; predator policy is "
                "crossed with prey policy in predator-present conditions."
            ),
            "conditions_held_constant": (
                "World dimensions, initial plant coordinates, initial "
                "positions of shared agents, energy, movement cost, "
                "respawn settings, seed, and episode length."
            ),
        },
        "models": {
            model_id: {
                species: {
                    "path": str(model["path"]),
                    "sha256": model["sha256"],
                    "parameter_count": int(model["genome"].size),
                }
                for species, model in species_models.items()
            }
            for model_id, species_models in models.items()
        },
        "initial_layouts": {
            str(seed): {
                "sha256": layout["sha256"],
                "plants": layout["plants"],
                "prey_positions": layout["prey_positions"],
                "predator_positions": layout["predator_positions"],
            }
            for seed, layout in layouts.items()
        },
        "episode_count": episode_count,
        "agent_diagnostic_rows": len(detail_rows),
        "episode_summary_rows": len(summary_rows),
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)

    return {
        "details": details_path,
        "summary": summary_path,
        "manifest": manifest_path,
        "episode_count": episode_count,
        "agent_diagnostic_rows": len(detail_rows),
        "episode_summary_rows": len(summary_rows),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evalúa modelos congelados en condiciones de población controladas."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directorio explícito para los CSV y el manifiesto.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(TRAINING_SEEDS),
        help="Semillas no negativas; por defecto 42, 123 y 456.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_STEPS,
        help="Pasos por episodio (por defecto: 500).",
    )
    args = parser.parse_args()

    outputs = run_population_condition_probe(
        output_dir=args.output_dir,
        seeds=args.seeds,
        max_steps=args.max_steps,
    )
    print(
        f"Episodios: {outputs['episode_count']}; "
        f"filas por agente: {outputs['agent_diagnostic_rows']}; "
        f"resúmenes: {outputs['episode_summary_rows']}"
    )
    for name in ("details", "summary", "manifest"):
        print(f"{name}: {outputs[name]}")


if __name__ == "__main__":
    main()
