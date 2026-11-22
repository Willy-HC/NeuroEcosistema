
import csv
import json
from pathlib import Path

import numpy as np

from neuroecosistema.evolution.episode_evaluator import EpisodeEvaluator
from neuroecosistema.evolution.fitness import validate_fitness_options
from neuroecosistema.evolution.genetic_algorithm import GeneticAlgorithm
from neuroecosistema.neural_network.network import NeuralNetwork


def _average_prey_diagnostics(diagnostics_by_evaluation, fitness_values):
    """Promedia diagnósticos por índice y conserva el fitness agregado."""

    diagnostic_fields = (
        "survival",
        "food",
        "exploration",
        "stagnation",
        "food_eaten",
        "steps_survived",
        "visited_regions",
        "stagnation_steps",
        "food_progress_score",
        "food_progress",
    )

    indexed_diagnostics = [
        {
            diagnostic["genome_index"]: diagnostic
            for diagnostic in evaluation
        }
        for evaluation in diagnostics_by_evaluation
    ]

    return [
        {
            "genome_index": genome_index,
            **{
                field: float(
                    np.mean(
                        [
                            evaluation[genome_index][field]
                            for evaluation in indexed_diagnostics
                        ]
                    )
                )
                for field in diagnostic_fields
            },
            "fitness": float(fitness_values[genome_index]),
        }
        for genome_index in range(len(fitness_values))
    ]


class EvolutionTrainer:
    """Entrena redes neuronales mediante neuroevolución."""

    def __init__(
        self,
        simulation_config,
        neural_config,
        evolution_config,
        species="prey",
        evaluation_seeds=(42, 123, 456),
        fitness_mode="baseline",
        food_progress_weight=0.05,
    ):
        if species not in ("prey", "predator"):
            raise ValueError("Especie no válida.")

        if not evaluation_seeds:
            raise ValueError(
                "Se necesita al menos una semilla de evaluación."
            )

        (
            self.fitness_mode,
            self.food_progress_weight,
        ) = validate_fitness_options(
            fitness_mode,
            food_progress_weight,
        )
        if species == "predator" and self.fitness_mode != "baseline":
            raise ValueError(
                "El progreso alimentario solo puede aplicarse "
                "al fitness de presas."
            )

        if len(set(evaluation_seeds)) != len(evaluation_seeds):
            raise ValueError(
                "Las semillas de evaluación deben ser diferentes."
            )

        self.simulation_config = simulation_config
        self.neural_config = neural_config
        self.config = evolution_config
        self.species = species
        self.evaluation_seeds = tuple(evaluation_seeds)

        self.generations = int(self.config["generations"])

        if self.generations < 1:
            raise ValueError(
                "Debe configurarse al menos una generación."
            )

        self.algorithm = GeneticAlgorithm(
            population_size=self.config["population_size"],
            elitism=self.config["elitism"],
            mutation_rate=self.config["mutation_rate"],
            mutation_strength=self.config["mutation_strength"],
            crossover_rate=self.config["crossover_rate"],
            seed=self.config["seed"],
        )

        if self.config["selection"] != "tournament":
            raise ValueError(
                "Actualmente solo se admite selección por torneo."
            )

        self.rng = np.random.default_rng(self.config["seed"])

        self.best_genome = None
        self.best_fitness = float("-inf")
        self.best_generation = None

        self.history = []

    def create_initial_population(self):
        """Genera una población inicial de genomas."""

        genomes = []

        for _ in range(self.algorithm.population_size):
            network = NeuralNetwork(
                input_size=self.neural_config["input_size"],
                hidden_layers=self.neural_config["hidden_layers"],
                output_size=self.neural_config["output_size"],
                hidden_activation=self.neural_config[
                    "activation_hidden"
                ],
                output_activation=self.neural_config[
                    "activation_output"
                ],
                seed=int(self.rng.integers(0, 2**32)),
            )

            genomes.append(network.get_genome())

        return genomes

    def evaluate_population(self, genomes, include_diagnostics=False):
        """Calcula el fitness promedio usando varias semillas."""

        results = []
        diagnostics_by_evaluation = []

        for seed in self.evaluation_seeds:
            evaluator = EpisodeEvaluator(
                simulation_config=self.simulation_config,
                neural_config=self.neural_config,
                seed=seed,
                fitness_mode=self.fitness_mode,
                food_progress_weight=self.food_progress_weight,
            )

            if include_diagnostics:
                fitness, diagnostics = evaluator.evaluate(
                    genomes,
                    species=self.species,
                    include_diagnostics=True,
                )
                if self.species == "prey":
                    diagnostics_by_evaluation.append(diagnostics)
            else:
                fitness = evaluator.evaluate(
                    genomes,
                    species=self.species,
                )

            results.append(fitness)

        mean_fitness = np.mean(
            np.stack(results, axis=0),
            axis=0,
        )

        if include_diagnostics:
            diagnostics = (
                _average_prey_diagnostics(
                    diagnostics_by_evaluation,
                    mean_fitness,
                )
                if self.species == "prey"
                else []
            )
            return mean_fitness, diagnostics

        return mean_fitness

    def train(self):
        """Ejecuta todas las generaciones de entrenamiento."""

        genomes = self.create_initial_population()

        for generation in range(self.generations):
            fitness_values = self.evaluate_population(genomes)

            best_index = int(np.argmax(fitness_values))
            generation_best = float(fitness_values[best_index])
            generation_mean = float(np.mean(fitness_values))

            if generation_best > self.best_fitness:
                self.best_fitness = generation_best
                self.best_genome = genomes[best_index].copy()
                self.best_generation = generation + 1

            statistics = {
                "generation": generation + 1,
                "best_fitness": generation_best,
                "mean_fitness": generation_mean,
                "worst_fitness": float(np.min(fitness_values)),
                "std_fitness": float(np.std(fitness_values)),
                "best_so_far": self.best_fitness,
            }

            self.history.append(statistics)

            print(
                f"[{self.species}] "
                f"Generación {generation + 1}/{self.generations} | "
                f"Mejor: {generation_best:.4f} | "
                f"Promedio: {generation_mean:.4f} | "
                f"Récord: {self.best_fitness:.4f}",
                flush=True,
            )

            # La última población ya fue evaluada.
            # No generamos descendientes que no serán evaluados.
            if generation < self.generations - 1:
                genomes = self.algorithm.evolve(
                    genomes,
                    fitness_values,
                )

        return self.best_genome.copy()

    def save_results(self, output_dir="results"):
        """Guarda el mejor genoma y el historial."""

        if self.best_genome is None or not self.history:
            raise RuntimeError(
                "Debes ejecutar train() antes de guardar."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"best_{self.species}.npz"
        history_path = output_dir / f"history_{self.species}.csv"
        metadata_path = output_dir / f"metadata_{self.species}.json"

        np.savez_compressed(
            model_path,
            genome=self.best_genome,
        )

        with history_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(self.history[0].keys()),
            )
            writer.writeheader()
            writer.writerows(self.history)

        metadata = {
            "species": self.species,
            "best_fitness": self.best_fitness,
            "best_generation": self.best_generation,
            "generations": self.generations,
            "evaluation_seeds": list(self.evaluation_seeds),
            "fitness_mode": self.fitness_mode,
            "food_progress_weight": self.food_progress_weight,
            "neural_config": self.neural_config,
            "evolution_config": self.config,
            "simulation_config": self.simulation_config,
        }

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=4)

        return {
            "model": model_path,
            "history": history_path,
            "metadata": metadata_path,
        }