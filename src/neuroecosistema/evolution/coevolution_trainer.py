
import csv
import json
from pathlib import Path

import numpy as np

from neuroecosistema.evolution.episode_evaluator import (
    EpisodeEvaluator,
)
from neuroecosistema.evolution.trainer import (
    EvolutionTrainer,
    _average_prey_diagnostics,
)

FITNESS_DIAGNOSTICS_COLUMNS = (
    "generation",
    "genome_index",
    "fitness",
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

COEVOLUTION_OUTPUT_FILENAMES = (
    "best_prey.npz",
    "history_prey.csv",
    "metadata_prey.json",
    "best_predator.npz",
    "history_predator.csv",
    "metadata_predator.json",
    "fitness_diagnostics_prey.csv",
)


class CoevolutionTrainer:
    """Evoluciona simultáneamente presas y depredadores."""

    def __init__(
        self,
        simulation_config,
        neural_config,
        evolution_config,
        evaluation_seeds=(42, 123, 456),
        fitness_mode="baseline",
        food_progress_weight=0.05,
        pretrained_prey_path=None,
        pretrained_predator_path=None,
        pretrained_fraction=0.5,
        pretrained_noise=0.02,
        hall_of_fame_size=3,
        historical_opponent_weight=0.2,
    ):
        if hall_of_fame_size < 0:
            raise ValueError("El tamaño del archivo no puede ser negativo.")
        if not 0.0 <= historical_opponent_weight < 1.0:
            raise ValueError(
                "El peso de rivales históricos debe estar en [0, 1)."
            )

        self.simulation_config = simulation_config
        self.neural_config = neural_config
        self.config = evolution_config
        self.evaluation_seeds = tuple(evaluation_seeds)
        self.fitness_mode = fitness_mode
        self.food_progress_weight = food_progress_weight
        self.hall_of_fame_size = int(hall_of_fame_size)
        self.historical_opponent_weight = float(
            historical_opponent_weight
        )
        self.prey_hall_of_fame = []
        self.predator_hall_of_fame = []

        self.prey_trainer = EvolutionTrainer(
            simulation_config=simulation_config,
            neural_config=neural_config,
            evolution_config=evolution_config,
            species="prey",
            evaluation_seeds=evaluation_seeds,
            fitness_mode=fitness_mode,
            food_progress_weight=food_progress_weight,
            pretrained_model_path=pretrained_prey_path,
            pretrained_fraction=pretrained_fraction,
            pretrained_noise=pretrained_noise,
        )

        # Utilizar otra semilla genética para evitar
        # poblaciones iniciales idénticas.
        predator_config = evolution_config.copy()
        predator_config["seed"] = (
            int(evolution_config["seed"]) + 1
        )

        self.predator_trainer = EvolutionTrainer(
            simulation_config=simulation_config,
            neural_config=neural_config,
            evolution_config=predator_config,
            species="predator",
            evaluation_seeds=evaluation_seeds,
            fitness_mode="baseline",
            food_progress_weight=food_progress_weight,
            pretrained_model_path=pretrained_predator_path,
            pretrained_fraction=pretrained_fraction,
            pretrained_noise=pretrained_noise,
        )

        self.generations = self.prey_trainer.generations
        self.record_diagnostics = False
        self.fitness_diagnostics = []

    @staticmethod
    def validate_output_directory(output_dir):
        """Rechaza destinos que contienen resultados de otra ejecución."""

        output_dir = Path(output_dir)

        if output_dir.exists() and not output_dir.is_dir():
            raise FileExistsError(
                f"El destino no es un directorio: {output_dir}"
            )

        conflicts = [
            output_dir / filename
            for filename in COEVOLUTION_OUTPUT_FILENAMES
            if (output_dir / filename).exists()
        ]

        if conflicts:
            conflict_list = "\n".join(
                str(path) for path in conflicts
            )
            raise FileExistsError(
                "No se sobrescribirán resultados existentes:\n"
                f"{conflict_list}"
            )

    def evaluate_populations(
        self,
        prey_genomes,
        predator_genomes,
        generation=0,
        include_diagnostics=False,
    ):
        """Promedia el fitness de ambas especies entre episodios."""

        prey_results = []
        predator_results = []
        diagnostic_results = []
        historical_prey_fitness = None
        historical_predator_fitness = None
        historical_seed_index = generation % len(self.evaluation_seeds)

        for index, seed in enumerate(self.evaluation_seeds):
            evaluator = EpisodeEvaluator(
                simulation_config=self.simulation_config,
                neural_config=self.neural_config,
                seed=seed,
                fitness_mode=self.fitness_mode,
                food_progress_weight=self.food_progress_weight,
            )

            rotation = generation + index

            if include_diagnostics:
                (
                    prey_fitness,
                    predator_fitness,
                    diagnostics,
                ) = evaluator.evaluate_both(
                    prey_genomes,
                    predator_genomes,
                    rotation=rotation,
                    include_diagnostics=True,
                )
                diagnostic_results.append(diagnostics)
            else:
                prey_fitness, predator_fitness = (
                    evaluator.evaluate_both(
                        prey_genomes,
                        predator_genomes,
                        rotation=rotation,
                    )
                )

            prey_results.append(prey_fitness)
            predator_results.append(predator_fitness)

            if (
                self.hall_of_fame_size
                and self.prey_hall_of_fame
                and self.predator_hall_of_fame
                and index == historical_seed_index
            ):
                historical_predators = self._historical_population(
                    self.predator_hall_of_fame,
                    len(prey_genomes),
                    generation + index + 1,
                )
                historical_prey = self._historical_population(
                    self.prey_hall_of_fame,
                    len(predator_genomes),
                    generation + index + 1,
                )
                historical_prey_fitness, _ = evaluator.evaluate_both(
                    prey_genomes,
                    historical_predators,
                    rotation=rotation,
                )
                _, historical_predator_fitness = (
                    evaluator.evaluate_both(
                        historical_prey,
                        predator_genomes,
                        rotation=rotation,
                    )
                )

        current_prey_fitness = np.mean(
            np.stack(prey_results),
            axis=0,
        )
        current_predator_fitness = np.mean(
            np.stack(predator_results),
            axis=0,
        )
        mean_prey_fitness = current_prey_fitness
        mean_predator_fitness = current_predator_fitness
        if (
            historical_prey_fitness is not None
            and historical_predator_fitness is not None
        ):
            history_weight = self.historical_opponent_weight
            mean_prey_fitness = (
                (1.0 - history_weight) * current_prey_fitness
                + history_weight * historical_prey_fitness
            )
            mean_predator_fitness = (
                (1.0 - history_weight) * current_predator_fitness
                + history_weight * historical_predator_fitness
            )

        if include_diagnostics:
            diagnostics = _average_prey_diagnostics(
                diagnostic_results,
                mean_prey_fitness,
            )
            return (
                mean_prey_fitness,
                mean_predator_fitness,
                diagnostics,
            )

        return mean_prey_fitness, mean_predator_fitness

    @staticmethod
    def _historical_population(archive, population_size, offset):
        return [
            archive[(index + offset) % len(archive)].copy()
            for index in range(population_size)
        ]

    @staticmethod
    def _remember_champion(archive, genomes, fitness, maximum_size):
        if maximum_size == 0:
            return
        champion = np.asarray(
            genomes[int(np.argmax(fitness))],
            dtype=np.float64,
        ).copy()
        archive[:] = [
            genome
            for genome in archive
            if not np.array_equal(genome, champion)
        ]
        archive.append(champion)
        del archive[:-maximum_size]

    @staticmethod
    def register_statistics(trainer, genomes, fitness, generation):
        """Registra estadísticas y conserva el mejor genoma."""

        best_index = int(np.argmax(fitness))
        generation_best = float(fitness[best_index])

        if generation_best > trainer.best_fitness:
            trainer.best_fitness = generation_best
            trainer.best_genome = genomes[best_index].copy()
            trainer.best_generation = generation

        statistics = {
            "generation": generation,
            "best_fitness": generation_best,
            "mean_fitness": float(np.mean(fitness)),
            "worst_fitness": float(np.min(fitness)),
            "std_fitness": float(np.std(fitness)),
            "best_so_far": trainer.best_fitness,
        }

        trainer.history.append(statistics)

    def train(self, record_diagnostics=False):
        """Ejecuta generaciones de coevolución."""

        self.record_diagnostics = record_diagnostics
        self.fitness_diagnostics = []

        prey_genomes = (
            self.prey_trainer.create_initial_population()
        )

        predator_genomes = (
            self.predator_trainer.create_initial_population()
        )

        for generation_index in range(self.generations):
            generation = generation_index + 1

            if record_diagnostics:
                (
                    prey_fitness,
                    predator_fitness,
                    diagnostics,
                ) = self.evaluate_populations(
                    prey_genomes,
                    predator_genomes,
                    generation=generation_index,
                    include_diagnostics=True,
                )
                self.fitness_diagnostics.extend(
                    {
                        "generation": generation,
                        **diagnostic,
                    }
                    for diagnostic in diagnostics
                )
            else:
                prey_fitness, predator_fitness = (
                    self.evaluate_populations(
                        prey_genomes,
                        predator_genomes,
                        generation=generation_index,
                    )
                )

            self.register_statistics(
                self.prey_trainer,
                prey_genomes,
                prey_fitness,
                generation,
            )

            self.register_statistics(
                self.predator_trainer,
                predator_genomes,
                predator_fitness,
                generation,
            )
            self._remember_champion(
                self.prey_hall_of_fame,
                prey_genomes,
                prey_fitness,
                self.hall_of_fame_size,
            )
            self._remember_champion(
                self.predator_hall_of_fame,
                predator_genomes,
                predator_fitness,
                self.hall_of_fame_size,
            )

            print(
                f"[coevolution] "
                f"Generación {generation}/{self.generations} | "
                f"Presas: {np.max(prey_fitness):.4f} | "
                f"Depredadores: {np.max(predator_fitness):.4f}",
                flush=True,
            )

            if generation_index < self.generations - 1:
                # Ambas especies evolucionan usando los
                # resultados del mismo conjunto de episodios.
                prey_genomes = (
                    self.prey_trainer.algorithm.evolve(
                        prey_genomes,
                        prey_fitness,
                    )
                )

                predator_genomes = (
                    self.predator_trainer.algorithm.evolve(
                        predator_genomes,
                        predator_fitness,
                    )
                )

        return (
            self.prey_trainer.best_genome.copy(),
            self.predator_trainer.best_genome.copy(),
        )

    def save_results(self, output_dir="results"):
        """Guarda modelos, historiales y metadatos."""

        output_dir = Path(output_dir)

        if self.record_diagnostics:
            self.validate_output_directory(output_dir)

        prey_paths = self.prey_trainer.save_results(
            output_dir
        )

        predator_paths = self.predator_trainer.save_results(
            output_dir
        )
        coevolution_metadata = {
            "hall_of_fame_size": self.hall_of_fame_size,
            "historical_opponent_weight": (
                self.historical_opponent_weight
            ),
            "pretrained_prey_path": (
                self.prey_trainer.pretrained_model_path
            ),
            "pretrained_predator_path": (
                self.predator_trainer.pretrained_model_path
            ),
            "pretrained_fraction": (
                self.prey_trainer.pretrained_fraction
            ),
            "pretrained_noise": self.prey_trainer.pretrained_noise,
        }
        for paths in (prey_paths, predator_paths):
            metadata_path = paths["metadata"]
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            metadata["coevolution"] = coevolution_metadata
            metadata_path.write_text(
                json.dumps(metadata, indent=4) + "\n",
                encoding="utf-8",
            )

        if self.record_diagnostics:
            diagnostics_path = (
                output_dir / "fitness_diagnostics_prey.csv"
            )
            with diagnostics_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=FITNESS_DIAGNOSTICS_COLUMNS,
                )
                writer.writeheader()
                writer.writerows(self.fitness_diagnostics)

        return {
            "prey": prey_paths,
            "predator": predator_paths,
        }