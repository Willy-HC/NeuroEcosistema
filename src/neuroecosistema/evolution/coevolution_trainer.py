
import numpy as np

from neuroecosistema.evolution.episode_evaluator import (
    EpisodeEvaluator,
)
from neuroecosistema.evolution.trainer import (
    EvolutionTrainer,
    _average_prey_diagnostics,
)


class CoevolutionTrainer:
    """Evoluciona simultáneamente presas y depredadores."""

    def __init__(
        self,
        simulation_config,
        neural_config,
        evolution_config,
        evaluation_seeds=(42, 123, 456),
    ):
        self.simulation_config = simulation_config
        self.neural_config = neural_config
        self.config = evolution_config
        self.evaluation_seeds = tuple(evaluation_seeds)

        self.prey_trainer = EvolutionTrainer(
            simulation_config=simulation_config,
            neural_config=neural_config,
            evolution_config=evolution_config,
            species="prey",
            evaluation_seeds=evaluation_seeds,
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
        )

        self.generations = self.prey_trainer.generations

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

        for index, seed in enumerate(self.evaluation_seeds):
            evaluator = EpisodeEvaluator(
                simulation_config=self.simulation_config,
                neural_config=self.neural_config,
                seed=seed,
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

        mean_prey_fitness = np.mean(
            np.stack(prey_results),
            axis=0,
        )
        mean_predator_fitness = np.mean(
            np.stack(predator_results),
            axis=0,
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

    def train(self):
        """Ejecuta generaciones de coevolución."""

        prey_genomes = (
            self.prey_trainer.create_initial_population()
        )

        predator_genomes = (
            self.predator_trainer.create_initial_population()
        )

        for generation_index in range(self.generations):
            generation = generation_index + 1

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

        prey_paths = self.prey_trainer.save_results(
            output_dir
        )

        predator_paths = self.predator_trainer.save_results(
            output_dir
        )

        return {
            "prey": prey_paths,
            "predator": predator_paths,
        }