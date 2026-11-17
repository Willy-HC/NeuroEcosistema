
import numpy as np

from neuroecosistema.evolution.crossover import uniform_crossover
from neuroecosistema.evolution.mutation import gaussian_mutation
from neuroecosistema.evolution.selection import (
    select_elites,
    tournament_selection,
)


class GeneticAlgorithm:
    """Algoritmo genético para evolucionar genomas neuronales."""

    def __init__(
        self,
        population_size=50,
        elitism=5,
        mutation_rate=0.10,
        mutation_strength=0.15,
        crossover_rate=0.70,
        tournament_size=3,
        seed=42,
    ):
        if population_size < 2:
            raise ValueError(
                "La población debe tener al menos dos individuos."
            )

        if not 0 <= elitism <= population_size:
            raise ValueError("El elitismo es inválido.")

        if not 1 <= tournament_size <= population_size:
            raise ValueError("El tamaño de torneo es inválido.")

        if not 0 <= mutation_rate <= 1:
            raise ValueError("La tasa de mutación es inválida.")

        if not np.isfinite(mutation_strength) or mutation_strength < 0:
            raise ValueError("La intensidad de mutación es inválida.")

        if not 0 <= crossover_rate <= 1:
            raise ValueError("La tasa de cruce es inválida.")

        self.population_size = population_size
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

        self.rng = np.random.default_rng(seed)
        self.generation = 0
        self.history = []

    def evolve(self, genomes, fitness_values):
        """Genera una nueva población a partir de una evaluada."""

        if len(genomes) != self.population_size:
            raise ValueError(
                "La población no coincide con el tamaño configurado."
            )

        if len(fitness_values) != self.population_size:
            raise ValueError(
                "La cantidad de fitness no coincide con la población."
            )

        genomes = [
            np.asarray(genome, dtype=np.float64)
            for genome in genomes
        ]

        genome_size = genomes[0].size

        if genome_size == 0 or any(
            genome.ndim != 1 or genome.size != genome_size
            for genome in genomes
        ):
            raise ValueError(
                "Todos los genomas deben ser vectores del mismo tamaño."
            )

        if any(
            not np.all(np.isfinite(genome))
            for genome in genomes
        ):
            raise ValueError("Los genomas deben ser finitos.")

        fitness_values = np.asarray(
            fitness_values,
            dtype=np.float64,
        )

        if not np.all(np.isfinite(fitness_values)):
            raise ValueError("Los fitness deben ser finitos.")

        # Registrar estadísticas de la generación evaluada.
        statistics = {
            "generation": self.generation,
            "best_fitness": float(np.max(fitness_values)),
            "mean_fitness": float(np.mean(fitness_values)),
            "worst_fitness": float(np.min(fitness_values)),
            "std_fitness": float(np.std(fitness_values)),
        }

        # Conservar los mejores individuos.
        new_population = select_elites(
            genomes,
            fitness_values,
            elitism=self.elitism,
        )

        # Generar descendientes.
        while len(new_population) < self.population_size:
            parent_a_index = tournament_selection(
                fitness_values,
                tournament_size=self.tournament_size,
                rng=self.rng,
            )

            parent_b_index = tournament_selection(
                fitness_values,
                tournament_size=self.tournament_size,
                rng=self.rng,
            )

            parent_a = genomes[parent_a_index]
            parent_b = genomes[parent_b_index]

            child = uniform_crossover(
                parent_a,
                parent_b,
                crossover_rate=self.crossover_rate,
                rng=self.rng,
            )

            child = gaussian_mutation(
                child,
                mutation_rate=self.mutation_rate,
                mutation_strength=self.mutation_strength,
                rng=self.rng,
            )

            new_population.append(child)

        self.history.append(statistics)
        self.generation += 1

        return new_population

    def get_history(self):
        """Devuelve una copia del historial de generaciones."""
        return [statistics.copy() for statistics in self.history]
    