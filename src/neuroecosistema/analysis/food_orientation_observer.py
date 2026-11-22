import math

from neuroecosistema.analysis.food_orientation_tracker import (
    FoodOrientationTracker,
)


class FoodOrientationObserver:
    """Observa el progreso alimentario sin alterar la simulación."""

    def __init__(self, environment):
        self.environment = environment
        self.trackers = {}
        self.targets = {}

        diagonal = math.hypot(
            environment.width,
            environment.height,
        )

        if diagonal <= 0:
            raise ValueError("La diagonal del mapa debe ser positiva.")

        for prey in environment.prey:
            target = environment.nearest_plant(prey, environment.plants)

            tracker = FoodOrientationTracker(map_diagonal=diagonal)

            prey_id = id(prey)
            self.trackers[prey_id] = tracker

            if target is None:
                continue

            target_id = id(target)
            distance = math.dist(
                (prey.x, prey.y),
                (target.x, target.y),
            )

            tracker.start(target_id, distance)
            self.targets[prey_id] = target

    def after_movement(self, evaluated_prey=None):
        """Registra el movimiento, incluso de presas recién eliminadas."""

        if evaluated_prey is None:
            evaluated_prey = self.environment.prey

        for prey in evaluated_prey:
            prey_id = id(prey)
            tracker = self.trackers.get(prey_id)
            target = self.targets.get(prey_id)

            if tracker is None or target is None:
                continue

            if tracker.is_closed:
                continue

            # Dejar la clasificación de desapariciones
            # para after_interactions().
            if not any(plant is target for plant in self.environment.plants):
                continue

            distance = math.dist(
                (prey.x, prey.y),
                (target.x, target.y),
            )

            tracker.observe(id(target), distance)

    def after_interactions(self, evaluated_prey):
        """Clasifica consumos, desapariciones y muertes."""

        active_plants = {id(plant) for plant in self.environment.plants}

        for prey in evaluated_prey:
            tracker = self.trackers.get(id(prey))

            if tracker is None or tracker.target_id is None or tracker.is_closed:
                continue

            target_id = tracker.target_id

            reference = prey.food_progress_references.get(target_id)

            if reference is not None and reference.status == "consumed":
                tracker.close("self_consumed")

            elif target_id not in active_plants:
                if self._consumed_by_other(target_id, prey, evaluated_prey):
                    tracker.close("other_consumed")
                else:
                    tracker.close("disappeared")

            elif not prey.alive:
                tracker.close("prey_dead")

    def finish(self):
        """Cierra las referencias restantes al terminar el episodio."""

        for tracker in self.trackers.values():
            if tracker.target_id is None or tracker.is_closed:
                continue

            tracker.close("episode_end")

    def results(self):
        """Devuelve los resultados individuales del diagnóstico."""

        return {
            prey_id: {
                "score": tracker.score,
                "close_reason": tracker.close_reason,
                "target_id": tracker.target_id,
            }
            for prey_id, tracker in self.trackers.items()
        }

    @staticmethod
    def _consumed_by_other(target_id, prey, evaluated_prey):
        """Comprueba si otra presa consumió la planta objetivo."""
        for other in evaluated_prey:
            if other is prey:
                continue

            reference = other.food_progress_references.get(target_id)

            if reference is not None and reference.status == "consumed":
                return True

        return False
