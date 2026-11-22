
from dataclasses import dataclass

from neuroecosistema.analysis.food_orientation_metric import (
    normalized_net_progress,
)


@dataclass
class FoodOrientationTracker:
    """Sigue una única referencia alimentaria por presa."""

    map_diagonal: float
    target_id: int | None = None
    initial_distance: float | None = None
    final_distance: float | None = None
    close_reason: str | None = None

    @property
    def is_closed(self) -> bool:
        return self.close_reason is not None

    def start(self, plant_id: int, distance: float) -> None:
        """Selecciona la primera referencia del episodio."""
        if self.target_id is not None or self.is_closed:
            raise RuntimeError("La referencia ya fue iniciada.")

        normalized_net_progress(
            distance, distance, self.map_diagonal
        )

        self.target_id = plant_id
        self.initial_distance = distance
        self.final_distance = distance

    def observe(self, plant_id: int, distance: float) -> None:
        """Actualiza la distancia a la referencia original."""
        if self.is_closed or self.target_id is None:
            return

        if plant_id != self.target_id:
            return

        normalized_net_progress(
            self.initial_distance, distance, self.map_diagonal
        )
        self.final_distance = distance

    def close(self, reason: str) -> None:
        """Finaliza el seguimiento sin acumular segmentos."""
        if self.target_id is None:
            raise RuntimeError("No existe una referencia iniciada.")

        if self.is_closed:
            raise RuntimeError("La referencia ya fue cerrada.")

        if reason not in {
            "self_consumed",
            "other_consumed",
            "disappeared",
            "episode_end",
            "prey_dead",
        }:
            raise ValueError("Motivo de cierre no válido.")

        self.close_reason = reason

    @property
    def score(self) -> float:
        if self.initial_distance is None:
            return 0.0

        return normalized_net_progress(
            self.initial_distance,
            self.final_distance,
            self.map_diagonal,
        )
