
import math
from dataclasses import dataclass, field


@dataclass
class FoodProgressReference:
    plant_id: int
    plant: object = field(compare=False, repr=False)
    initial_distance: float
    current_distance: float
    net_progress: float = 0.0
    status: str = "active"
    credit: float = 0.0


@dataclass
class BaseAgent:
    """Clase base para los organismos del ecosistema."""

    x: float
    y: float
    energy: float = 100.0
    speed: float = 1.5
    radius: int = 6
    color: tuple[int, int, int] = (255, 255, 255)

    direction: float = 0.0
    direction_timer: int = 0
    direction_change_interval: int = 45

    # Métricas de comportamiento.
    steps_survived: int = 0
    food_eaten: int = 0
    captures: int = 0
    stagnation_steps: int = 0
    distance_traveled: float = 0.0
    steps_near_wall: int = 0
    food_approach_progress: float = 0.0
    
    # Seguimiento del progreso hacia una planta.
    food_net_progress: float = 0.0
    food_target_switches: int = 0
    food_progress_score: float = field(default=0.0, compare=False)
    food_progress_references: dict[int, FoodProgressReference] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    # Estado temporal del objetivo alimentario.
    food_target_id: int | None = field(
        default=None,
        compare=False,
    )
    food_target_initial_distance: float = 0.0
    food_target_min_distance: float = 0.0

    # Estado del mecanismo de recuperación de paredes.
    wall_escape_active: bool = False

    # Regiones exploradas por el agente.
    visited_regions: set[tuple[int, int]] = field(
        default_factory=set,
        compare=False,
    )

    alive: bool = True

    def synchronize_food_progress_references(self, plants):
        """Track active plant identities without rewarding their appearance."""

        plants = tuple(plants)
        active_ids = {id(plant) for plant in plants}

        for reference in self.food_progress_references.values():
            if (
                reference.status == "active"
                and reference.plant_id not in active_ids
            ):
                reference.status = "discarded"

        for plant in plants:
            plant_id = id(plant)
            if plant_id in self.food_progress_references:
                continue

            distance = math.dist(
                (self.x, self.y),
                (plant.x, plant.y),
            )
            self.food_progress_references[plant_id] = (
                FoodProgressReference(
                    plant_id=plant_id,
                    plant=plant,
                    initial_distance=distance,
                    current_distance=distance,
                )
            )

    def record_food_progress_movement(
        self,
        plant_snapshots,
        distances_before,
        map_diagonal,
    ):
        """Accumulate signed progress for each plant during real movement."""

        if map_diagonal <= 0:
            raise ValueError("La diagonal del mapa debe ser positiva.")

        for plant, plant_x, plant_y in plant_snapshots:
            reference = self.food_progress_references[id(plant)]
            if reference.status != "active":
                continue

            distance_after = math.dist(
                (self.x, self.y),
                (plant_x, plant_y),
            )
            distance_before = distances_before[id(plant)]
            reference.net_progress += (
                distance_before - distance_after
            ) / map_diagonal
            reference.current_distance = distance_after

    def record_food_plant_consumed(
        self,
        plant,
        consumption_distance,
        map_diagonal,
    ):
        """Finalize one progress credit for this prey's own consumption."""

        if map_diagonal <= 0:
            raise ValueError("La diagonal del mapa debe ser positiva.")

        plant_id = id(plant)
        reference = self.food_progress_references.get(plant_id)
        if reference is None:
            reference = FoodProgressReference(
                plant_id=plant_id,
                plant=plant,
                initial_distance=consumption_distance,
                current_distance=consumption_distance,
            )
            self.food_progress_references[plant_id] = reference

        if reference.status != "active":
            return

        reference.current_distance = consumption_distance
        reference.status = "consumed"
        reference.credit = max(
            0.0,
            min(
                1.0,
                (
                    reference.initial_distance
                    - consumption_distance
                ) / map_diagonal,
            ),
        )

        consumed_credits = [
            item.credit
            for item in self.food_progress_references.values()
            if item.status == "consumed"
        ]
        self.food_progress_score = (
            sum(consumed_credits) / len(consumed_credits)
        )

    def register_food_target_progress(
        self,
        target_id,
        distance,
    ):
        """Registra el progreso y los cambios de objetivo alimentario."""

        if distance < 0:
            raise ValueError(
                "La distancia al objetivo no puede ser negativa."
            )

        if self.food_target_id != target_id:
            if self.food_target_id is not None:
                self.food_target_switches += 1

            self.food_target_id = target_id
            self.food_target_initial_distance = distance
            self.food_target_min_distance = distance
            return

        previous_minimum = self.food_target_min_distance

        if distance < previous_minimum:
            self.food_target_min_distance = distance

            self.food_net_progress += (
                previous_minimum - distance
            )

    def register_region(self, width, height, grid_size=5):
        """Registra la región espacial ocupada por el agente."""

        if width <= 0 or height <= 0:
            raise ValueError(
                "Las dimensiones deben ser positivas."
            )

        if grid_size <= 0:
            raise ValueError(
                "El tamaño de cuadrícula debe ser positivo."
            )

        column = min(
            grid_size - 1,
            max(0, int(self.x / width * grid_size)),
        )

        row = min(
            grid_size - 1,
            max(0, int(self.y / height * grid_size)),
        )

        self.visited_regions.add((column, row))

    def update(
        self,
        width,
        height,
        rng,
        movement_cost=0.1,
        neural_direction=None,
    ):
        """Actualiza el movimiento y la energía del agente."""

        if not self.alive:
            return

        # Movimiento controlado por la red neuronal.
        if neural_direction is not None:
            dx, dy = neural_direction
            magnitude = math.hypot(dx, dy)

            if magnitude > 1e-8:
                self.direction = math.atan2(dy, dx)

                movement_x = self.speed * dx / magnitude
                movement_y = self.speed * dy / magnitude
            else:
                movement_x = 0.0
                movement_y = 0.0

        # Movimiento aleatorio cuando no existe control neuronal.
        else:
            self.direction_timer += 1

            if self.direction_timer >= self.direction_change_interval:
                self.direction = rng.uniform(0, 2 * math.pi)
                self.direction_timer = 0

            movement_x = math.cos(self.direction) * self.speed
            movement_y = math.sin(self.direction) * self.speed

        # Guardar posición anterior.
        previous_x = self.x
        previous_y = self.y

        # Aplicar movimiento.
        self.x += movement_x
        self.y += movement_y

        # Mantener al agente dentro del entorno.
        self.x = max(
            self.radius,
            min(width - self.radius, self.x),
        )
        self.y = max(
            self.radius,
            min(height - self.radius, self.y),
        )

        # Calcular desplazamiento real.
        displacement = math.dist(
            (previous_x, previous_y),
            (self.x, self.y),
        )

        # Registrar distancia recorrida.
        self.distance_traveled += displacement

        # Registrar pasos de estancamiento acumulados.
        if displacement < 0.1:
            self.stagnation_steps += 1

        # Registrar pasos cerca de los límites del mapa.
        wall_margin = min(width, height) * 0.05

        distance_to_wall = min(
            self.x - self.radius,
            width - self.radius - self.x,
            self.y - self.radius,
            height - self.radius - self.y,
        )

        if distance_to_wall <= wall_margin:
            self.steps_near_wall += 1

        # Registrar regiones exploradas.
        self.register_region(width, height)

        # Actualizar supervivencia y energía.
        self.steps_survived += 1
        self.energy -= movement_cost

        if self.energy <= 0:
            self.energy = 0
            self.alive = False