"""Controles de configuración y consulta de resultados para Pygame."""

import csv
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pygame

from neuroecosistema.simulation.metrics import SimulationHistory

BACKGROUND = (13, 22, 33)
PANEL = (22, 35, 49)
PANEL_LIGHT = (31, 48, 64)
TEXT = (226, 236, 244)
MUTED = (150, 169, 184)
ACCENT = (42, 190, 170)
PREY_COLOR = (245, 196, 81)
PREDATOR_COLOR = (239, 112, 111)
PLANT_COLOR = (104, 205, 132)
GRID = (53, 70, 84)
DISABLED = (81, 94, 105)


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    action: str
    enabled: bool = True


def draw_button(
    screen: pygame.Surface,
    font: pygame.font.Font,
    button: Button,
    mouse_position: tuple[int, int],
) -> None:
    hovered = button.enabled and button.rect.collidepoint(mouse_position)
    color = (
        DISABLED if not button.enabled
        else ACCENT if hovered
        else PANEL_LIGHT
    )
    pygame.draw.rect(screen, color, button.rect, border_radius=8)
    pygame.draw.rect(
        screen,
        (76, 100, 117),
        button.rect,
        width=1,
        border_radius=8,
    )
    label = font.render(button.label, True, TEXT)
    screen.blit(label, label.get_rect(center=button.rect.center))


class Slider:
    def __init__(
        self,
        label: str,
        minimum: int,
        maximum: int,
        value: int,
        x: int,
        y: int,
        width: int = 390,
    ) -> None:
        self.label = label
        self.minimum = minimum
        self.maximum = maximum
        self.value = max(minimum, min(maximum, value))
        self.rect = pygame.Rect(x, y, width, 54)
        self.dragging = False

    def _set_from_x(self, x: int) -> None:
        track_left = self.rect.x + 12
        track_right = self.rect.right - 12
        ratio = (x - track_left) / max(1, track_right - track_left)
        ratio = max(0.0, min(1.0, ratio))
        self.value = round(
            self.minimum + ratio * (self.maximum - self.minimum)
        )

    def handle_event(self, event: pygame.event.Event) -> None:
        track = pygame.Rect(
            self.rect.x,
            self.rect.y + 24,
            self.rect.width,
            27,
        )
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if track.collidepoint(event.pos):
                self.dragging = True
                self._set_from_x(event.pos[0])
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False

    def draw(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
        enabled: bool = True,
    ) -> None:
        color = TEXT if enabled else MUTED
        screen.blit(font.render(self.label, True, color), (self.rect.x, self.rect.y))
        value_text = small_font.render(str(self.value), True, ACCENT if enabled else MUTED)
        screen.blit(value_text, value_text.get_rect(right=self.rect.right, top=self.rect.y + 2))
        left = self.rect.x + 12
        right = self.rect.right - 12
        center_y = self.rect.y + 39
        rail_color = PANEL_LIGHT if enabled else DISABLED
        pygame.draw.line(screen, rail_color, (left, center_y), (right, center_y), 6)
        ratio = (
            (self.value - self.minimum) / (self.maximum - self.minimum)
            if self.maximum > self.minimum
            else 0
        )
        knob_x = round(left + ratio * (right - left))
        pygame.draw.line(screen, ACCENT if enabled else DISABLED, (left, center_y), (knob_x, center_y), 6)
        pygame.draw.circle(screen, TEXT if enabled else MUTED, (knob_x, center_y), 9)


class ConfigScreen:
    """Pantalla de parámetros; no crea ni muta el entorno en ejecución."""

    def __init__(
        self,
        config: dict,
        stabilizer_default: float | None = None,
        world_height: int | None = None,
    ) -> None:
        simulation = config["simulation"]
        environment = config["environment"]
        population = config["population"]
        self.width = simulation["width"]
        self.height = simulation["height"]
        self.world_height = world_height or self.height
        self.max_steps = simulation["max_steps"]
        initial_food = environment["initial_food"]
        food_respawn = environment.get("food_respawn", 0)
        self.sliders = {
            "prey": Slider("Presas iniciales", 0, 80, population["prey"], 55, 152),
            "predators": Slider("Depredadores iniciales", 0, 20, population["predators"], 55, 220),
            "initial_food": Slider("Plantas iniciales", 0, 200, initial_food, 55, 288),
            "food_respawn": Slider("Intervalo de regeneración (pasos)", 1, 300, max(1, food_respawn), 55, 395),
            "max_food": Slider("Límite máximo de plantas", 0, 300, max(initial_food, environment.get("max_food", initial_food)), 55, 463),
            "turn_degrees": Slider("Giro máximo de depredadores (grados)", 1, 180, round(stabilizer_default or 45), 525, 222),
        }
        self.food_respawn_enabled = food_respawn > 0
        self.stabilizer_enabled = stabilizer_default is not None
        self.seed_text = str(environment.get("seed", 42))
        self.seed_active = False
        self.error = ""
        self.buttons = [
            Button(pygame.Rect(700, self.height - 76, 260, 54), "Iniciar simulación", "start"),
            Button(pygame.Rect(430, self.height - 76, 250, 54), "Ver resultados", "results"),
            Button(pygame.Rect(40, self.height - 76, 330, 54), "Volver a la simulación", "resume"),
        ]
        self.buttons[1].enabled = False
        self.buttons[2].enabled = False
        self.seed_rect = pygame.Rect(55, 545, 210, 38)
        self.new_seed_button = Button(pygame.Rect(277, 545, 168, 38), "Nueva semilla", "new_seed")
        self.respawn_rect = pygame.Rect(55, 349, 20, 20)
        self.stabilizer_rect = pygame.Rect(525, 160, 20, 20)

    def set_settings(self, settings: dict) -> None:
        for key in ("prey", "predators", "initial_food", "food_respawn", "max_food"):
            slider = self.sliders[key]
            slider.value = max(slider.minimum, min(slider.maximum, settings[key]))
        self.sliders["max_food"].minimum = self.sliders["initial_food"].value
        self.sliders["max_food"].value = max(
            self.sliders["max_food"].minimum,
            settings["max_food"],
        )
        self.sliders["food_respawn"].value = max(
            1,
            settings["food_respawn"],
        )
        self.food_respawn_enabled = settings["food_respawn"] > 0
        self.seed_text = str(settings["seed"])
        self.stabilizer_enabled = settings["predator_max_turn_degrees"] is not None
        if self.stabilizer_enabled:
            self.sliders["turn_degrees"].value = settings[
                "predator_max_turn_degrees"
            ]

    def handle_event(
        self,
        event: pygame.event.Event,
        has_history: bool,
        can_resume: bool,
    ) -> str | None:
        self.buttons[1].enabled = has_history
        self.buttons[2].enabled = can_resume
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.seed_active = self.seed_rect.collidepoint(event.pos)
            if self.respawn_rect.inflate(12, 12).collidepoint(event.pos):
                self.food_respawn_enabled = not self.food_respawn_enabled
            elif self.stabilizer_rect.inflate(12, 12).collidepoint(event.pos):
                self.stabilizer_enabled = not self.stabilizer_enabled
            if self.new_seed_button.rect.collidepoint(event.pos):
                self.seed_text = str(secrets.randbelow(2**32))
                self.seed_active = False
            for button in self.buttons:
                if button.enabled and button.rect.collidepoint(event.pos):
                    return button.action
        elif event.type == pygame.KEYDOWN and self.seed_active:
            if event.key == pygame.K_BACKSPACE:
                self.seed_text = self.seed_text[:-1]
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.seed_active = False
            elif event.unicode.isdecimal() and len(self.seed_text) < 10:
                self.seed_text += event.unicode

        slider_keys = ["prey", "predators", "initial_food", "max_food"]
        if self.food_respawn_enabled:
            slider_keys.append("food_respawn")
        for key in slider_keys:
            self.sliders[key].handle_event(event)
        if self.stabilizer_enabled:
            self.sliders["turn_degrees"].handle_event(event)
        self.sliders["max_food"].minimum = self.sliders["initial_food"].value
        self.sliders["max_food"].value = max(
            self.sliders["max_food"].minimum,
            self.sliders["max_food"].value,
        )
        return None

    def get_settings(self) -> dict:
        try:
            seed = int(self.seed_text)
        except ValueError as error:
            raise ValueError("Ingresa una semilla numérica.") from error
        if not 0 <= seed < 2**32:
            raise ValueError("La semilla debe estar entre 0 y 4294967295.")
        initial_food = self.sliders["initial_food"].value
        return {
            "prey": self.sliders["prey"].value,
            "predators": self.sliders["predators"].value,
            "initial_food": initial_food,
            "food_respawn": (
                self.sliders["food_respawn"].value
                if self.food_respawn_enabled
                else 0
            ),
            "max_food": max(initial_food, self.sliders["max_food"].value),
            "seed": seed,
            "predator_max_turn_degrees": (
                self.sliders["turn_degrees"].value
                if self.stabilizer_enabled
                else None
            ),
        }

    def draw(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
        has_history: bool = False,
        can_resume: bool = False,
    ) -> None:
        self.buttons[1].enabled = has_history
        self.buttons[2].enabled = can_resume
        screen.fill(BACKGROUND)
        screen.blit(font.render("NEUROECOSISTEMA", True, TEXT), (40, 28))
        screen.blit(small_font.render("Laboratorio de simulación evolutiva", True, MUTED), (40, 61))
        pygame.draw.rect(screen, PANEL, pygame.Rect(30, 104, 450, 484), border_radius=12)
        pygame.draw.rect(screen, PANEL, pygame.Rect(500, 104, 470, 484), border_radius=12)
        screen.blit(font.render("Ecosistema", True, TEXT), (55, 116))
        screen.blit(font.render("Control neuronal", True, TEXT), (525, 116))
        for key in ("prey", "predators", "initial_food"):
            self.sliders[key].draw(screen, small_font, small_font)
        self._draw_toggle(
            screen, small_font, self.respawn_rect,
            "Regeneración automática de plantas", self.food_respawn_enabled,
        )
        self.sliders["food_respawn"].draw(
            screen, small_font, small_font, self.food_respawn_enabled,
        )
        self.sliders["max_food"].draw(screen, small_font, small_font)
        screen.blit(small_font.render("Semilla aleatoria", True, TEXT), (55, 520))
        pygame.draw.rect(
            screen,
            PANEL_LIGHT if not self.seed_active else (40, 69, 84),
            self.seed_rect,
            border_radius=6,
        )
        seed_surface = small_font.render(self.seed_text or " ", True, TEXT)
        screen.blit(seed_surface, (self.seed_rect.x + 10, self.seed_rect.y + 9))
        draw_button(screen, small_font, self.new_seed_button, pygame.mouse.get_pos())

        self._draw_toggle(
            screen, small_font, self.stabilizer_rect,
            "Estabilización experimental", self.stabilizer_enabled,
        )
        self.sliders["turn_degrees"].draw(
            screen, small_font, small_font, self.stabilizer_enabled,
        )
        info = (
            "El estabilizador solo se aplica a depredadores",
            "con red neuronal. No modifica sus pesos ni afecta",
            "a presas, estrategias heurísticas o aleatorias.",
            "",
            f"Resolución del mundo: {self.width} × {self.world_height}",
            f"Límite de simulación: {self.max_steps} pasos",
            "La misma semilla y configuración reproducen",
            "la secuencia aleatoria del entorno.",
        )
        for index, line in enumerate(info):
            screen.blit(small_font.render(line, True, MUTED), (525, 300 + index * 27))

        if self.error:
            screen.blit(small_font.render(self.error, True, PREDATOR_COLOR), (40, self.height - 106))
        mouse = pygame.mouse.get_pos()
        for button in self.buttons:
            draw_button(screen, small_font, button, mouse)

    @staticmethod
    def _draw_toggle(
        screen: pygame.Surface,
        font: pygame.font.Font,
        rect: pygame.Rect,
        label: str,
        checked: bool,
    ) -> None:
        pygame.draw.rect(screen, ACCENT if checked else PANEL_LIGHT, rect, border_radius=4)
        if checked:
            pygame.draw.line(screen, BACKGROUND, (rect.x + 4, rect.y + 10), (rect.x + 8, rect.y + 15), 2)
            pygame.draw.line(screen, BACKGROUND, (rect.x + 8, rect.y + 15), (rect.x + 17, rect.y + 4), 2)
        screen.blit(font.render(label, True, TEXT), (rect.right + 10, rect.y - 2))


class StatisticsScreen:
    """Representa series observadas y exporta la ejecución sin borrarla."""

    def __init__(self) -> None:
        self.selected = 0
        self.options: list[dict] = []
        self.buttons = [
            Button(pygame.Rect(35, 28, 145, 42), "Volver", "back"),
            Button(pygame.Rect(720, 28, 120, 42), "Exportar CSV", "export_csv"),
            Button(pygame.Rect(850, 28, 115, 42), "Exportar PNG", "export_png"),
            Button(pygame.Rect(370, 640, 105, 38), "Anterior", "previous"),
            Button(pygame.Rect(500, 640, 105, 38), "Siguiente", "next"),
        ]

    def handle_event(self, event: pygame.event.Event, option_count: int) -> str | None:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                self.selected = max(0, self.selected - 1)
            elif event.key == pygame.K_RIGHT:
                self.selected = min(option_count - 1, self.selected + 1)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for button in self.buttons:
                if button.rect.collidepoint(event.pos):
                    if button.action == "previous":
                        self.selected = max(0, self.selected - 1)
                    elif button.action == "next":
                        self.selected = min(option_count - 1, self.selected + 1)
                    else:
                        return button.action
        return None

    def draw(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
        history: SimulationHistory,
        training_histories: list[dict],
        status: str,
    ) -> None:
        self.options = self._chart_options(history, training_histories)
        self.selected = min(self.selected, len(self.options) - 1)
        screen.fill(BACKGROUND)
        mouse = pygame.mouse.get_pos()
        for button in self.buttons[:3]:
            draw_button(screen, small_font, button, mouse)
        title = self.options[self.selected]["title"]
        screen.blit(font.render(f"Resultados de la ejecución: {status}", True, TEXT), (35, 88))
        screen.blit(font.render(title, True, ACCENT), (35, 132))
        option_label = f"Gráfica {self.selected + 1}/{len(self.options)} — ← → o botones"
        screen.blit(small_font.render(option_label, True, MUTED), (35, 166))
        self._draw_chart(screen, self.options[self.selected], small_font)
        for button in self.buttons[3:]:
            draw_button(screen, small_font, button, mouse)
        screen.blit(
            small_font.render(
                f"Pasos observados: {len(history.rows)}   "
                f"Capturas: {history.rows[-1]['captures'] if history.rows else 0}   "
                f"Alimento: {history.rows[-1]['food_consumed'] if history.rows else 0}",
                True,
                MUTED,
            ),
            (35, 610),
        )

    @staticmethod
    def export_paths() -> tuple[Path, Path]:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory = Path("results") / "simulation_exports"
        return (
            directory / f"simulation_{stamp}.csv",
            directory / f"simulation_{stamp}.png",
        )

    def export_csv(
        self,
        path: str | Path,
        history: SimulationHistory,
        training_histories: list[dict],
    ) -> Path:
        options = self._chart_options(history, training_histories)
        option = options[min(self.selected, len(options) - 1)]
        rows = option["rows"]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    @staticmethod
    def _chart_options(history: SimulationHistory, training_histories: list[dict]) -> list[dict]:
        options = [
            {
                "title": "Población por paso",
                "unit": "Paso",
                "x_field": "step",
                "series": (
                    ("Presas", "prey", PREY_COLOR),
                    ("Depredadores", "predators", PREDATOR_COLOR),
                ),
                "rows": history.rows,
            },
            {
                "title": "Plantas y alimento por paso",
                "unit": "Paso",
                "x_field": "step",
                "series": (
                    ("Plantas disponibles", "plants", PLANT_COLOR),
                    ("Alimento consumido", "food_consumed", ACCENT),
                ),
                "rows": history.rows,
            },
            {
                "title": "Capturas por paso",
                "unit": "Paso",
                "x_field": "step",
                "series": (("Capturas acumuladas", "captures", PREDATOR_COLOR),),
                "rows": history.rows,
            },
            {
                "title": "Energía promedio por paso",
                "unit": "Paso",
                "x_field": "step",
                "series": (("Energía promedio", "average_energy", PREY_COLOR),),
                "rows": history.rows,
            },
        ]
        for training in training_histories:
            options.append({
                "title": f"Entrenamiento: {training['label']}",
                "unit": training["unit"],
                "x_field": training["x_field"],
                "series": tuple(
                    (
                        "Mejor fitness" if metric == "best_fitness"
                        else "Fitness promedio" if metric == "mean_fitness"
                        else "Entrenamiento" if metric == "train_loss"
                        else "Validación",
                        metric,
                        PREY_COLOR if index == 0 else ACCENT,
                    )
                    for index, metric in enumerate(training["metrics"])
                ),
                "rows": training["rows"],
            })
        return options

    @staticmethod
    def _draw_chart(
        screen: pygame.Surface,
        option: dict,
        font: pygame.font.Font,
    ) -> None:
        plot = pygame.Rect(100, 220, 820, 345)
        pygame.draw.rect(screen, PANEL, plot, border_radius=8)
        graph = plot.inflate(-68, -54)
        for index in range(5):
            y = graph.y + round(index * graph.height / 4)
            pygame.draw.line(screen, GRID, (graph.x, y), (graph.right, y), 1)
        pygame.draw.line(screen, MUTED, graph.bottomleft, graph.bottomright, 1)
        pygame.draw.line(screen, MUTED, graph.topleft, graph.bottomleft, 1)
        rows = option["rows"]
        if not rows:
            label = font.render("No hay datos disponibles para esta gráfica.", True, MUTED)
            screen.blit(label, label.get_rect(center=plot.center))
            return
        x_field = option["x_field"]
        x_values = [float(row[x_field]) for row in rows]
        x_min, x_max = min(x_values), max(x_values)
        if x_max == x_min:
            x_max = x_min + 1
        series_values = [
            (label, key, color, [float(row[key]) for row in rows if key in row])
            for label, key, color in option["series"]
        ]
        all_values = [value for _, _, _, values in series_values for value in values]
        if not all_values:
            return
        y_min = min(0.0, min(all_values))
        y_max = max(all_values)
        if y_max == y_min:
            y_max = y_min + 1
        for label, key, color, values in series_values:
            matching_rows = [row for row in rows if key in row]
            points = []
            for row, value in zip(matching_rows, values):
                x = graph.x + round((float(row[x_field]) - x_min) / (x_max - x_min) * graph.width)
                y = graph.bottom - round((value - y_min) / (y_max - y_min) * graph.height)
                points.append((x, y))
            if len(points) > 1:
                pygame.draw.lines(screen, color, False, points, 3)
            for point in points:
                pygame.draw.circle(screen, color, point, 3)
            legend = font.render(label, True, color)
            screen.blit(legend, (graph.x + 14, plot.y + 12 + len(series_values) * 22 * option["series"].index((label, key, color))))
        screen.blit(font.render(option["unit"], True, TEXT), (graph.centerx - 30, plot.bottom - 3))
        screen.blit(font.render(f"{x_min:g}", True, MUTED), (graph.left, graph.bottom + 5))
        screen.blit(font.render(f"{x_max:g}", True, MUTED), (graph.right - 45, graph.bottom + 5))
        screen.blit(font.render(f"{y_max:g}", True, MUTED), (plot.x + 8, graph.top - 8))
        screen.blit(font.render(f"{y_min:g}", True, MUTED), (plot.x + 8, graph.bottom - 10))
