import argparse
from pathlib import Path

import pygame
import yaml

from neuroecosistema.neural_network.model_loader import (
    load_trained_network,
)
from neuroecosistema.simulation.environment import Environment
from neuroecosistema.simulation.metrics import (
    SimulationHistory,
    load_training_histories,
)
from neuroecosistema.visualization.interface import (
    ACCENT,
    BACKGROUND,
    PANEL,
    PREY_COLOR,
    TEXT,
    Button,
    ConfigScreen,
    StatisticsScreen,
    draw_button,
)
from neuroecosistema.visualization.renderer import Renderer

TOOLBAR_HEIGHT = 64


def load_configuration() -> tuple[dict, dict]:
    """Carga la configuración del ecosistema y de las redes neuronales."""

    project_root = Path(__file__).resolve().parents[2]

    simulation_path = project_root / "config" / "simulation.yaml"
    neural_path = project_root / "config" / "neural_network.yaml"

    with open(simulation_path, encoding="utf-8") as file:
        simulation_config = yaml.safe_load(file)

    with open(neural_path, encoding="utf-8") as file:
        neural_config = yaml.safe_load(file)["neural_network"]

    return simulation_config, neural_config


def assign_trained_models(
    environment,
    neural_config,
    model_dir="results",
):
    """Asigna una copia del modelo entrenado a cada agente."""

    model_dir = Path(model_dir)
    models = (
        ("prey", environment.prey, environment.prey_networks),
        ("predator", environment.predators, environment.predator_networks),
    )

    for species, agents, networks in models:
        model_path = model_dir / f"best_{species}.npz"
        trained_network = load_trained_network(model_path, neural_config)
        genome = trained_network.get_genome()

        for agent in agents:
            networks[id(agent)].set_genome(genome)


def create_environment(
    config: dict,
    neural_config: dict,
    settings: dict,
    model_dir: str | Path = "results",
    use_trained: bool = False,
    world_height: int | None = None,
) -> Environment:
    """Crea desde cero una ejecución reproducible y sus redes."""

    simulation = config["simulation"]
    energy = config["energy"]
    environment = Environment(
        width=simulation["width"],
        height=world_height or simulation["height"],
        initial_food=settings["initial_food"],
        seed=settings["seed"],
        food_energy=energy["food_reward"],
        food_respawn=settings["food_respawn"],
        max_food=settings["max_food"],
        predator_max_turn_degrees=settings["predator_max_turn_degrees"],
    )
    environment.generate_agents(
        prey_quantity=settings["prey"],
        predator_quantity=settings["predators"],
    )
    environment.initialize_neural_networks(neural_config)

    if use_trained:
        assign_trained_models(environment, neural_config, model_dir)

    return environment


class SimulationApp:
    def __init__(
        self,
        config: dict,
        neural_config: dict,
        model_dir: str | Path,
        use_trained: bool,
    ) -> None:
        self.config = config
        self.neural_config = neural_config
        self.model_dir = Path(model_dir)
        self.use_trained = use_trained
        self.simulation = config["simulation"]
        self.energy = config["energy"]
        self.screen = pygame.display.set_mode(
            (self.simulation["width"], self.simulation["height"])
        )
        pygame.display.set_caption("NeuroEcosistema — laboratorio de IA")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)
        self.small_font = pygame.font.Font(None, 23)
        self.world_height = max(1, self.simulation["height"] - TOOLBAR_HEIGHT)
        self.renderer = Renderer(
            self.screen,
            pygame.Rect(
                0,
                TOOLBAR_HEIGHT,
                self.simulation["width"],
                self.world_height,
            ),
        )
        self.config_screen = ConfigScreen(
            config,
        )
        self.statistics_screen = StatisticsScreen()
        self.training_histories = load_training_histories(self.model_dir)
        self.state = "config"
        self.return_state = "paused"
        self.environment: Environment | None = None
        self.history = SimulationHistory()
        self.settings: dict | None = None
        self.status_message = ""
        self.prompt_buttons = [
            Button(pygame.Rect(220, 320, 560, 50), "Reiniciar con esta configuración", "restart_same"),
            Button(pygame.Rect(220, 384, 560, 50), "Modificar configuración", "restart_configure"),
            Button(pygame.Rect(220, 448, 560, 50), "Cancelar", "restart_cancel"),
        ]
        self.result_buttons = [
            Button(pygame.Rect(180, 445, 190, 52), "Ver gráficas", "stats"),
            Button(pygame.Rect(405, 445, 230, 52), "Reiniciar", "restart"),
            Button(pygame.Rect(670, 445, 180, 52), "Menú", "menu"),
        ]

    def run(self) -> None:
        try:
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        continue
                    running = self._handle_event(event)

                if self.state == "running":
                    self._update_simulation()

                self._draw()
                pygame.display.flip()
                self.clock.tick(self.simulation["fps"])
        finally:
            pygame.quit()

    def _handle_event(self, event: pygame.event.Event) -> bool:
        if self.state == "config":
            action = self.config_screen.handle_event(
                event,
                has_history=bool(self.history.rows),
                can_resume=self.environment is not None
                and self.state != "finished"
                and self.return_state == "paused",
            )
            if action == "start":
                try:
                    self._start_simulation(self.config_screen.get_settings())
                except ValueError as error:
                    self.config_screen.error = str(error)
            elif action == "results":
                if self.return_state != "finished":
                    self.return_state = "paused"
                self.state = "stats"
            elif action == "resume":
                self.state = "paused"
            return True

        if self.state == "stats":
            action = self.statistics_screen.handle_event(
                event,
                4 + len(self.training_histories),
            )
            if action == "back":
                self.state = self.return_state
            elif action == "export_csv":
                csv_path, _ = self.statistics_screen.export_paths()
                self.statistics_screen.export_csv(
                    csv_path,
                    self.history,
                    self.training_histories,
                )
                self.status_message = f"CSV guardado: {csv_path}"
                print(self.status_message)
            elif action == "export_png":
                _, png_path = self.statistics_screen.export_paths()
                png_path.parent.mkdir(parents=True, exist_ok=True)
                self.statistics_screen.draw(
                    self.screen,
                    self.font,
                    self.small_font,
                    self.history,
                    self.training_histories,
                    self._status_label(),
                )
                pygame.image.save(self.screen, png_path)
                self.status_message = f"Gráfica guardada: {png_path}"
                print(self.status_message)
            return True

        if self.state == "restart_prompt":
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in self.prompt_buttons:
                    if button.rect.collidepoint(event.pos):
                        if button.action == "restart_same" and self.settings is not None:
                            self._start_simulation(self.settings.copy())
                        elif button.action == "restart_configure" and self.settings is not None:
                            self.config_screen.set_settings(self.settings)
                            if self.return_state == "running":
                                self.return_state = "paused"
                            self.state = "config"
                        elif button.action == "restart_cancel":
                            self.state = self.return_state
                        break
            return True

        if self.state == "finished":
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for button in self.result_buttons:
                    if button.rect.collidepoint(event.pos):
                        if button.action == "stats":
                            self.return_state = "finished"
                            self.state = "stats"
                        elif button.action == "restart":
                            self.return_state = "finished"
                            self.state = "restart_prompt"
                        elif button.action == "menu":
                            if self.settings is not None:
                                self.config_screen.set_settings(self.settings)
                            self.return_state = "finished"
                            self.state = "config"
                        break
            return True

        if self.state in ("running", "paused"):
            for button in self._toolbar_buttons():
                if (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and button.rect.collidepoint(event.pos)
                ):
                    self._handle_toolbar_action(button.action)
                    break
        return True

    def _handle_toolbar_action(self, action: str) -> None:
        if action == "pause":
            self.state = "paused" if self.state == "running" else "running"
        elif action == "restart":
            self.return_state = self.state
            self.state = "restart_prompt"
        elif action == "menu":
            if self.settings is not None:
                self.config_screen.set_settings(self.settings)
            self.return_state = "paused"
            self.state = "config"
        elif action == "stats":
            self.return_state = self.state
            self.state = "stats"
        elif action == "exit":
            pygame.event.post(pygame.event.Event(pygame.QUIT))

    def _start_simulation(self, settings: dict) -> None:
        self.environment = create_environment(
            self.config,
            self.neural_config,
            settings,
            model_dir=self.model_dir,
            use_trained=self.use_trained,
        )
        self.settings = settings.copy()
        self.history = SimulationHistory()
        self.history.record(self.environment)
        self.state = (
            "running"
            if self.environment.prey or self.environment.predators
            else "finished"
        )
        self.config_screen.error = ""

    def _update_simulation(self) -> None:
        if self.environment is None:
            raise RuntimeError("No hay una simulación activa.")
        if self.environment.step_count >= self.simulation["max_steps"]:
            self.state = "finished"
            return
        self.environment.update_agents(
            movement_cost=self.energy["movement_cost"],
        )
        self.environment.resolve_interactions(
            capture_reward=self.energy["capture_reward"],
        )
        self.environment.update_food()
        self.history.record(self.environment)
        if (
            (not self.environment.prey and not self.environment.predators)
            or self.environment.step_count >= self.simulation["max_steps"]
        ):
            self.state = "finished"

    def _draw(self) -> None:
        if self.state == "config":
            self.config_screen.draw(
                self.screen,
                self.font,
                self.small_font,
                has_history=bool(self.history.rows),
                can_resume=(
                    self.environment is not None
                    and self.return_state == "paused"
                    and self.environment.step_count < self.simulation["max_steps"]
                ),
            )
        elif self.state == "stats":
            self.statistics_screen.draw(
                self.screen,
                self.font,
                self.small_font,
                self.history,
                self.training_histories,
                self._status_label(),
            )
        elif self.state == "finished":
            self._draw_results()
        else:
            self._draw_simulation()
            if self.state == "restart_prompt":
                self._draw_restart_prompt()

    def _draw_simulation(self) -> None:
        if self.environment is None:
            self.screen.fill(BACKGROUND)
            return
        self.renderer.draw_environment(self.environment)
        self.renderer.draw_agents(self.environment)
        pygame.draw.rect(self.screen, PANEL, pygame.Rect(0, 0, self.simulation["width"], TOOLBAR_HEIGHT))
        status_color = ACCENT if self.state == "running" else PREY_COLOR
        status = "SIMULACIÓN ACTIVA" if self.state == "running" else "PAUSADA"
        status_surface = self.small_font.render(status, True, status_color)
        self.screen.blit(status_surface, (18, 8))
        step_text = (
            f"Paso {self.environment.step_count}/{self.simulation['max_steps']}  "
            f"Presas: {len(self.environment.prey)}  "
            f"Depredadores: {len(self.environment.predators)}  "
            f"Plantas: {len(self.environment.plants)}"
        )
        if self.small_font.size(step_text)[0] > 390:
            step_text = (
                f"Paso {self.environment.step_count}  "
                f"P: {len(self.environment.prey)}  "
                f"D: {len(self.environment.predators)}  "
                f"Plantas: {len(self.environment.plants)}"
            )
        self.screen.blit(self.small_font.render(step_text, True, TEXT), (18, 35))
        for button in self._toolbar_buttons():
            draw_button(self.screen, self.small_font, button, pygame.mouse.get_pos())

    def _draw_restart_prompt(self) -> None:
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((5, 10, 17, 190))
        self.screen.blit(overlay, (0, 0))
        panel = pygame.Rect(170, 245, 660, 300)
        pygame.draw.rect(self.screen, PANEL, panel, border_radius=14)
        heading = self.font.render("Reiniciar simulación", True, TEXT)
        self.screen.blit(heading, heading.get_rect(center=(panel.centerx, 285)))
        for button in self.prompt_buttons:
            draw_button(self.screen, self.small_font, button, pygame.mouse.get_pos())

    def _draw_results(self) -> None:
        self.screen.fill(BACKGROUND)
        pygame.draw.rect(self.screen, PANEL, pygame.Rect(70, 105, 860, 495), border_radius=16)
        self.screen.blit(self.font.render("Ejecución finalizada", True, ACCENT), (110, 145))
        last = self.history.rows[-1] if self.history.rows else {}
        values = (
            ("Pasos", last.get("step", 0)),
            ("Presas restantes", last.get("prey", 0)),
            ("Depredadores restantes", last.get("predators", 0)),
            ("Plantas disponibles", last.get("plants", 0)),
            ("Capturas", last.get("captures", 0)),
            ("Alimento consumido", last.get("food_consumed", 0)),
        )
        for index, (label, value) in enumerate(values):
            column = index % 3
            row = index // 3
            x = 120 + column * 260
            y = 235 + row * 100
            screen_label = self.small_font.render(label, True, TEXT)
            screen_value = self.font.render(str(value), True, ACCENT)
            self.screen.blit(screen_label, (x, y))
            self.screen.blit(screen_value, (x, y + 30))
        for button in self.result_buttons:
            draw_button(self.screen, self.small_font, button, pygame.mouse.get_pos())

    def _toolbar_buttons(self) -> list[Button]:
        labels = (
            (
                "Reanudar"
                if self.state == "paused"
                or (self.state == "restart_prompt" and self.return_state == "paused")
                else "Pausar",
                "pause",
                116,
            ),
            ("Reiniciar", "restart", 104),
            ("Menú", "menu", 104),
            ("Estadísticas", "stats", 116),
            ("Salir", "exit", 88),
        )
        buttons = []
        x = self.simulation["width"] - sum(item[2] for item in labels) - 8 * (len(labels) - 1) - 14
        for label, action, width in labels:
            buttons.append(
                Button(pygame.Rect(x, 10, width, 43), label, action)
            )
            x += width + 8
        return buttons

    def _status_label(self) -> str:
        return {
            "running": "activa",
            "paused": "pausada",
            "finished": "finalizada",
        }.get(self.return_state, "finalizada")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulación visual de NeuroEcosistema"
    )
    parser.add_argument(
        "--trained",
        action="store_true",
        help="Utilizar los mejores modelos entrenados",
    )
    parser.add_argument(
        "--model-dir",
        default="results",
        help="Directorio de los modelos guardados",
    )
    args = parser.parse_args()
    pygame.init()
    try:
        config, neural_config = load_configuration()
        app = SimulationApp(
            config,
            neural_config,
            args.model_dir,
            args.trained,
        )
        app.run()
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
