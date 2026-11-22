import csv

from neuroecosistema.simulation.environment import Environment
from neuroecosistema.simulation.metrics import (
    SimulationHistory,
    load_training_histories,
)
from neuroecosistema.visualization.interface import StatisticsScreen


def test_history_records_observed_step_metrics_and_exports(tmp_path):
    environment = Environment(
        width=500,
        height=400,
        initial_food=2,
        seed=42,
    )
    environment.generate_agents(prey_quantity=2, predator_quantity=1)
    history = SimulationHistory()
    history.record(environment)

    environment.step_count = 1
    environment.total_captures = 1
    environment.total_food_consumed = 2
    environment.prey[0].energy = 60
    environment.prey[1].energy = 80
    environment.predators[0].energy = 70
    history.record(environment)

    assert history.rows[0] == {
        "step": 0,
        "prey": 2,
        "predators": 1,
        "plants": 2,
        "captures": 0,
        "food_consumed": 0,
        "average_energy": 100.0,
    }
    assert history.rows[1]["average_energy"] == 70.0
    assert history.rows[1]["captures"] == 1
    assert history.rows[1]["food_consumed"] == 2

    path = history.export_csv(tmp_path / "run.csv")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert rows[-1]["step"] == "1"
    assert rows[-1]["average_energy"] == "70.0"


def test_history_is_retained_until_explicitly_cleared():
    history = SimulationHistory()
    environment = Environment(500, 400, initial_food=0, seed=3)
    history.record(environment)

    Environment(500, 400, initial_food=0, seed=3)
    assert len(history.rows) == 1

    history.clear()
    assert history.rows == []


def test_training_histories_preserve_generation_and_epoch_units(tmp_path):
    (tmp_path / "history_prey.csv").write_text(
        "generation,best_fitness,mean_fitness\n1,0.8,0.6\n",
        encoding="utf-8",
    )
    (tmp_path / "imitation_history.csv").write_text(
        "species,phase,epoch,train_loss,validation_loss\n"
        "predator,all,1,0.3,0.4\n",
        encoding="utf-8",
    )

    histories = load_training_histories(tmp_path)

    assert [history["unit"] for history in histories] == [
        "Generación",
        "Época",
    ]
    assert histories[0]["rows"][0]["best_fitness"] == 0.8
    assert histories[1]["rows"][0]["validation_loss"] == 0.4


def test_statistics_exports_the_selected_training_chart_data(tmp_path):
    history = SimulationHistory()
    history.record(Environment(500, 400, initial_food=0, seed=1))
    training = [{
        "label": "Presas (generación)",
        "unit": "Generación",
        "x_field": "generation",
        "metrics": ("best_fitness", "mean_fitness"),
        "rows": [{
            "generation": 1.0,
            "best_fitness": 0.8,
            "mean_fitness": 0.6,
        }],
    }]
    screen = StatisticsScreen()
    screen.selected = 4

    path = screen.export_csv(
        tmp_path / "training.csv",
        history,
        training,
    )
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert rows == [{
        "generation": "1.0",
        "best_fitness": "0.8",
        "mean_fitness": "0.6",
    }]
