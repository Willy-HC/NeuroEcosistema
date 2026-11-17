
import csv

import pytest

from neuroecosistema.analysis.fitness_analysis import (
    load_fitness_history,
    plot_fitness_history,
    plot_species_comparison,
    summarize_history,
)


def create_history():
    return [
        {
            "generation": 1,
            "best_fitness": 0.5,
            "mean_fitness": 0.3,
            "worst_fitness": 0.1,
            "std_fitness": 0.1,
            "best_so_far": 0.5,
        },
        {
            "generation": 2,
            "best_fitness": 0.7,
            "mean_fitness": 0.4,
            "worst_fitness": 0.2,
            "std_fitness": 0.1,
            "best_so_far": 0.7,
        },
    ]


def test_load_history(tmp_path):
    path = tmp_path / "history.csv"
    history = create_history()

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0]),
        )
        writer.writeheader()
        writer.writerows(history)

    loaded = load_fitness_history(path)

    assert loaded == history


def test_missing_history_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fitness_history(
            tmp_path / "missing.csv"
        )


def test_empty_history_raises_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text(
        "generation,best_fitness,mean_fitness,"
        "worst_fitness,std_fitness,best_so_far\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_fitness_history(path)


def test_plot_fitness_creates_png(tmp_path):
    path = tmp_path / "fitness.png"

    result = plot_fitness_history(
        create_history(),
        "prey",
        path,
    )

    assert result.exists()
    assert result.stat().st_size > 0


def test_comparison_creates_png(tmp_path):
    path = tmp_path / "comparison.png"

    result = plot_species_comparison(
        create_history(),
        create_history(),
        path,
    )

    assert result.exists()
    assert result.stat().st_size > 0


def test_summary_calculation():
    summary = summarize_history(
        create_history()
    )

    assert summary["generations"] == 2
    assert summary["best_fitness"] == 0.7
    assert summary["best_generation"] == 2
    assert summary["mean_change"] == pytest.approx(0.1)


def test_empty_summary_raises_error():
    with pytest.raises(ValueError):
        summarize_history([])