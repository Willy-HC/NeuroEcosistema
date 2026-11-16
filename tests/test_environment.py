from neuroecosistema.simulation.environment import Environment


def test_environment_generates_correct_number_of_plants():
    environment = Environment(1000, 700, 100, seed=42)

    assert len(environment.plants) == 100


def test_environment_generates_plants_inside_boundaries():
    environment = Environment(1000, 700, 100, seed=42)

    for plant in environment.plants:
        assert 20 <= plant.x <= 980
        assert 20 <= plant.y <= 680


def test_same_seed_generates_same_distribution():
    environment_a = Environment(1000, 700, 100, seed=42)
    environment_b = Environment(1000, 700, 100, seed=42)

    positions_a = [(plant.x, plant.y) for plant in environment_a.plants]
    positions_b = [(plant.x, plant.y) for plant in environment_b.plants]

    assert positions_a == positions_b