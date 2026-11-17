
from neuroecosistema.agents.predator import Predator
from neuroecosistema.agents.prey import Prey
from neuroecosistema.simulation.environment import Environment, Plant


def test_prey_consumes_plant_and_recovers_energy():
    environment = Environment(500, 400, 0, seed=42)

    prey = Prey(x=100, y=100, energy=50)
    plant = Plant(x=100, y=100, energy=20)

    environment.prey.append(prey)
    environment.plants.append(plant)

    environment.resolve_interactions()

    assert prey.energy == 70
    assert len(environment.plants) == 0
    assert environment.total_food_consumed == 1


def test_plant_can_only_be_consumed_once():
    environment = Environment(500, 400, 0, seed=42)

    first_prey = Prey(x=100, y=100, energy=50)
    second_prey = Prey(x=100, y=100, energy=50)

    environment.prey.extend([first_prey, second_prey])
    environment.plants.append(Plant(x=100, y=100, energy=20))

    environment.resolve_interactions()

    assert first_prey.energy == 70
    assert second_prey.energy == 50
    assert environment.total_food_consumed == 1


def test_predator_captures_prey_and_recovers_energy():
    environment = Environment(500, 400, 0, seed=42)

    prey = Prey(x=100, y=100)
    predator = Predator(x=100, y=100, energy=50)

    environment.prey.append(prey)
    environment.predators.append(predator)

    environment.resolve_interactions(capture_reward=30)

    assert len(environment.prey) == 0
    assert predator.energy == 80
    assert not prey.alive
    assert environment.total_captures == 1


def test_predator_cannot_capture_same_prey_twice():
    environment = Environment(500, 400, 0, seed=42)

    prey = Prey(x=100, y=100)
    first_predator = Predator(x=100, y=100)
    second_predator = Predator(x=100, y=100)

    environment.prey.append(prey)
    environment.predators.extend([first_predator, second_predator])

    environment.resolve_interactions()

    assert len(environment.prey) == 0
    assert environment.total_captures == 1
