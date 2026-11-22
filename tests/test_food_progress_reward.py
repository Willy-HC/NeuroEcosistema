import math
import random

import pytest

from neuroecosistema.agents.base_agent import BaseAgent
from neuroecosistema.agents.prey import Prey
from neuroecosistema.evolution.fitness import (
    calculate_prey_fitness_components,
    validate_fitness_options,
)
from neuroecosistema.simulation.environment import Environment, Plant

MAP_WIDTH = 1000
MAP_HEIGHT = 1000
MAP_DIAGONAL = math.hypot(MAP_WIDTH, MAP_HEIGHT)
WEIGHT = 0.05


class FixedNetwork:
    def __init__(self, dx=1.0, dy=0.0):
        self.outputs = (dx, dy)

    def forward(self, _inputs):
        return self.outputs


def create_environment(
    prey,
    plants,
    *,
    food_respawn=0,
    max_food=0,
    seed=42,
):
    environment = Environment(
        width=MAP_WIDTH,
        height=MAP_HEIGHT,
        initial_food=0,
        food_respawn=food_respawn,
        max_food=max_food,
        seed=seed,
    )
    environment.prey = list(prey)
    environment.plants = list(plants)
    environment.prey_networks = {
        id(agent): FixedNetwork()
        for agent in environment.prey
    }
    return environment


def move(environment, prey, dx=1.0, dy=0.0):
    environment.prey_networks[id(prey)].outputs = (dx, dy)
    environment.update_agents(strategy="neural")


def reference(prey, plant):
    return prey.food_progress_references[id(plant)]


def assert_experimental_contribution(prey, expected):
    components = calculate_prey_fitness_components(
        prey,
        max_steps=500,
        fitness_mode="food_progress",
        food_progress_weight=WEIGHT,
    )
    assert components["food_progress"] == pytest.approx(
        WEIGHT * expected
    )
    return components


def test_case_a_approach_without_consumption_stays_pending():
    prey = Prey(x=500, y=500, speed=100)
    plant = Plant(x=800, y=500)
    environment = create_environment([prey], [plant])

    move(environment, prey)

    assert reference(prey, plant).net_progress == pytest.approx(
        100 / MAP_DIAGONAL
    )
    assert prey.food_progress_score == 0.0
    assert_experimental_contribution(prey, 0.0)


def test_case_b_unrelated_disappearance_preserves_other_reference():
    prey = Prey(x=500, y=500, speed=100)
    target = Plant(x=800, y=500)
    unrelated = Plant(x=500, y=900)
    environment = create_environment([prey], [target, unrelated])

    move(environment, prey)
    environment.plants.remove(unrelated)
    environment.synchronize_food_progress([prey])

    assert reference(prey, target).status == "active"
    assert reference(prey, target).net_progress == pytest.approx(
        100 / MAP_DIAGONAL
    )
    assert reference(prey, unrelated).status == "discarded"
    assert prey.food_progress_score == 0.0


def test_case_c_own_target_consumption_realizes_progress_credit():
    prey = Prey(x=500, y=500, speed=100)
    plant = Plant(x=611, y=500)
    environment = create_environment([prey], [plant])

    move(environment, prey)
    environment.resolve_interactions()

    expected_credit = 100 / MAP_DIAGONAL
    assert prey.food_eaten == 1
    assert reference(prey, plant).status == "consumed"
    assert reference(prey, plant).credit == pytest.approx(
        expected_credit
    )
    assert prey.food_progress_score == pytest.approx(expected_credit)
    components = assert_experimental_contribution(
        prey,
        expected_credit,
    )
    assert components["food"] == pytest.approx(0.1)


def test_case_d_target_change_does_not_transfer_or_sum_credit():
    prey = Prey(x=500, y=500, speed=100)
    old_target = Plant(x=800, y=500)
    environment = create_environment([prey], [old_target])

    move(environment, prey)
    environment.plants = [Plant(x=100, y=500)]
    new_target = environment.plants[0]
    environment.synchronize_food_progress([prey])
    move(environment, prey)

    assert reference(prey, old_target).status == "discarded"
    assert reference(prey, new_target).net_progress == pytest.approx(
        -100 / MAP_DIAGONAL
    )
    assert prey.food_progress_score == 0.0
    assert_experimental_contribution(prey, 0.0)


def test_case_e_away_then_back_cancels_for_same_plant():
    prey = Prey(x=500, y=500, speed=100)
    plant = Plant(x=800, y=500)
    environment = create_environment([prey], [plant])

    move(environment, prey, dx=-1.0)
    move(environment, prey, dx=1.0)

    assert reference(prey, plant).net_progress == pytest.approx(0.0)
    assert prey.food_progress_score == 0.0
    assert_experimental_contribution(prey, 0.0)


def test_case_f_target_alternation_without_consumption_earns_nothing():
    prey = Prey(x=500, y=500, speed=100)
    environment = create_environment([prey], [Plant(x=700, y=500)])

    for _ in range(8):
        environment.plants = [Plant(x=700, y=500)]
        move(environment, prey, dx=1.0)
        environment.plants = [Plant(x=450, y=500)]
        move(environment, prey, dx=-1.0)

    assert (prey.x, prey.y) == pytest.approx((500, 500))
    assert prey.food_eaten == 0
    assert prey.food_progress_score == 0.0
    assert all(
        item.status != "consumed"
        for item in prey.food_progress_references.values()
    )
    assert_experimental_contribution(prey, 0.0)


def _case_g_replenishment_from_other_agent(helper_consumes):
    prey = Prey(x=500, y=500, speed=100)
    helper = Prey(
        x=500 if helper_consumes else 100,
        y=900 if helper_consumes else 100,
        speed=0,
    )
    target = Plant(x=800, y=500)
    unrelated = Plant(x=500, y=900)
    environment = create_environment(
        [prey, helper],
        [target, unrelated],
        food_respawn=1,
        max_food=2,
        seed=123,
    )

    move(environment, prey)
    tracked_position = (prey.x, prey.y)
    environment.resolve_interactions()
    environment.update_food()
    environment.synchronize_food_progress([prey, helper])
    return (
        prey,
        target,
        unrelated,
        tracked_position,
        {id(plant) for plant in environment.plants},
    )


def test_case_g_unrelated_consumption_and_respawn_do_not_reset_target():
    control = _case_g_replenishment_from_other_agent(False)
    treatment = _case_g_replenishment_from_other_agent(True)

    (
        control_prey,
        control_target,
        control_unrelated,
        control_path,
        control_plant_ids,
    ) = control
    (
        treatment_prey,
        treatment_target,
        treatment_unrelated,
        treatment_path,
        treatment_plant_ids,
    ) = treatment

    assert control_path == treatment_path == (600, 500)
    assert control_plant_ids != treatment_plant_ids
    assert reference(control_prey, control_unrelated).status == "active"
    assert reference(treatment_prey, treatment_unrelated).status == "discarded"
    for prey, target in (
        (control_prey, control_target),
        (treatment_prey, treatment_target),
    ):
        assert reference(prey, target).status == "active"
        assert reference(prey, target).net_progress == pytest.approx(
            100 / MAP_DIAGONAL
        )
        assert prey.food_progress_score == 0.0


def test_case_h_three_successive_consumptions_are_averaged():
    prey = Prey(x=500, y=500, speed=100)
    environment = create_environment([prey], [])
    credits = []

    for _ in range(3):
        plant = Plant(x=prey.x + 111, y=prey.y)
        environment.plants = [plant]
        environment.synchronize_food_progress([prey])
        move(environment, prey)
        environment.resolve_interactions()
        credits.append(reference(prey, plant).credit)

    expected_credit = 100 / MAP_DIAGONAL
    assert prey.food_eaten == 3
    assert credits == pytest.approx([expected_credit] * 3)
    assert prey.food_progress_score == pytest.approx(expected_credit)
    assert_experimental_contribution(prey, expected_credit)


def test_case_i_oscillation_before_consumption_gets_only_net_credit():
    prey = Prey(x=500, y=500, speed=100)
    plant = Plant(x=700, y=500)
    environment = create_environment([prey], [plant])

    move(environment, prey, dx=1.0)
    move(environment, prey, dx=-1.0)
    prey.speed = 189
    move(environment, prey, dx=1.0)
    environment.resolve_interactions()

    expected_credit = 189 / MAP_DIAGONAL
    assert prey.food_eaten == 1
    assert reference(prey, plant).net_progress == pytest.approx(
        expected_credit
    )
    assert prey.food_progress_score == pytest.approx(expected_credit)
    assert_experimental_contribution(prey, expected_credit)


def test_case_j_new_nearer_plant_gets_its_own_reference():
    prey = Prey(x=500, y=500, speed=30)
    old_target = Plant(x=800, y=500)
    environment = create_environment([prey], [old_target])

    move(environment, prey)

    assert reference(prey, old_target).status == "active"
    assert reference(prey, old_target).net_progress == pytest.approx(
        30 / MAP_DIAGONAL
    )

    new_target = Plant(x=550, y=500)
    environment.plants.append(new_target)
    environment.synchronize_food_progress([prey])

    assert reference(prey, new_target).initial_distance == pytest.approx(20)
    assert reference(prey, new_target).net_progress == 0.0
    assert prey.food_progress_score == 0.0

    prey.speed = 9
    move(environment, prey)
    environment.resolve_interactions()

    assert reference(prey, new_target).net_progress == pytest.approx(
        9 / MAP_DIAGONAL
    )

    expected_credit = 9 / MAP_DIAGONAL
    assert reference(prey, new_target).credit == pytest.approx(
        expected_credit
    )
    assert prey.food_progress_score == pytest.approx(expected_credit)
    assert_experimental_contribution(prey, expected_credit)


def test_case_k_long_trajectory_without_consumption_gets_no_credit():
    prey = Prey(x=500, y=500, speed=100)
    plant = Plant(x=800, y=500)
    environment = create_environment([prey], [plant])

    move(environment, prey, dx=1.0)
    prey.speed = 400
    move(environment, prey, dx=-1.0)

    assert prey.distance_traveled == pytest.approx(500)
    assert reference(prey, plant).net_progress == pytest.approx(
        -300 / MAP_DIAGONAL
    )
    assert prey.food_progress_score == 0.0
    assert_experimental_contribution(prey, 0.0)


def _case_l_other_agent_replenishment(helper_consumes):
    prey = Prey(x=500, y=500, speed=100)
    helper = Prey(
        x=500 if helper_consumes else 100,
        y=900 if helper_consumes else 100,
        speed=0,
    )
    target = Plant(x=611, y=500)
    unrelated = Plant(x=500, y=900)
    environment = create_environment(
        [prey, helper],
        [target, unrelated],
        food_respawn=1,
        max_food=2,
        seed=321,
    )

    move(environment, prey)
    environment.resolve_interactions()
    environment.update_food()
    environment.synchronize_food_progress([prey, helper])
    return prey, target, len(environment.plants)


def test_case_l_identical_paths_keep_credit_across_unrelated_respawns():
    control = _case_l_other_agent_replenishment(False)
    treatment = _case_l_other_agent_replenishment(True)

    control_prey, control_target, control_food = control
    treatment_prey, treatment_target, treatment_food = treatment

    assert control_food != treatment_food
    assert control_prey.food_eaten == treatment_prey.food_eaten == 1
    assert reference(control_prey, control_target).credit == pytest.approx(
        100 / MAP_DIAGONAL
    )
    assert reference(
        treatment_prey,
        treatment_target,
    ).credit == pytest.approx(100 / MAP_DIAGONAL)
    assert control_prey.food_progress_score == pytest.approx(
        treatment_prey.food_progress_score
    )


def test_only_the_prey_that_consumes_gets_credit():
    first = Prey(x=500, y=500, speed=100)
    second = Prey(x=500, y=500, speed=100)
    plant = Plant(x=611, y=500)
    environment = create_environment([first, second], [plant])

    environment.update_agents(strategy="neural")
    environment.resolve_interactions()

    assert first.food_eaten == 1
    assert second.food_eaten == 0
    assert first.food_progress_score > 0
    assert second.food_progress_score == 0.0
    assert reference(first, plant).status == "consumed"
    assert reference(second, plant).status == "discarded"


def test_each_plant_identity_has_an_independent_lifecycle():
    prey = Prey(x=500, y=500, speed=100)
    target = Plant(x=611, y=500)
    unrelated = Plant(x=500, y=900)
    environment = create_environment([prey], [target, unrelated])

    move(environment, prey)
    environment.resolve_interactions()

    assert reference(prey, target).status == "consumed"
    assert reference(prey, target).credit > 0
    assert reference(prey, unrelated).status == "active"
    assert reference(prey, unrelated).credit == 0.0


def test_plant_appearance_starts_at_current_distance_without_reward():
    prey = Prey(x=500, y=500, speed=0)
    environment = create_environment([prey], [])

    environment.synchronize_food_progress([prey])
    environment.plants = [Plant(x=550, y=500)]
    plant = environment.plants[0]
    environment.synchronize_food_progress([prey])

    assert reference(prey, plant).initial_distance == pytest.approx(50)
    assert reference(prey, plant).net_progress == 0.0
    assert prey.food_progress_score == 0.0


def test_no_plants_and_new_episode_start_with_zero_score():
    prey = Prey(x=500, y=500)
    environment = create_environment([prey], [])
    environment.update_agents(strategy="neural")
    environment.resolve_interactions()
    environment.synchronize_food_progress([prey])

    next_episode_prey = Prey(x=500, y=500)
    assert prey.food_progress_score == 0.0
    assert next_episode_prey.food_progress_score == 0.0
    assert next_episode_prey.food_progress_references == {}


def test_baseline_fitness_is_exactly_unchanged():
    prey = Prey(x=100, y=100)
    prey.steps_survived = 500
    prey.food_eaten = 2
    prey.stagnation_steps = 100
    prey.visited_regions = {(0, 0), (1, 0), (2, 0)}
    prey.food_progress_score = 0.8

    original = calculate_prey_fitness_components(prey, 1000)
    explicit_baseline = calculate_prey_fitness_components(
        prey,
        1000,
        fitness_mode="baseline",
        food_progress_weight=WEIGHT,
    )

    assert explicit_baseline == original


def test_experimental_fitness_is_saturated_at_one():
    prey = Prey(x=100, y=100)
    prey.steps_survived = 1000
    prey.food_eaten = 5
    prey.visited_regions = {(x, y) for x in range(5) for y in range(5)}
    prey.food_progress_score = 1.0

    components = calculate_prey_fitness_components(
        prey,
        1000,
        fitness_mode="food_progress",
        food_progress_weight=0.5,
    )

    assert components["fitness"] == 1.0
    assert components["food_progress"] == 0.5


@pytest.mark.parametrize(
    "progress_score",
    [-0.01, 1.01, math.inf, math.nan],
)
def test_experimental_fitness_rejects_scores_outside_credit_range(
    progress_score,
):
    prey = Prey(x=100, y=100)
    prey.food_progress_score = progress_score

    with pytest.raises(ValueError):
        calculate_prey_fitness_components(
            prey,
            max_steps=500,
            fitness_mode="food_progress",
        )


@pytest.mark.parametrize(
    ("mode", "weight"),
    [
        ("unknown", WEIGHT),
        ("food_progress", -0.1),
        ("food_progress", math.inf),
        ("food_progress", True),
    ],
)
def test_invalid_fitness_options_are_rejected(mode, weight):
    with pytest.raises(ValueError):
        validate_fitness_options(mode, weight)


def test_tracking_does_not_change_movement_or_rng(monkeypatch):
    def run():
        prey = Prey(x=100, y=100)
        environment = create_environment(
            [prey],
            [Plant(x=300, y=100)],
            seed=246,
        )
        for _ in range(80):
            environment.update_agents(strategy="random")
        return (
            prey.x,
            prey.y,
            prey.direction,
            prey.energy,
            prey.distance_traveled,
            prey.steps_survived,
            prey.stagnation_steps,
            environment.random.getstate(),
        )

    tracked = run()
    with monkeypatch.context() as patch:
        patch.setattr(
            BaseAgent,
            "synchronize_food_progress_references",
            lambda agent, plants: None,
        )
        patch.setattr(
            BaseAgent,
            "record_food_progress_movement",
            lambda agent, snapshots, distances, diagonal: None,
        )
        untracked = run()

    assert tracked == untracked


def test_progress_tracking_does_not_consume_extra_random_values():
    prey = Prey(x=100, y=100, direction_timer=44)
    environment = create_environment(
        [prey],
        [Plant(x=300, y=100)],
        seed=987,
    )
    expected_rng = random.Random()
    expected_rng.setstate(environment.random.getstate())
    expected_rng.uniform(0, 2 * math.pi)

    environment.update_agents(strategy="random")

    assert environment.random.getstate() == expected_rng.getstate()

