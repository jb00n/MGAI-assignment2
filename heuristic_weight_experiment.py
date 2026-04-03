import argparse
import random
import typing
from collections import defaultdict
import csv
from trueskill import Rating, rate
from MCTS_random import GameSim
from heuristic_agent import choose_heuristic_move

# -------- WEIGHTS ---------

BASELINE_WEIGHTS = {
    "free_space_weight": 50,
    "dead_end_penalty_weight": 50,
    "nearest_food_distance_weight_h": 2.0,
    "nearest_food_distance_weight_l": 6.0,
    "food_avoid_penalty": 0.2,
    "wall_clearence_weight": 0.8,
    "center_weight": 1.5,
    "head_to_head_weight": 1.0,
    "danger_base": 8.0,
    "danger_size_scale": 1.5,
    "hazard_weight": 50,
}

WEIGHT_TEST_VALUES = {
    "free_space_weight": [20, 50, 100],
    "dead_end_penalty_weight": [25, 50, 150],
    "nearest_food_distance_weight_h": [1.0, 2.0, 4.0],
    "nearest_food_distance_weight_l": [3.0, 6.0, 10.0],
    "food_avoid_penalty": [0.0, 0.2, 3.0],
    "wall_clearence_weight": [0.2, 0.8, 2.0],
    "center_weight": [0.5, 1.5, 3.0],
    "head_to_head_weight": [0.0, 1.0, 3.0],
    "danger_base": [3.0, 8.0, 15.0],
    "danger_size_scale": [0.5, 1.5, 3.0],
    "hazard_weight": [20, 50, 100],
}

# -------- AGENT WRAPPER ---------

# helper to create an agent function with specific weights for the heuristic evaluation. This allows us to easily test different weight configurations in the experiment.
def make_heuristic_agent(weights):
    def agent_fn(state):
        return choose_heuristic_move(state, weights=weights)
    return agent_fn



# -------- ELO ---------

# originally designed for chess - every game treated as a series of 1v1 matches between players, with a win/loss/draw outcome
# expected score = 1 / (1 + 10^((opponent_rating - your_rating) / 400))
# new rating = old_rating + K * (actual_score - expected_score)
# if you beat someone higher than you = you earn more points 
# if you beat someone lower than you = you earn fewer points

INITIAL_ELO = 1000.0
ELO_K = 32

# returns expected score for player given their elo and opponent's elo
def expected_elo(player_elo, opponent_elo):
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))


# updates ratings in place based on ranking of players in a game (first place beats everyone else, second place beats everyone except first etc)
def _update_elo(ratings, ranking):
    n = len(ranking)
    deltas = defaultdict(float)
    for i in range(n):
        for j in range(i + 1, n):
            winner, loser = ranking[i], ranking[j]
            ew = expected_elo(ratings[winner], ratings[loser])
            deltas[winner] += ELO_K * (1.0 - ew)
            deltas[loser] += ELO_K * (0.0 - (1.0 - ew))
    for name, delta in deltas.items():
        ratings[name] += delta



# ---------- GAME HELPERS ----------

# create a random game state with the given dimensions and agent IDs
def make_game_state(width, height, agent_ids):
    n = len(agent_ids)

    # pick from a set of candidate spawn points that are reasonably spaced out and not too close to walls.
    candidates = []
    for x in [1, width // 2, width - 2]:
        for y in [1, height // 2, height - 2]:
            candidates.append({"x": x, "y": y})
    random.shuffle(candidates)
    spawns = candidates[:n]

    snakes = []
    # assign spawn points to agents in order and create snake bodies (all 3 segments on same cell at spawn)
    for i, aid in enumerate(agent_ids):
        pos = spawns[i]
        snakes.append({
            "id": aid,
            "health": 100,
            "length": 3,
            "body": [pos, pos, pos],
        })

    # scatter food randomly (avoid snake heads)
    heads = {(s["body"][0]["x"], s["body"][0]["y"]) for s in snakes}
    food = []
    while len(food) < max(2, n):
        fx, fy = random.randrange(width), random.randrange(height)
        if (fx, fy) not in heads:
            food.append({"x": fx, "y": fy})
            heads.add((fx, fy))

    return {
        "turn": 0,
        "you": snakes[0],
        "board": {
            "width": width,
            "height": height,
            "food": food,
            "hazards": [],
            "snakes": snakes,
        },
        "game": {
            "ruleset": {
                "settings": {
                    "hazardDamagePerTurn": 14,
                    "minimumFood": 2,
                }
            }
        },
    }


# convert base game state to POV for a given snake ID
def _pov_state(base_state, my_id):
    board = base_state["board"]
    you = next(s for s in board["snakes"] if s["id"] == my_id)
    return {**base_state, "you": you}


# convert GameSim state back to base game state format for move functions
def _sim_to_base_state(sim):
    snakes = []
    for s in sim.snakes:
        if not s.alive:
            continue
        snakes.append({
            "id": s.id,
            "health": s.health,
            "length": s.length,
            "body": [{"x": x, "y": y} for x, y in s.body],
        })

    return {
        "turn": sim.turn,
        "you": snakes[0] if snakes else {},
        "board": {
            "width": sim.width,
            "height": sim.height,
            "food": [{"x": x, "y": y} for x, y in sim.food],
            "hazards": [{"x": x, "y": y} for x, y in sim.hazards],
            "snakes": snakes,
        },
        "game": {
            "ruleset": {
                "settings": {
                    "hazardDamagePerTurn": sim.hazard_dmg,
                    "minimumFood": sim.min_food,
                }
            }
        },
    }




# -------- GAME RUNNER ---------

# run a game with the given agent names and move functions, returning the final ranking
def run_game(agent_names: typing.List[str],
             move_fns: typing.List[typing.Callable],
             width: int = 11,
             height: int = 11,
             max_turns: int = 500) -> typing.List[str]:
    
   # create initial game state and simulator
    base = make_game_state(width, height, agent_names)
    sim = GameSim(base)

    # create mapping of agent IDs to their move functions for easy lookup during the game loop
    agent_map = dict(zip(agent_names, move_fns))
    death_order = []

    # game loop - step through turns until max_turns or all snakes dead, applying moves from each agent's move function
    for _ in range(max_turns):
        if sim.is_terminal():
            break

        base_state = _sim_to_base_state(sim)
        actions = {}

        # on each turn, get the current state, ask each alive snake for its move, and then apply all moves simultaneously
        for snake in sim.alive_snakes():
            fn = agent_map[snake.id]
            pov = _pov_state(base_state, snake.id)
            try:
                move = fn(pov)
            except Exception:
                move = "down"
            actions[snake.id] = move

        # determine ranking based on death order and survivors
        alive_before = {s.id for s in sim.alive_snakes()}
        sim.step(actions)
        alive_after = {s.id for s in sim.alive_snakes()}

        newly_dead = alive_before - alive_after
        if newly_dead:
            death_order.append(list(newly_dead))

    # survivors are ranked above all dead snakes
    survivors = [s.id for s in sim.alive_snakes()]
    if survivors:
        # survivors are ranked above all dead snakes, but in no particular order among themselves since we dont know who would have won if the game had continued
        ranking = survivors + [n for g in reversed(death_order) for n in g]
    else:
        # if no survivors, just rank by death order (later deaths are better)
        ranking = [n for g in reversed(death_order) for n in g]

    return ranking




# -------- EXPERIMENT ---------

# runs a series of games with the test agent using the provided weights against 3 baseline agents. 
# It collects statistics on wins, games played, winrate, ELO rating, and TrueSkill ratings to evaluate the performance of the test weights.
def run_weight_experiment(test_weights, n_games, width, height):
    agents = {
        "baseline_1": make_heuristic_agent(BASELINE_WEIGHTS),
        "baseline_2": make_heuristic_agent(BASELINE_WEIGHTS),
        "baseline_3": make_heuristic_agent(BASELINE_WEIGHTS),
        "test": make_heuristic_agent(test_weights),
    }

    wins = defaultdict(int)
    played = defaultdict(int)
    elo = {name: INITIAL_ELO for name in agents}
    ts = {name: Rating() for name in agents}

    # run the specified number of games, randomly shuffling the order of agents each time to ensure fairness in starting positions and turn order.
    for _ in range(n_games):
        participants = ["baseline_1", "baseline_2", "baseline_3", "test"]
        random.shuffle(participants)

        # get the corresponding agent functions for the current participants and run a game simulation to get the ranking of players.
        fns = [agents[name] for name in participants]
        ranking = run_game(participants, fns, width, height)

        if ranking:
            wins[ranking[0]] += 1
        for name in participants:
            played[name] += 1

        _update_elo(elo, ranking)

        ts_groups = [(ts[name],) for name in ranking]
        new_ratings = rate(ts_groups)

        for i, name in enumerate(ranking):
            ts[name] = new_ratings[i][0]

    t = ts["test"]

    return {
        "wins": wins["test"],
        "games": played["test"],
        "winrate": wins["test"] / played["test"],
        "elo": elo["test"],
        "mu": t.mu,
        "sigma": t.sigma,
        "conservative": t.mu - 3 * t.sigma,
    }




# -------- LOOP ---------

# runs the full set of weight experiments by iterating over each weight and its test values, 
# running the specified number of games for each configuration, and collecting the results in a list of dictionaries.
def run_all_weight_experiments(n_games, width, height):
    results = []

    for weight_name, values in WEIGHT_TEST_VALUES.items():
        print(f"\n=== Testing {weight_name} ===")

        for val in values:
            test_weights = BASELINE_WEIGHTS.copy()
            test_weights[weight_name] = val

            print(f"  -> {val}")

            stats = run_weight_experiment(test_weights, n_games, width, height)

            results.append({
                "weight": weight_name,
                "value": val,
                **stats
            })

    return results




# -------- SAVE ---------

# saves the collected results to a CSV file for later analysis. Each row corresponds to a specific weight configuration and its performance metrics.
def save_results(results):
    with open("weight_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved to weight_results.csv")


# prints the rankings of weight configurations based on their TrueSkill conservative score.
def print_rankings(results):
    print("\n" + "=" * 60)
    print("  BEST VALUES PER WEIGHT (by TrueSkill conservative score)")
    print("=" * 60)

    # group results by weight
    grouped = defaultdict(list)
    for r in results:
        grouped[r["weight"]].append(r)

    for weight, rows in grouped.items():
        print(f"\n{weight}:")

        # sort by conservative score (descending)
        ranked = sorted(rows, key=lambda r: r["conservative"], reverse=True)

        for i, r in enumerate(ranked, 1):
            print(
                f"  {i}. value={r['value']:<6} "
                f"score={r['conservative']:.3f} "
                f"(winrate={r['winrate']:.2f}, mu={r['mu']:.2f}, sigma={r['sigma']:.2f})"
            )

        best = ranked[0]
        print(f"  -> BEST: {best['value']} (score={best['conservative']:.3f})")




# -------- MAIN ---------

# entry point for running the experiment. It parses command line arguments for the number of games, board dimensions, and random seed,
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=11)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    results = run_all_weight_experiments(
        n_games=args.games,
        width=args.width,
        height=args.height
    )

    save_results(results)

    print_rankings(results)