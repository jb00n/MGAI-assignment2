# run with:
# python3 hyperparam_experiment_MCTS.py --reps 5 --snakes 4 --width 11 --height 11 --seed 42

import argparse
import random
import csv
import typing
from collections import defaultdict

from trueskill import Rating, rate
import MCTS_random  
from MCTS_random import GameSim

# ---------- HYPERPARAMETER GRID ----------

PARAM_GRID = {
    "ucb_c": [1.0, 1.41, 2.0],
    "max_depth": [10, 20, 50],
    "time_limit_ms": [700, 850, 1000],
}

BASELINE = {
    "ucb_c": 1.41,
    "max_depth": 20,
    "time_limit_ms": 850,
}

# ---------- ELO ----------

INITIAL_ELO = 1000.0
ELO_K = 32

# calculate expected win probability of player A against player B
def expected_elo(player_elo: float, opponent_elo: float) -> float:
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))

# update ratings based on ranking of players in a game
def _update_elo(ratings: typing.Dict[str, float], ranking: typing.List[str]) -> None:
    n = len(ranking)
    deltas = defaultdict(float)
    for i in range(n):
        for j in range(i + 1, n):
            winner, loser = ranking[i], ranking[j]
            ew = expected_elo(ratings[winner], ratings[loser])
            deltas[winner] += ELO_K * (1.0 - ew)
            deltas[loser]  += ELO_K * (0.0 - (1.0 - ew))
    for name, delta in deltas.items():
        ratings[name] += delta

# ---------- GAME HELPERS ----------

# create a random game state with the given dimensions and agent IDs
def make_game_state(width: int, height: int, agent_ids: typing.List[str]) -> typing.Dict:
    n = len(agent_ids)
    candidates = []
    for x in [1, width // 2, width - 2]:
        for y in [1, height // 2, height - 2]:
            candidates.append({"x": x, "y": y})
    random.shuffle(candidates)
    spawns = candidates[:n]

    snakes = []
    for i, aid in enumerate(agent_ids):
        pos = spawns[i]
        snakes.append({
            "id": aid,
            "health": 100,
            "length": 3,
            "body": [pos, pos, pos],
        })

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
def _pov_state(base_state: typing.Dict, my_id: str) -> typing.Dict:
    board = base_state["board"]
    you = next(s for s in board["snakes"] if s["id"] == my_id)
    return {**base_state, "you": you}

# convert GameSim state back to base game state format for move functions
def _sim_to_base_state(sim: GameSim) -> typing.Dict:
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

# run a game with the given agent names and move functions, returning the final ranking
def run_game(agent_names, move_fns, width=11, height=11, max_turns=500):
    base = make_game_state(width, height, agent_names)
    sim = GameSim(base)
    agent_map = dict(zip(agent_names, move_fns))
    death_order = []

    for _ in range(max_turns):
        if sim.is_terminal():
            break

        base_state = _sim_to_base_state(sim)
        actions = {}

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

    survivors = [s.id for s in sim.alive_snakes()]
    if survivors:
        ranking = survivors + [n for g in reversed(death_order) for n in g]
    else:
        ranking = [n for g in reversed(death_order) for n in g]

    return ranking

# ---------- PATCHING ----------

# helper functions to patch MCTS parameters for the candidate and restore baseline config for baselines
def _patch_mcts(ucb_c, max_depth, time_limit_ms):
    MCTS_random.UCB_C = ucb_c
    MCTS_random.MAX_DEPTH = max_depth
    MCTS_random.TIME_LIMIT_MS = time_limit_ms

# restore baseline config in MCTS module (since baselines are run after candidate in each game, we can just patch before each game)
def _restore_baseline():
    _patch_mcts(**BASELINE)

# create a move function for the baseline that uses the current MCTS parameters (which will be patched to baseline values before each game)
def _make_baseline_fn():
    def baseline_move(game_state):
        _patch_mcts(**BASELINE)
        return MCTS_random.choose_mcts_move(game_state)
    return baseline_move

# ---------- EXPERIMENT ----------

# run experiments for each config in the grid, saving results to a file and printing best config at the end
def run_experiments(reps, snakes_per_game, width, height, output_file):

    # 9 configs (3 params × 3 values)
    configs = []
    for param, values in PARAM_GRID.items():
        for val in values:
            cfg = dict(BASELINE)
            cfg[param] = val
            cfg["varied_param"] = param
            configs.append(cfg)

    candidate_name = "candidate"
    baseline_names = [f"baseline_{i}" for i in range(snakes_per_game - 1)]
    all_names = [candidate_name] + baseline_names
    baseline_fn = _make_baseline_fn()

    rows = []

    # loop through configs and run games, tracking ELO and TrueSkill ratings of candidate against baselines
    for cfg_idx, cfg in enumerate(configs):
        elo = {name: INITIAL_ELO for name in all_names}
        ts = {name: Rating() for name in all_names}
        wins = 0

        for rep in range(reps):

            print(f"[Config {cfg_idx+1}/{len(configs)}] Rep {rep+1}/{reps}")

            _patch_mcts(cfg["ucb_c"], cfg["max_depth"], cfg["time_limit_ms"])

            move_fns = [MCTS_random.choose_mcts_move] + [baseline_fn]*(snakes_per_game-1)
            ranking = run_game(all_names, move_fns, width, height)

            if ranking[0] == candidate_name:
                wins += 1

            _update_elo(elo, ranking)

            ts_groups = [(ts[n],) for n in ranking]
            new_ratings = rate(ts_groups)
            for i, name in enumerate(ranking):
                ts[name] = new_ratings[i][0]

        win_pct = 100 * wins / reps
        mu = ts[candidate_name].mu
        sigma = ts[candidate_name].sigma
        conservative = mu - 3*sigma

        rows.append({
            **cfg,
            "wins": wins,
            "win_pct": round(win_pct,2),
            "elo": round(elo[candidate_name],2),
            "trueskill_conservative": round(conservative,4)
        })

    _restore_baseline()

    # save
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # BEST CONFIG
    best = max(rows, key=lambda r: r["trueskill_conservative"])

    print("\n===== BEST CONFIG =====")
    for k, v in best.items():
        print(f"{k}: {v}")

    print(f"\nSaved to {output_file}")

# ---------- MAIN ----------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--snakes", type=int, default=4)
    parser.add_argument("--width", type=int, default=11)
    parser.add_argument("--height", type=int, default=11)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default="results.txt")

    args = parser.parse_args()

    if args.seed:
        random.seed(args.seed)

    run_experiments(args.reps, args.snakes, args.width, args.height, args.output)