# should be able to run by executing:
# python experiments.py --games 100 --snakes 4 --width 11 --height 11 --seed 42
# this should simulate instead of actually running on the Battlesnake server

import argparse
import random
import math
import typing
from collections import defaultdict
from trueskill import Rating, rate, BETA, quality_1vs1
import time
import csv

from MCTS_random import GameSim, SnakeState, ALL_MOVES, _apply

# agent imports
from heuristic_agent import choose_heuristic_move
from MCTS_random import choose_mcts_move
from MCTS_heuristic import choose_mcts_heuristic_move
# import RAVE move function and other improvements here when ready


AGENTS: typing.Dict[str, typing.Callable] = {
    "heuristic": choose_heuristic_move,
    "mcts": choose_mcts_move,
    "mcts_heuristic": choose_mcts_heuristic_move
    # other agents here when ready
}

# -------- ELO ---------

# originally designed for chess - every game treated as a series of 1v1 matches between players, with a win/loss/draw outcome
# expected score = 1 / (1 + 10^((opponent_rating - your_rating) / 400))
# new rating = old_rating + K * (actual_score - expected_score)
# if you beat someone higher than you = you earn more points 
# if you beat someone lower than you = you earn fewer points

INITIAL_ELO = 1000.0
ELO_K = 32

# returns expected score for player given their elo and opponent's elo
def expected_elo(player_elo: float, opponent_elo: float) -> float:
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))

# updates ratings in place based on ranking of players in a game (first place beats everyone else, second place beats everyone except first etc)
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
        
# -------- GAME SIMULATION ---------

# helper function to create a game state with random snake positions and food (for testing agents without running an actual server)
def make_game_state(width: int, height: int, agent_ids: typing.List[str]) -> typing.Dict:
   
    n = len(agent_ids)
    # candidate spawn points spread around the board
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
            "id":     aid,
            "health": 100,
            "length": 3,
            "body":   [pos, pos, pos],   # all 3 segments on same cell at spawn
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
        "you":  snakes[0],          # placeholder — overridden per-agent below
        "board": {
            "width":   width,
            "height":  height,
            "food":    food,
            "hazards": [],
            "snakes":  snakes,
        },
        "game": {
            "ruleset": {
                "settings": {
                    "hazardDamagePerTurn": 14,
                    "minimumFood":         2,
                }
            }
        },
    }
    
# helper function to convert a base game state into the perspective of a specific agent (setting 'you' to the snake matching my_id)
def _pov_state(base_state: typing.Dict, my_id: str) -> typing.Dict:
    """Return a copy of base_state with 'you' set to the snake matching my_id."""
    board = base_state["board"]
    you = next(s for s in board["snakes"] if s["id"] == my_id)
    return {**base_state, "you": you}


# helper function to convert a live GameSim back into a game-state dict for the agents (for testing agents without running an actual server)
def _sim_to_base_state(sim: GameSim) -> typing.Dict:

    snakes = []
    for s in sim.snakes:
        if not s.alive:
            continue
        snakes.append({
            "id":     s.id,
            "health": s.health,
            "length": s.length,
            "body":   [{"x": x, "y": y} for x, y in s.body],
        })
    return {
        "turn": sim.turn,
        "you":  snakes[0] if snakes else {},
        "board": {
            "width":   sim.width,
            "height":  sim.height,
            "food":    [{"x": x, "y": y} for x, y in sim.food],
            "hazards": [{"x": x, "y": y} for x, y in sim.hazards],
            "snakes":  snakes,
        },
        "game": {
            "ruleset": {
                "settings": {
                    "hazardDamagePerTurn": sim.hazard_dmg,
                    "minimumFood":         sim.min_food,
                }
            }
        },
    }

# helper function to run a game between agents without needing to start an actual server (for testing agents against each other)
# returns ranking of agents by finish (first place is best, last place is worst)
def run_game(agent_names: typing.List[str],
             move_fns: typing.List[typing.Callable],
             width: int = 11,
             height: int = 11,
             max_turns: int = 500) -> typing.List[str]:
   
   # create initial game state and simulator
    base = make_game_state(width, height, agent_names)
    sim  = GameSim(base)

    # create mapping of agent IDs to their move functions for easy lookup during the game loop
    agent_map = dict(zip(agent_names, move_fns))
    death_order: typing.List[typing.List[str]] = []   # groups of same-turn deaths

    # game loop - step through turns until max_turns or all snakes dead, applying moves from each agent's move function
    for _ in range(max_turns):
        if sim.is_terminal():
            break

        base_state = _sim_to_base_state(sim)
        actions: typing.Dict[str, str] = {}

        for snake in sim.alive_snakes():
            fn = agent_map[snake.id]
            pov = _pov_state(base_state, snake.id)
            try:
                move = fn(pov)
            except Exception:
                move = "down"
            actions[snake.id] = move

        alive_before = {s.id for s in sim.alive_snakes()}
        sim.step(actions)
        alive_after  = {s.id for s in sim.alive_snakes()}

        newly_dead = alive_before - alive_after
        if newly_dead:
            death_order.append(list(newly_dead))

    # survivors are ranked above all dead snakes
    survivors = [s.id for s in sim.alive_snakes()]
    if survivors:
        # survivors are ranked above all dead snakes, but in no particular order among themselves since we dont know who would have won if the game had continued
        ranking = survivors + [name for group in reversed(death_order) for name in group]
    else:
        # if no survivors, just rank by death order (later deaths are better)
        ranking = [name for group in reversed(death_order) for name in group]

    return ranking

# ---------- RUN TOURNAMENT ---------

# helper function to run a tournament of multiple games between agents and track stats of wins, ELO ratings, and TrueSkill ratings
def run_tournament(n_games: int,
                   snakes_per_game: int,
                   width: int = 11,
                   height: int = 11) -> None:

    agent_names = list(AGENTS.keys())
    move_fns = list(AGENTS.values())
    n_agents = len(agent_names)

    # initialise stats
    wins = defaultdict(int)
    played = defaultdict(int)
    elo = {name: INITIAL_ELO for name in agent_names}
    ts = {name: Rating() for name in agent_names}  # TrueSkill
    
    # Trueskill is when each player has two values = mu and sigma, representing their skill and uncertainty about their skill
    # players start with mu=25 and sigma=8.333 and as they play games, their mu and sigma are updated based on the outcomes
    # player who wins against a strong opponent will see a bigger increase in mu than if they win against a weak opponent
    # player who loses against a strong opponent will see a smaller decrease in mu than if they lose against a weak opponent
    # the sigma value decreases as the system becomes more certain about a player's skill level, which happens as they play more games

    print(f"\n{'='*60}")
    print(f"  Tournament: {n_games} games | {snakes_per_game} snakes/game")
    print(f"  Agents: {', '.join(agent_names)}")
    print(f"{'='*60}\n")

    for game_idx in range(n_games):
        # pick snakes_per_game agents (sample with replacement if needed)
        if snakes_per_game <= n_agents:
            participants = random.sample(agent_names, snakes_per_game)
        else:
            participants = random.choices(agent_names, k=snakes_per_game) # sample with replacement if more snakes per game than available agents

        # look up their move functions
        fns = [AGENTS[name] for name in participants]

        # run the game and get the ranking of agents by finish
        ranking = run_game(participants, fns, width, height)

        # win / played
        if ranking:
            wins[ranking[0]] += 1
        for name in participants:
            played[name] += 1

        # ELO update
        _update_elo(elo, ranking)

        # TrueSkill update — rate() takes groups; each agent is its own team
        ts_groups  = [(ts[name],) for name in ranking]
        new_ratings = rate(ts_groups)
        
        # update ratings in place
        for i, name in enumerate(ranking):
            ts[name] = new_ratings[i][0]

        # print progress every 10 games
        if (game_idx + 1) % 10 == 0:
            print(f"  Completed {game_idx + 1}/{n_games} games...")
            
# ---------- RESULTS ---------
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    # print table of results with agent name, games played, wins, win percentage, ELO rating, and TrueSkill mu and sigma
    print(f"\n{'Agent':<20} {'Games':>6} {'Wins':>6} {'Win%':>7} "
          f"{'ELO':>8} {'TrueSkill mu':>12} {'TrueSkill sigma':>12}")
    print("-" * 75)

    sorted_agents = sorted(agent_names,
                           key=lambda n: ts[n].mu - 3 * ts[n].sigma,
                           reverse=True)

    for name in sorted_agents:
        g = played[name]
        w = wins[name]
        wr = 100.0 * w / g if g > 0 else 0.0
        print(f"{name:<20} {g:>6} {w:>6} {wr:>6.1f}% "
              f"{elo[name]:>8.1f} {ts[name].mu:>12.3f} {ts[name].sigma:>12.3f}")

    print(f"\n  Ranked by TrueSkill conservative score (mu - 3*sigma):")
    # this gives us a lower bound on the agent's skill with high confidence (99% confidence that true skill is above this value)
    for rank, name in enumerate(sorted_agents, 1):
        conservative = ts[name].mu - 3 * ts[name].sigma
        print(f"  {rank}. {name:<20} conservative score = {conservative:.3f}")
    print()
    
    # save to TXT
    with open("results.txt", "w") as f:
        f.write(f"Tournament: {n_games} games | {snakes_per_game} snakes/game\n")
        f.write(f"{'Agent':<20} {'Games':>6} {'Wins':>6} {'Win%':>7} {'ELO':>8} {'TrueSkill mu':>13} {'TrueSkill sigma':>15} {'Conservative':>13}\n")
        f.write("-" * 85 + "\n")
        for name in sorted_agents:
            g = played[name]
            w = wins[name]
            wr = 100.0 * w / g if g > 0 else 0.0
            conservative = ts[name].mu - 3 * ts[name].sigma
            f.write(f"{name:<20} {g:>6} {w:>6} {wr:>6.1f}% {elo[name]:>8.1f} {ts[name].mu:>13.3f} {ts[name].sigma:>15.3f} {conservative:>13.3f}\n")

    print("Results saved to results.txt")
    
#---------- MAIN ---------
    
# run the tournament when this script is executed directly (not imported as a module)
if __name__ == "__main__":
    
    # parse command line arguments for tournament settings (number of games, snakes per game, board size, random seed)
    parser = argparse.ArgumentParser(description="Battlesnake agent tournament")
    parser.add_argument("--games",  type=int, default=100,
                        help="Number of games to simulate (default: 100)")
    parser.add_argument("--snakes", type=int, default=4,
                        help="Number of snakes per game (default: 4)")
    parser.add_argument("--width",  type=int, default=11,
                        help="Board width (default: 11)")
    parser.add_argument("--height", type=int, default=11,
                        help="Board height (default: 11)")
    parser.add_argument("--seed",   type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    run_tournament(args.games, args.snakes, args.width, args.height)
    
    

