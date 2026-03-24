from collections import deque
import random
import typing


Move = str
Point = typing.Dict[str, int]

DIRECTIONS: typing.Dict[Move, typing.Tuple[int, int]] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

def choose_heuristic_move(game_state: typing.Dict) -> Move:
    candidates = _safe_moves(game_state)
    if not candidates:
        return "down"

    # Pre-compute once, pass into evaluator
    board = game_state["board"]
    occupied = _occupied_cells(board["snakes"])
    hazards = hazard_cells(game_state)
    damage = hazard_damage_per_turn(game_state)

    scored_moves = [
        (move, _evaluate_move(game_state, move, occupied, hazards, damage))
        for move in candidates
    ]
    return max(scored_moves, key=lambda pair: pair[1])[0]


def backwards_move(game_state: typing.Dict) -> typing.Optional[Move]:
    # We've included code to prevent your Battlesnake from moving backwards
    body = game_state["you"]["body"]

    my_head = body[0]
    my_neck = body[1]

    # at the start of the game, head and neck can be in the same position because of how the game engine spawns snakes. In this case we don't want to restrict any backwards move because we don't know which way we are moving yet.
    if my_neck["x"] == my_head["x"] and my_neck["y"] == my_head["y"]:
        return None

    if my_neck["x"] < my_head["x"]:  # Neck is left of head, don't move left
        neck_move = "left"

    elif my_neck["x"] > my_head["x"]:  # Neck is right of head, don't move right
        neck_move = "right"

    elif my_neck["y"] < my_head["y"]:  # Neck is below head, don't move down
        neck_move = "down"

    elif my_neck["y"] > my_head["y"]:  # Neck is above head, don't move up
        neck_move = "up"

    return neck_move

def _safe_moves(game_state: typing.Dict) -> typing.List[Move]:
    is_move_safe = {"up": True, "down": True, "left": True, "right": True}
    # Prevent moving backwards
    back = backwards_move(game_state)
    if back:
        is_move_safe[back] = False

    # prevent going into walls
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    my_head = game_state["you"]["body"][0]

    if my_head["x"] == 0:
        is_move_safe["left"] = False
    elif my_head["x"] == board_width - 1:
        is_move_safe["right"] = False
    if my_head["y"] == 0:
        is_move_safe["down"] = False
    elif my_head["y"] == board_height - 1:
        is_move_safe["up"] = False

    # prevent colliding with self
    my_body = game_state['you']['body']
    for segment in my_body[2:]: # start at 2 because you can never collide with first 3 segments
        if segment["x"] == my_head["x"] and segment["y"] == my_head["y"] + 1:
            is_move_safe["up"] = False
        elif segment["x"] == my_head["x"] and segment["y"] == my_head["y"] - 1:
            is_move_safe["down"] = False
        elif segment["x"] == my_head["x"] - 1 and segment["y"] == my_head["y"]:
            is_move_safe["left"] = False
        elif segment["x"] == my_head["x"] + 1 and segment["y"] == my_head["y"]:
            is_move_safe["right"] = False


    # prevent colliding with other snakes
    opponents = game_state['board']['snakes']
    for snake in opponents[1:]: # start at 2nd snake because i am first snake in list
        for segment in snake['body']:
            if segment["x"] == my_head["x"] and segment["y"] == my_head["y"] + 1:
                is_move_safe["up"] = False
            elif segment["x"] == my_head["x"] and segment["y"] == my_head["y"] - 1:
                is_move_safe["down"] = False
            elif segment["x"] == my_head["x"] - 1 and segment["y"] == my_head["y"]:
                is_move_safe["left"] = False
            elif segment["x"] == my_head["x"] + 1 and segment["y"] == my_head["y"]:
                is_move_safe["right"] = False

    # TODO: make a check. If colide with head of other snake, only unsafe if other snake is same length or longer


    return [move for move, safe in is_move_safe.items() if safe]


def hazard_cells(game_state: typing.Dict) -> typing.Set[typing.Tuple[int, int]]:
    """ returns the set of all currently active hazard pit coordinates."""
    return{
        (h["x"], h["y"])
        for h in game_state["board"].get("hazards", [])
    }
        
def hazard_damage_per_turn(game_state: typing.Dict) -> int:
    """ falls back to 14 (one stack) if field isnt present, each stack adds 14 damage so fully stacked (4) = 56 per turn"""
    settings = (
        game_state
        .get("game", {})
        .get("ruleset", {})
        .get("settings", {})
    )
    
    return int(settings.get("hazardDamagePerTurn", 14))

def _evaluate_move(game_state: typing.Dict, move: Move, occupied: typing.Set[typing.Tuple[int, int]], hazards: typing.Set[typing.Tuple[int, int]], hazard_damage: int) -> float:
    board = game_state["board"]
    width = board["width"]
    height = board["height"]
    food = board["food"]

    you = game_state["you"]
    health = you["health"]
    body = you["body"]
    head = body[0]

    dx, dy = DIRECTIONS[move]
    next_pos = (head["x"] + dx, head["y"] + dy)

   
    # Remove  tails that will vacate next turn from blocked set (all tails of snakes that didnt just eat)
    vacating = _tails_vacating_next_turn(board["snakes"])
    occupied -= vacating


    # Feature 1: prefer positions with more reachable space.
    free_space_weight = 180
    # flood fill algorithm to find how much free space is reachable from the next position. This helps avoid moves that lead to traps.
    free_space = _flood_fill_area(next_pos, occupied, width, height)
    # normalize
    max_area = width * height
    free_space_normalized = free_space / max_area  # now in range [0.0, 1.0]


    # Feature 2: Hard penalty for moves leading into spaces too small to survive
    dead_end_penalty = 0.0
    if free_space < len(body):
        dead_end_penalty = 50.0  # large enough to override all other features
    

    # Feature 3: encourage food seeking when health is low.
    nearest_food_distance = min(
        _manhattan(next_pos, (f["x"], f["y"])) for f in food
    )
    nearest_food_distance_weight = 0.6 if health >= 40 else 2.0
    nearest_food_score = max(0, 10 - nearest_food_distance)
    

    # Feature 4: avoid walls to reduce trap risk.
    wall_clearence_weight = 0.5
    wall_clearance = min(
        next_pos[0],
        next_pos[1],
        (width - 1) - next_pos[0],
        (height - 1) - next_pos[1],
    )

  # Feature 5: reward moving next to a weaker enemy head (aggression)
    head_to_head_weight = 1.5
    length_advantage_score = 0.0
    my_length = len(body)
    for snake in board["snakes"]:
        if snake["id"] == you["id"]:
            continue
        enemy_length = len(snake["body"])
        enemy_head = snake["body"][0]
        dist = _manhattan(next_pos, (enemy_head["x"], enemy_head["y"]))
        if dist == 1 and enemy_length < my_length:
            # Scale reward: bigger size gap = more confident kill
            length_advantage_score += (my_length - enemy_length)

    # Feature 6: penalise moving next to a stronger/equal enemy head
    danger_penalty = 0.0
    for snake in board["snakes"]:
        if snake["id"] == you["id"]:
            continue
        enemy_length = len(snake["body"])
        enemy_head = snake["body"][0]
        dist = _manhattan(next_pos, (enemy_head["x"], enemy_head["y"]))
        if dist == 1 and enemy_length >= my_length:
            # Scale penalty: much longer enemy = much more dangerous
            size_gap = enemy_length - my_length
            danger_penalty += 4.0 + size_gap * 1.5

    # Feature 7: hazard pit penalty 
    # penalised by scaling how much damage hazard will deal and how low our health is (lower = worse)
    hazard_weight = 3.0
    hazard_penalty = 0.0
    
    if next_pos in hazards:
        # health ratio: fraction of health lost if in one turn inside the pit
        health_ratio = hazard_damage / (max(1, health))
        # clamp to [0,1] rankge so fully stacked pit at low health gives penalty of 1.0
        hazard_penalty = hazard_weight * min(1.0, health_ratio)


    # Feature 8: soft preference for center of board
    center_weight = 0.3
    center_x, center_y = (width - 1) / 2, (height - 1) / 2
    center_distance = _manhattan(next_pos, (int(center_x), int(center_y)))

    
    # Feature 9: food avoidance when healthy soft penalty 
    food_positions = {(f["x"], f["y"]) for f in food}
    food_avoidance_penalty = 0.0
    if next_pos in food_positions and health >= 30:
        food_avoidance_penalty = 3.0  # soft penalty, overrideable if all moves are bad
        
    score = (
        free_space_weight * free_space_normalized 
        + nearest_food_distance_weight * nearest_food_score 
        + wall_clearence_weight * wall_clearance 
        + head_to_head_weight * length_advantage_score 
        - center_weight * center_distance
        - dead_end_penalty
        - danger_penalty
        - hazard_penalty  
        - food_avoidance_penalty 
    )

    return score

# ----- Utility functions -----

def _enemy_head_threat_cells(snakes: typing.List[typing.Dict], my_length: int) -> typing.Set[typing.Tuple[int, int]]:
    threat_cells: typing.Set[typing.Tuple[int, int]] = set()

    for snake in snakes:
        if len(snake["body"]) < my_length:
            continue

        head = snake["body"][0]
        for dx, dy in DIRECTIONS.values():
            threat_cells.add((head["x"] + dx, head["y"] + dy))

    return threat_cells


def _occupied_cells(snakes: typing.List[typing.Dict]) -> typing.Set[typing.Tuple[int, int]]:
    occupied: typing.Set[typing.Tuple[int, int]] = set()
    for snake in snakes:
        for segment in snake["body"]:
            occupied.add((segment["x"], segment["y"]))
    return occupied


def _tails_vacating_next_turn(
    snakes: typing.List[typing.Dict],
) -> typing.Set[typing.Tuple[int, int]]:
    """
    Returns tail positions that will be vacated next turn.
    A tail does NOT vacate if the snake just ate (health == 100).
    """
    vacating = set()
    for snake in snakes:
        # If health is 100, the snake just ate and its tail stays
        if snake.get("health", 0) == 100:
            continue
        tail = snake["body"][-1]
        vacating.add((tail["x"], tail["y"]))
    return vacating


def _flood_fill_area(
    start: typing.Tuple[int, int],
    blocked: typing.Set[typing.Tuple[int, int]],
    width: int,
    height: int,
) -> int:
    if start in blocked:
        return 0

    q = deque([start])
    visited: typing.Set[typing.Tuple[int, int]] = {start}

    while q:
        x, y = q.popleft()
        for dx, dy in DIRECTIONS.values():
            nx, ny = x + dx, y + dy
            cell = (nx, ny)
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if cell in blocked or cell in visited:
                continue
            visited.add(cell)
            q.append(cell)

    return len(visited)

def _manhattan(a: typing.Tuple[int, int], b: typing.Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

