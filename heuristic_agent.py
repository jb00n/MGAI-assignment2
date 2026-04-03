from collections import deque
import random
import typing


Move = str
Point = typing.Dict[str, int]

DIRECTIONS: typing.Dict[Move, typing.Tuple[int, int]] = {
    "up":       (0, 1),
    "down":     (0, -1),
    "left":     (-1, 0),
    "right":    (1, 0),
}

# weights
free_space_weight               = 100
dead_end_penalty_weight         = 50     
nearest_food_distance_weight_h  = 4.0
nearest_food_distance_weight_l  = 6.0
food_avoid_penalty              = 0.2
wall_clearence_weight           = 0.2
center_weight                   = 0.5
head_to_head_weight             = 3.0
danger_base                     = 8.0
danger_size_scale               = 1.5
hazard_weight                   = 100             


# main function to choose a move based on heuristics. Evaluates each safe move using a weighted combination of features and picks the best one.
def choose_heuristic_move(game_state: typing.Dict, weights=None) -> Move:
    # default weights fallback
    if weights is None:
        weights = {
            "free_space_weight": free_space_weight,
            "dead_end_penalty_weight": dead_end_penalty_weight,
            "nearest_food_distance_weight_h": nearest_food_distance_weight_h,
            "nearest_food_distance_weight_l": nearest_food_distance_weight_l,
            "food_avoid_penalty": food_avoid_penalty,
            "wall_clearence_weight": wall_clearence_weight,
            "center_weight": center_weight,
            "head_to_head_weight": head_to_head_weight,
            "danger_base": danger_base,
            "danger_size_scale": danger_size_scale,
            "hazard_weight": hazard_weight,
        }

    candidates = _safe_moves(game_state)
    if not candidates:
        return "down"

    board       = game_state["board"]
    occupied    = _occupied_cells(board["snakes"])
    hazards     = hazard_cells(game_state)
    damage      = hazard_damage_per_turn(game_state)

    scored_moves = [
        (
            move,
            _evaluate_move(
                game_state,
                move,
                occupied,
                hazards,
                damage,
                weights,
            ),
        )
        for move in candidates
    ]

    return max(scored_moves, key=lambda pair: pair[1])[0]

# prevent your Battlesnake from moving backwards
def backwards_move(game_state: typing.Dict) -> typing.Optional[Move]:
    body = game_state["you"]["body"]

    my_head = body[0]
    my_neck = body[1]

    # at the start of the game, head and neck can be in the same position because of how the game engine spawns snakes. In this case we don't want to restrict any backwards move because we don't know which way we are moving yet.
    if my_neck["x"] == my_head["x"] and my_neck["y"] == my_head["y"]:
        return None

    if my_neck["x"] < my_head["x"]:  # neck is left of head, don't move left
        neck_move = "left"

    elif my_neck["x"] > my_head["x"]:  # neck is right of head, don't move right
        neck_move = "right"

    elif my_neck["y"] < my_head["y"]:  # neck is below head, don't move down
        neck_move = "down"

    elif my_neck["y"] > my_head["y"]:  # neck is above head, don't move up
        neck_move = "up"

    return neck_move

# returns list of moves that dont immediately lead to death (moving into wall, self, or other snake body)
def _safe_moves(game_state: typing.Dict) -> typing.List[Move]:
    is_move_safe = {"up": True, "down": True, "left": True, "right": True}
    my_id = game_state["you"]["id"]

    # prevent moving backwards
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


    for snake in game_state['board']['snakes']:
        if snake["id"] == my_id:
            continue  # skip self

        # avoid body segments (excluding their tail which may vacate)
        for segment in snake['body'][:-1]:  # exclude tail
            if segment["x"] == my_head["x"] and segment["y"] == my_head["y"] + 1:
                is_move_safe["up"] = False
            elif segment["x"] == my_head["x"] and segment["y"] == my_head["y"] - 1:
                is_move_safe["down"] = False
            elif segment["x"] == my_head["x"] - 1 and segment["y"] == my_head["y"]:
                is_move_safe["left"] = False
            elif segment["x"] == my_head["x"] + 1 and segment["y"] == my_head["y"]:
                is_move_safe["right"] = False
    return [move for move, safe in is_move_safe.items() if safe]


# returns the set of all currently active hazard pit coordinates
def hazard_cells(game_state: typing.Dict) -> typing.Set[typing.Tuple[int, int]]:
    return{
        (h["x"], h["y"])
        for h in game_state["board"].get("hazards", [])
    }
        

# falls back to 14 (one stack) if field isnt present, each stack adds 14 damage so fully stacked (4) = 56 per turn
def hazard_damage_per_turn(game_state: typing.Dict) -> int:
    settings = (
        game_state
        .get("game", {})
        .get("ruleset", {})
        .get("settings", {})
    )
    
    return int(settings.get("hazardDamagePerTurn", 14))

# evaluates a move based on multiple heuristics and returns a score. Higher score = better move.
def _evaluate_move(game_state: typing.Dict, move: Move, occupied: typing.Set[typing.Tuple[int, int]], hazards: typing.Set[typing.Tuple[int, int]], hazard_damage: int, weights: typing.Dict[str, float]=None) -> float:
    # default weights fallback
    if weights is None:
        weights = {
            "free_space_weight": free_space_weight,
            "dead_end_penalty_weight": dead_end_penalty_weight,
            "nearest_food_distance_weight_h": nearest_food_distance_weight_h,
            "nearest_food_distance_weight_l": nearest_food_distance_weight_l,
            "food_avoid_penalty": food_avoid_penalty,
            "wall_clearence_weight": wall_clearence_weight,
            "center_weight": center_weight,
            "head_to_head_weight": head_to_head_weight,
            "danger_base": danger_base,
            "danger_size_scale": danger_size_scale,
            "hazard_weight": hazard_weight,
        }
    
    board   = game_state["board"]
    width   = board["width"]
    height  = board["height"]
    food    = board["food"]

    you     = game_state["you"]
    health  = you["health"]
    body    = you["body"]
    head    = body[0]

    dx, dy = DIRECTIONS[move]
    next_pos = (head["x"] + dx, head["y"] + dy)

   
    # remove tails that will vacate next turn from blocked set (all tails of snakes that didnt just eat)
    vacating = _tails_vacating_next_turn(board["snakes"])
    occupied = occupied - vacating # dont mutate original set


    # Feature 1: prefer positions with more reachable space.
    # flood fill algorithm to find how much free space is reachable from the next position. This helps avoid moves that lead to traps.
    free_space = _flood_fill_area(next_pos, occupied, width, height)
    # normalize
    max_area = width * height
    free_space_normalized = free_space / max_area  # now in range [0.0, 1.0]


    # Feature 2: Hard penalty for moves leading into spaces too small to survive
    dead_end_penalty = 0.0
    if free_space < len(body):
        dead_end_penalty = weights["dead_end_penalty_weight"]  # large enough to override all other features
    

    # Feature 3: prefer moves that get closer to food, but only if the food is safe (not in hazard or next to hazard when low health). If no safe food, dont encourage getting closer to unsafe food.
    nearest_food_score = 0.0

    safe_food_distances = []
    risky_food_distances = []

    for f in food:
        food_pos = (f["x"], f["y"])
        dist = _manhattan(next_pos, food_pos)

        if food_pos in hazards:
            total_damage = hazard_damage + 1

            # ff it kills us → NEVER consider
            if total_damage >= health:
                continue

            remaining_health = health - total_damage

            # risky but survivable food
            if remaining_health < 20:
                risky_food_distances.append(dist)
            else:
                safe_food_distances.append(dist)
        else:
            # normal safe food
            safe_food_distances.append(dist)


    nearest_food_distance_weight = weights["nearest_food_distance_weight_h"] if health >= 40 else weights["nearest_food_distance_weight_l"]

    if safe_food_distances:
        # prefer safe food
        nearest_food_distance = min(safe_food_distances)
        nearest_food_score = max(0, 10 - nearest_food_distance)

    elif risky_food_distances:
        # no safe food → go for risky food (but weaker reward)
        nearest_food_distance = min(risky_food_distances)
        nearest_food_score = 0.5 * max(0, 10 - nearest_food_distance)

    else:
        # no reachable food at all
        nearest_food_score = 0
        

    # Feature 4: avoid walls to reduce trap risk.
    wall_clearance = min(
        next_pos[0],
        next_pos[1],
        (width - 1) - next_pos[0],
        (height - 1) - next_pos[1],
    )

  # Feature 5: reward moving next to a weaker enemy head (aggression)
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
            danger_penalty += weights["danger_base"] + size_gap * weights["danger_size_scale"]

    # Feature 7: hazard pit penalty 
    hazard_penalty = 0.0

    if next_pos in hazards:
        # damage taken this turn (hazard + normal turn damage)
        total_damage = hazard_damage + 1

        # Case 1: Immediate death → absolutely forbid move
        if total_damage >= health:
            hazard_penalty = 1000.0  # effectively "never pick this"

        else:
            # remaining health after stepping in hazard
            remaining_health = health - total_damage

            # strong nonlinear penalty as health gets low
            # (quadratic curve makes low health MUCH scarier)
            danger_ratio = total_damage / health
            hazard_penalty = weights["hazard_weight"] * (danger_ratio ** 2)

            # extra punishment if we'd be critically low
            if remaining_health < 15:
                hazard_penalty += 100

    # Feature 8: soft preference for center of board
    center_x, center_y = (width - 1) / 2, (height - 1) / 2
    center_distance = _manhattan(next_pos, (int(center_x), int(center_y)))

    
    # Feature 9: food avoidance when healthy soft penalty 
    food_positions = {(f["x"], f["y"]) for f in food}
    food_avoidance_penalty = 0.0
    if next_pos in food_positions and health >= 30:
        food_avoidance_penalty = weights["food_avoid_penalty"]  # soft penalty, overrideable if all moves are bad
        
    score = (
        weights["free_space_weight"] * free_space_normalized 
        + nearest_food_distance_weight * nearest_food_score 
        + weights["wall_clearence_weight"] * wall_clearance 
        + weights["head_to_head_weight"] * length_advantage_score 
        - weights["center_weight"] * center_distance
        - dead_end_penalty
        - danger_penalty
        - hazard_penalty  
        - food_avoidance_penalty 
    )

    return score

# ----- Utility functions -----

# returns the set of cells adjacent to enemy heads that are occupied by snakes at least as long as us (potential head-to-head collision threats).
def _enemy_head_threat_cells(snakes: typing.List[typing.Dict], my_length: int) -> typing.Set[typing.Tuple[int, int]]:
    threat_cells: typing.Set[typing.Tuple[int, int]] = set()

    for snake in snakes:
        if len(snake["body"]) < my_length:
            continue

        head = snake["body"][0]
        for dx, dy in DIRECTIONS.values():
            threat_cells.add((head["x"] + dx, head["y"] + dy))

    return threat_cells


# returns the set of all cells occupied by any snake body segment (including tails that will vacate)
def _occupied_cells(snakes: typing.List[typing.Dict]) -> typing.Set[typing.Tuple[int, int]]:
    occupied: typing.Set[typing.Tuple[int, int]] = set()
    for snake in snakes:
        for segment in snake["body"]:
            occupied.add((segment["x"], segment["y"]))
    return occupied


# returns the set of tail positions that will vacate next turn (all tails of snakes that didnt just eat)
def _tails_vacating_next_turn(
    snakes: typing.List[typing.Dict],
) -> typing.Set[typing.Tuple[int, int]]:
    vacating = set()
    for snake in snakes:
        # If health is 100, the snake just ate and its tail stays
        if snake.get("health", 0) == 100:
            continue
        tail = snake["body"][-1]
        vacating.add((tail["x"], tail["y"]))
    return vacating


# flood fill algorithm to calculate how many free cells are reachable from a given starting point, given a set of blocked cells. 
# Used to evaluate how much free space is accessible from a potential move.
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


# Manhattan distance heuristic for estimating distance between two points on the grid. Used for food distance and head-to-head threat evaluation.
def _manhattan(a: typing.Tuple[int, int], b: typing.Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])