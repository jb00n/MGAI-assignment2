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
    """Return the best move according to a lightweight heuristic score."""
    candidates = _safe_moves(game_state)

    # avoind food untill we need it
    candidates = _avoid_food(game_state, candidates)

    if not candidates:
        return "down"
    
    scored_moves = [(move, _evaluate_move(game_state, move)) for move in candidates]
    best_move = max(scored_moves, key=lambda pair: pair[1])
    return best_move


def backwards_move(game_state: typing.Dict) -> Move:
    # # We've included code to prevent your Battlesnake from moving backwards
    my_head = game_state["you"]["body"][0]  # Coordinates of your head
    my_neck = game_state["you"]["body"][1]  # Coordinates of your "neck"

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
    is_move_safe[backwards_move(game_state)] = False

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


def _avoid_food(game_state: typing.Dict, candidates: typing.List[Move]) -> typing.List[Move]:
    foods = game_state['board']['food']
    my_head = game_state["you"]["body"][0]
    if game_state["you"]["health"] >= 30:
        for food in foods:
            # If there's only one candidate move left, we have to take it even if it's food
            if len(candidates) <= 1:
                return candidates
            if food["x"] == my_head["x"] and food["y"] == my_head["y"] + 1:
                if "up" in candidates:
                    candidates.remove("up")
            elif food["x"] == my_head["x"] and food["y"] == my_head["y"] - 1:
                if "down" in candidates:
                    candidates.remove("down")
            elif food["x"] == my_head["x"] - 1 and food["y"] == my_head["y"]:
                if "left" in candidates:
                    candidates.remove("left")
            elif food["x"] == my_head["x"] + 1 and food["y"] == my_head["y"]:
                if "right" in candidates:
                    candidates.remove("right")
    return candidates

        


def _evaluate_move(game_state: typing.Dict, move: Move) -> float:
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

    occupied = _occupied_cells(board["snakes"])
    own_tail = body[-1]
    # we can move into own tail because own tail moves out of curren t position on next turn, 
    # TODO: unless we are eating food, then tail stays in place
    occupied.discard((own_tail["x"], own_tail["y"]))


    # Feature 1: prefer positions with more reachable space.
    free_space_weight = 1.8
    # flood fill algorithm to find how much free space is reachable from the next position. This helps avoid moves that lead to traps.
    free_space = _flood_fill_area(next_pos, occupied, width, height)
    

    # Feature 2: encourage food seeking when health is low.
    nearest_food_distance = min(
        _manhattan(next_pos, (f["x"], f["y"])) for f in food
    )
    nearest_food_distance_weight = 0.6 if health >= 40 else 2.0
    nearest_food_score = max(0, 10 - nearest_food_distance)
    

    # Feature 3: avoid walls to reduce trap risk.
    wall_clearence_weight = 0.5
    wall_clearance = min(
        next_pos[0],
        next_pos[1],
        (width - 1) - next_pos[0],
        (height - 1) - next_pos[1],
    )

    # Feature 4: when snake is longer than others, be more aggressive and allow moves that step next to enemy heads, to try to cut them off and win by head-to-head collision.
    head_to_head_weight = 1.5
    length_difference = 0
    my_length = len(body)
    for snake in board["snakes"][1:]: # skip first snake because it's me
        if len(snake["body"]) < my_length:
            enemy_head = snake["body"][0]
            if _manhattan(next_pos, (enemy_head["x"], enemy_head["y"])) == 1:
                length_difference = my_length - len(snake["body"])


    # Feature 5: avoid stepping next to stronger enemy heads.
    danger_penalty = 0.0
    for snake in board["snakes"][1:]: # skip first snake because it's me
        if len(snake["body"]) >= my_length:
            enemy_head = snake["body"][0]
            if _manhattan(next_pos, (enemy_head["x"], enemy_head["y"])) == 1:
                danger_penalty += 4.0


    score = free_space_weight * free_space + nearest_food_distance_weight * nearest_food_score + wall_clearence_weight * wall_clearance + head_to_head_weight * length_difference - danger_penalty

    return score


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


