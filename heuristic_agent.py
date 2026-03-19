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

    best_move = random.shuffle(candidates)
    
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

        
