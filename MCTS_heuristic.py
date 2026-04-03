import math
import random
import time
import typing
from heuristic_agent import _evaluate_move

# hyperparameters based on hyperarams found in hyperparam_experiment_MCTS.py

UCB_C = 1.41
MAX_DEPTH = 20
TIME_LIMIT_MS = 850

# directions
ALL_MOVES = ("up", "down", "left", "right")

DELTAS = {
    "up":    (0,  1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": (1,  0),
}

# given a position and a direction, return the new position after moving in that direction
def _apply(x: int, y: int, direction: str) -> typing.Tuple[int, int]:
    dx, dy = DELTAS[direction]
    return x + dx, y + dy


# class representing the state of a snake: its body segments, health, length, and alive status
class SnakeState:
    __slots__ = ("id", "body", "health", "length", "alive")

    def __init__(self, d: typing.Dict):
        self.id     = d["id"]
        self.body   = [(s["x"], s["y"]) for s in d["body"]]
        self.health = int(d["health"])
        self.length = int(d.get("length", len(self.body)))
        self.alive  = True


    def copy(self) -> "SnakeState":
        s        = object.__new__(SnakeState)
        s.id     = self.id
        s.body   = self.body[:]
        s.health = self.health
        s.length = self.length
        s.alive  = self.alive
        return s

    # return the current head position of the snake
    @property
    def head(self) -> typing.Tuple[int, int]:
        return self.body[0]



# class representing the entire game state: the board, food, hazards, snakes, and turn number
class GameSim:
    def __init__(self, game_state: typing.Dict):
        board        = game_state["board"]
        self.width   = board["width"]
        self.height  = board["height"]
        self.food    = {(f["x"], f["y"]) for f in board["food"]}
        self.hazards = {(h["x"], h["y"]) for h in board.get("hazards", [])}
        settings     = (game_state.get("game", {})
                                     .get("ruleset", {})
                                     .get("settings", {}))
        self.hazard_dmg = int(settings.get("hazardDamagePerTurn", 14))
        self.min_food   = int(settings.get("minimumFood", 2))
        self.snakes     = [SnakeState(s) for s in board["snakes"]]
        self.my_id      = game_state["you"]["id"]
        self.turn       = int(game_state["turn"])


    def copy(self) -> "GameSim":
        g            = object.__new__(GameSim)
        g.width      = self.width
        g.height     = self.height
        g.food       = self.food.copy()
        g.hazards    = self.hazards.copy()
        g.hazard_dmg = self.hazard_dmg
        g.min_food   = self.min_food
        g.snakes     = [s.copy() for s in self.snakes]
        g.my_id      = self.my_id
        g.turn       = self.turn
        return g


    # return a list of all alive snakes in the game
    def alive_snakes(self) -> typing.List[SnakeState]:
        return [s for s in self.snakes if s.alive]


    # return the snakestate of our own snake, or none if not found
    def my_snake(self) -> typing.Optional[SnakeState]:
        for s in self.snakes:
            if s.id == self.my_id:
                return s
        return None


    # check if the game has reached a terminal state (our snake is dead or only one snake remains alive)
    def is_terminal(self) -> bool:
        me = self.my_snake()
        if me is None or not me.alive:
            return True
        return len(self.alive_snakes()) <= 1


    # return a set of all board positions occupied by alive snakes (excluding their heads)
    def _occupied(self) -> typing.Set[typing.Tuple[int, int]]:
        occ = set()
        for s in self.snakes:
            if s.alive:
                for seg in s.body[:-1]:
                    occ.add(seg)
        return occ


    # gives a list of safe move directions for a given snake (ones that dont lead to death)
    def safe_moves(self, snake: SnakeState) -> typing.List[str]:
        occ = self._occupied()
        result = []
        for d in ALL_MOVES:
            nx, ny = _apply(snake.head[0], snake.head[1], d)
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if (nx, ny) not in occ:
                    result.append(d)
        return result if result else list(ALL_MOVES)


    # given a dictionary of actions for each snake, update the game state by applying actions 
    # and resolving the resulting state changes (movement, eating, hazards, collisions, etc)
    def step(self, actions: typing.Dict[str, str]) -> None:
        self.turn += 1

        new_heads: typing.Dict[str, typing.Tuple[int, int]] = {}
        for snake in self.alive_snakes():
            d = actions.get(snake.id)
            if d is None:
                safe = self.safe_moves(snake) # if no action provided, choose a random safe move
                d = random.choice(safe) if safe else "down"
            nx, ny = _apply(snake.head[0], snake.head[1], d) # calculate new head position based on the chosen direction
            new_heads[snake.id] = (nx, ny)
            snake.body.insert(0, (nx, ny)) # move the snake by adding the new head position to the front of its body
            snake.health -= 1

        # track which snakes have eaten food this turn so we can grow them and not remove their tails
        ate: typing.Set[str] = set() 
        for snake in self.alive_snakes():
            if new_heads[snake.id] in self.food:
                self.food.discard(new_heads[snake.id]) # remove the food from the board
                snake.health = 100
                snake.length += 1
                ate.add(snake.id)

        # if the snake didn't eat, remove the tail segment to keep its length constant
        for snake in self.alive_snakes():
            if snake.id not in ate:
                snake.body.pop() 

        # apply hazard damage to any snake whose head is in a hazard tile
        for snake in self.alive_snakes():
            if snake.head in self.hazards:
                snake.health -= self.hazard_dmg

        # check for collisions and out-of-bounds conditions to determine if any snakes died
        for snake in self.alive_snakes():
            hx, hy = snake.head
            if not (0 <= hx < self.width and 0 <= hy < self.height):
                snake.alive = False
            elif snake.health <= 0:
                snake.alive = False

        # check for head-to-body collisions (a snake's head colliding with another snake's body)
        all_bodies: typing.Set[typing.Tuple[int, int]] = set()
        for snake in self.alive_snakes():
            for segment in snake.body[1:]:
                all_bodies.add(segment)
        for snake in self.alive_snakes():
            if snake.head in all_bodies:
                snake.alive = False

        # check for head-to-head collisions (multiple snakes moving into the same position)
        head_groups: typing.Dict[typing.Tuple[int, int], typing.List[SnakeState]] = {}
        for snake in self.alive_snakes():
            head_groups.setdefault(snake.head, []).append(snake) # group snakes by their new head positions to find collisions
        for group in head_groups.values():
            if len(group) > 1:
                max_len = max(s.length for s in group)
                for snake in group:
                    if snake.length < max_len:
                        snake.alive = False

        # ensure there is always a minimum amount of food on the board 
        # (randomly added until we reach the minimum)
        while len(self.food) < self.min_food:
            for _ in range(20):
                x = random.randrange(self.width)
                y = random.randrange(self.height)
                if (x, y) not in self.food:
                    self.food.add((x, y))
                    break



# ------- Terminal evaluation ------------

# given a terminal game state, return a score representing how good the outcome is for our snake
def _evaluate(sim: GameSim) -> float:
    me = sim.my_snake()
    if me is None or not me.alive:
        return 0.0
    alive = sim.alive_snakes()
    if len(alive) == 1:
        return 1.0
    total = len(sim.snakes)
    n_dead = total - len(alive)
    survival = n_dead / max(1, total - 1)
    max_opp  = max((s.length for s in alive if s.id != sim.my_id), default=1) # find the length of the longest opponent snake still alive
    length_b = min(0.1, 0.1 * (me.length / max(1, max_opp) - 1)) # give a small bonus for having a longer snake (10%)
    return min(1.0, max(0.0, survival * 0.5 + 0.5 + length_b - 0.5))



# -------- MCTS Node ------------

# class representing a node in the MCTS tree containing game state, parent/child relationships, visit/win statistics, untried moves
class Node:
    __slots__ = ("game", "parent", "move", "children",
                 "visits", "wins", "untried_moves")

    def __init__(self, game: GameSim, parent=None, move=None):
        self.game          = game
        self.parent        = parent
        self.move          = move
        self.children: typing.List["Node"] = []
        self.visits        = 0
        self.wins          = 0.0
        me                 = game.my_snake()
        self.untried_moves = game.safe_moves(me) if (me and me.alive) else []


    # calculate the UCB1 score for this node to balance exploration and exploitation
    def ucb1(self) -> float:
        if self.visits == 0:
            return float("inf")
        return (self.wins / self.visits +
                UCB_C * math.sqrt(math.log(self.parent.visits) / self.visits))


    # check if all possible moves from this node have been tried (so if we can expand further)
    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0


    # check if this node represents a terminal game state (win/loss/draw)
    def is_terminal(self) -> bool:
        return self.game.is_terminal()


    # select the child node with the highest UCB1 score to explore next
    def best_child(self) -> "Node":
        return max(self.children, key=lambda n: n.ucb1())


    # expand this node by taking one of untried moves, applying it to game state and creating a new child node for that move
    def expand(self) -> "Node":
        direction = self.untried_moves.pop(random.randrange(len(self.untried_moves)))
        new_game = self.game.copy()
        actions: typing.Dict[str, str] = {} # build the actions dictionary for this turn
        for snake in new_game.alive_snakes():
            # for our snake use the chosen direction
            if snake.id == new_game.my_id:
                actions[snake.id] = direction
            # for opponents choose a random safe move (or "down" if no safe moves)
            else:
                safe = list(new_game.safe_moves(snake))
                actions[snake.id] = random.choice(safe) if safe else "down"
        new_game.step(actions)
        child = Node(new_game, parent=self, move=direction) # create a new child node with the resulting game state and move 
        self.children.append(child) # add new child node to this node's children list
        return child
    


    # ======== HEURISTIC ROLLOUT =======

    # perform a rollout (simulation) from this node's game state to a terminal state, 
    # using a heuristic policy for our snake and random moves for opponents, 
    # and return the resulting score of the terminal state.
    def rollout(self) -> float:
        sim = self.game.copy()
        depth = 0
        while not sim.is_terminal() and depth < MAX_DEPTH:
            actions = {}
            occupied = sim._occupied()
            for s in sim.alive_snakes():
                safe = sim.safe_moves(s)
                if s.id == sim.my_id:
                    # heuristic policy for our snake only
                    current_gs = _sim_to_game_state(sim, s.id)
                    actions[s.id] = max(safe, key=lambda m: _evaluate_move(
                        current_gs, m, occupied, sim.hazards, sim.hazard_dmg
                    ))
                else:
                    # random policy for opponents
                    actions[s.id] = random.choice(safe) if safe else "down"
            sim.step(actions)
            depth += 1
        return _evaluate(sim)


    # after rollout, backpropagate the result up the tree by updating visit/win stats for this node and all its ancestors
    def backpropagate(self, result: float) -> None:
        self.visits += 1
        self.wins   += result
        if self.parent:
            self.parent.backpropagate(result)



# -------- MCTS Move Function ------------

# main function to run MCTS and choose the best move direction based on the simulations
def choose_mcts_heuristic_move(game_state: typing.Dict) -> str:
    root_game = GameSim(game_state)
    root = Node(root_game)

    # if there are no safe moves from the root state, just return a random direction (or "down" if no safe moves)
    if not root.untried_moves:
        return "down"

    deadline   = time.time() + TIME_LIMIT_MS / 1000.0 # calculate the time limit for MCTS simulations
    iterations = 0

    while time.time() < deadline:
        node = root
        # traverse the tree by selecting best child nodes until we reach a node that is not fully expanded or is terminal
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand() # if not terminal and has untried moves, expand 
        result = node.rollout() # random rollout
        node.backpropagate(result) # backpropagate result up the tree
        iterations += 1

    # if we didn't explore any nodes, return a random safe move
    if not root.children:
        me = root_game.my_snake()
        return random.choice(root_game.safe_moves(me)) if me else "down"

    # select the child of the root with the most visits as the best move to take
    best = max(root.children, key=lambda n: n.visits)
    print(
        f"MCTS: {iterations} sims | "
        f"best={best.move} "
        f"visits={best.visits} "
        f"winrate={best.wins / best.visits:.2f}"
    )
    return best.move


 # helper function to convert internal GameSim state into structured dictionary format for heuristic function 
def _sim_to_game_state(sim: GameSim, pov_id: str) -> dict:
    snakes_list = []
    you_dict = None
    for s in sim.snakes:
        if not s.alive:
            continue
        # snake dictionary entry for each snake in the simulation state with id, body, health, and length attributes
        entry = {
            "id": s.id,
            "body": [{"x": x, "y": y} for x, y in s.body],
            "health": s.health,
            "length": s.length,
        }
        # identify own snake by matching the pov_id and place at the front of the snakes list
        # while other snakes are added to list in arbitrary order
        if s.id == pov_id:
            you_dict = entry
        else:
            snakes_list.append(entry)
    
    # if for some reason our snake isnt found in the simulation state 
    # create a default entry at front of list to avoid errors in heuristic function
    you = you_dict or snakes_list[0]

    # dictionary format representing the game state for the heuristic function
    return {
        "turn": sim.turn,
        "you": you,
        "board": {
            "width": sim.width,
            "height": sim.height,
            "food": [{"x": x, "y": y} for x, y in sim.food],
            "hazards": [{"x": x, "y": y} for x, y in sim.hazards],
            "snakes": ([you_dict] + snakes_list) if you_dict else snakes_list,
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