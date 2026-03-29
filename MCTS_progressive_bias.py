from logging import root
import math
import random
import time
import typing
from heuristic_agent import _evaluate_move

# Hyperparameters

UCB_C = 1.41
BIAS_C = 3.0
MAX_DEPTH = 20
TIME_LIMIT_MS = 850

# Directions

ALL_MOVES = ("up", "down", "left", "right")

DELTAS = {
    "up":    (0,  1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": (1,  0),
}


def _apply(x: int, y: int, direction: str) -> typing.Tuple[int, int]:
    dx, dy = DELTAS[direction]
    return x + dx, y + dy


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

    @property
    def head(self) -> typing.Tuple[int, int]:
        return self.body[0]


class GameSim:
    def __init__(self, game_state: typing.Dict):
        board           = game_state["board"]
        self.width      = board["width"]
        self.height     = board["height"]
        self.food       = {(f["x"], f["y"]) for f in board["food"]}
        self.hazards    = {(h["x"], h["y"]) for h in board.get("hazards", [])}
        settings        = (game_state.get("game", {})
                                     .get("ruleset", {})
                                     .get("settings", {}))
        self.hazard_dmg = int(settings.get("hazardDamagePerTurn", 14))
        self.min_food   = int(settings.get("minimumFood", 2))
        self.snakes     = [SnakeState(s) for s in board["snakes"]]
        self.my_id      = game_state["you"]["id"]
        self.turn       = int(game_state["turn"])

    def copy(self) -> "GameSim":
        g             = object.__new__(GameSim)
        g.width       = self.width
        g.height      = self.height
        g.food        = self.food.copy()
        g.hazards     = self.hazards.copy()
        g.hazard_dmg  = self.hazard_dmg
        g.min_food    = self.min_food
        g.snakes      = [s.copy() for s in self.snakes]
        g.my_id       = self.my_id
        g.turn        = self.turn
        return g

    def alive_snakes(self) -> typing.List[SnakeState]:
        return [s for s in self.snakes if s.alive]

    def my_snake(self) -> typing.Optional[SnakeState]:
        # TODO: we can change this, first snake is always my snake. Returning 1st is faster
        for s in self.snakes:
            if s.id == self.my_id:
                return s
        return None

    def is_terminal(self) -> bool:
        me = self.my_snake()
        if me is None or not me.alive:
            return True
        return len(self.alive_snakes()) <= 1

    def _occupied(self) -> typing.Set[typing.Tuple[int, int]]:
        occ = set()
        for s in self.snakes:
            if s.alive:
                for seg in s.body[:-1]:
                    occ.add(seg)
        return occ

    def safe_moves(self, snake: SnakeState) -> typing.List[str]:
        occ = self._occupied()
        result = []
        for d in ALL_MOVES:
            nx, ny = _apply(snake.head[0], snake.head[1], d)
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if (nx, ny) not in occ:
                    result.append(d)
        return result if result else list(ALL_MOVES)

    def step(self, actions: typing.Dict[str, str]) -> None:
        self.turn += 1

        new_heads: typing.Dict[str, typing.Tuple[int, int]] = {}
        for snake in self.alive_snakes():
            d = actions.get(snake.id)
            if d is None:
                safe = self.safe_moves(snake)
                d = random.choice(safe) if safe else "down"
            nx, ny = _apply(snake.head[0], snake.head[1], d)
            new_heads[snake.id] = (nx, ny)
            snake.body.insert(0, (nx, ny))
            snake.health -= 1

        ate: typing.Set[str] = set()
        for snake in self.alive_snakes():
            if new_heads[snake.id] in self.food:
                self.food.discard(new_heads[snake.id])
                snake.health = 100
                snake.length += 1
                ate.add(snake.id)

        for snake in self.alive_snakes():
            if snake.id not in ate:
                snake.body.pop()

        for snake in self.alive_snakes():
            if snake.head in self.hazards:
                snake.health -= self.hazard_dmg

        for snake in self.alive_snakes():
            hx, hy = snake.head
            if not (0 <= hx < self.width and 0 <= hy < self.height):
                snake.alive = False
            elif snake.health <= 0:
                snake.alive = False

        all_bodies: typing.Set[typing.Tuple[int, int]] = set()
        for snake in self.alive_snakes():
            for seg in snake.body[1:]:
                all_bodies.add(seg)
        for snake in self.alive_snakes():
            if snake.head in all_bodies:
                snake.alive = False

        head_groups: typing.Dict[typing.Tuple[int, int], typing.List[SnakeState]] = {}
        for snake in self.alive_snakes():
            head_groups.setdefault(snake.head, []).append(snake)
        for pos, group in head_groups.items():
            if len(group) > 1:
                max_len = max(s.length for s in group)
                for snake in group:
                    if snake.length <= max_len:
                        snake.alive = False

        while len(self.food) < self.min_food:
            for _ in range(20):
                x = random.randrange(self.width)
                y = random.randrange(self.height)
                if (x, y) not in self.food:
                    self.food.add((x, y))
                    break


# Terminal evaluation

def _evaluate(sim: GameSim) -> float:
    me = sim.my_snake()
    if me is None or not me.alive:
        return 0.0
    alive = sim.alive_snakes()
    if len(alive) == 1:
        return 1.0
    total    = len(sim.snakes)
    n_dead   = total - len(alive)
    survival = n_dead / max(1, total - 1)
    max_opp  = max((s.length for s in alive if s.id != sim.my_id), default=1)
    length_b = min(0.1, 0.1 * (me.length / max(1, max_opp) - 1))
    return min(1.0, max(0.0, survival * 0.5 + 0.5 + length_b - 0.5))



# -------- MCTS Node ------------

class Node:
    __slots__ = ("game", "parent", "move", "children",
                 "visits", "wins", "untried_moves", "heuristic_score")

    def __init__(self, game: GameSim, parent=None, move=None):
        self.game    = game
        self.parent  = parent
        self.move    = move
        self.children: typing.List["Node"] = []
        self.visits  = 0
        self.wins    = 0.0
        me = game.my_snake()
        self.untried_moves = game.safe_moves(me) if (me and me.alive) else []
        self.heuristic_score = 0.0
        

    def ucb1(self) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.wins / self.visits
        exploration  = UCB_C * math.sqrt(math.log(self.parent.visits) / self.visits)
        bias         = BIAS_C * self.heuristic_score / (self.visits + 1)
        return exploitation + exploration + bias

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

    def is_terminal(self) -> bool:
        return self.game.is_terminal()

    def best_child(self) -> "Node":
        return max(self.children, key=lambda n: n.ucb1())

    def expand(self, current_gs: typing.Dict) -> "Node":
        direction = self.untried_moves.pop(
            random.randrange(len(self.untried_moves))
        )
        new_game = self.game.copy()
        actions = {}
        for snake in new_game.alive_snakes():
            if snake.id == new_game.my_id:
                actions[snake.id] = direction
            else:
                safe = list(new_game.safe_moves(snake))
                actions[snake.id] = random.choice(safe) if safe else "down"
        new_game.step(actions)
        child = Node(new_game, parent=self, move=direction)

        # Use the CURRENT node's game state (not root) for heuristic scoring
        board = current_gs["board"]
        occupied = {
            (seg["x"], seg["y"])
            for snake in board["snakes"]
            for seg in snake["body"]
        }
        hazards = {(h["x"], h["y"]) for h in board.get("hazards", [])}
        damage = int(
            current_gs.get("game", {})
                    .get("ruleset", {})
                    .get("settings", {})
                    .get("hazardDamagePerTurn", 14)
        )
        raw_scores = [
            _evaluate_move(current_gs, m, occupied, hazards, damage)
            for m in (self.untried_moves + [direction])  # all siblings + this child
        ]
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        spread = max_s - min_s if max_s != min_s else 1.0
        raw = _evaluate_move(current_gs, direction, occupied, hazards, damage)
        child.heuristic_score = (raw - min_s) / spread  # normalized [0, 1]

        self.children.append(child)
        return child

    def rollout(self) -> float:
        sim   = self.game.copy()
        depth = 0
        while not sim.is_terminal() and depth < MAX_DEPTH:
            actions = {}
            for s in sim.alive_snakes():
                safe = sim.safe_moves(s)
                actions[s.id] = random.choice(safe) if safe else "down"
            sim.step(actions)
            depth += 1
        return _evaluate(sim)

    def backpropagate(self, result: float) -> None:
        self.visits += 1
        self.wins   += result
        if self.parent:
            self.parent.backpropagate(result)

# -------- MCTS Move Function ------------

def choose_mcts_progressive_bias_move(game_state: typing.Dict) -> str:
    """Run MCTS and return the best move direction."""
    root_game = GameSim(game_state)
    root      = Node(root_game)

    if not root.untried_moves:
        return "down"

    deadline   = time.time() + TIME_LIMIT_MS / 1000.0
    iterations = 0

    while time.time() < deadline:
        node = root
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()
        if not node.is_terminal() and not node.is_fully_expanded():
            # Reconstruct current game state from this node's sim, from our snake's POV
            current_gs = _sim_to_game_state(node.game, node.game.my_id)
            node = node.expand(current_gs=current_gs)
        result = node.rollout()
        node.backpropagate(result)
        iterations += 1

    if not root.children:
        me = root_game.my_snake()
        return random.choice(root_game.safe_moves(me)) if me else "down"

    best = max(root.children, key=lambda n: n.visits)
    print(
        f"MCTS: {iterations} sims | "
        f"best={best.move} "
        f"visits={best.visits} "
        f"winrate={best.wins / best.visits:.2f}"
    )
    
    print(f"  => chosen: {best.move}")
    return best.move


def _sim_to_game_state(sim: GameSim, pov_id: str) -> dict:
    snakes_list = []
    you_dict = None
    for s in sim.snakes:
        if not s.alive:
            continue
        entry = {
            "id": s.id,
            "body": [{"x": x, "y": y} for x, y in s.body],
            "health": s.health,
            "length": s.length,
        }
        if s.id == pov_id:
            you_dict = entry
        else:
            snakes_list.append(entry)

    if you_dict:
        snakes_list.insert(0, you_dict)

    you = you_dict or snakes_list[0]

    return {
        "turn": sim.turn,
        "you": you,
        "board": {
            "width": sim.width,
            "height": sim.height,
            "food": [{"x": x, "y": y} for x, y in sim.food],
            "hazards": [{"x": x, "y": y} for x, y in sim.hazards],
            "snakes": snakes_list,
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