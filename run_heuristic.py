import typing
from heuristic_agent import choose_heuristic_move


# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
def info() -> typing.Dict:
    print("INFO")

    return {
        "apiversion": "1",
        "author": "Jessica",
        "color": "#68C5CC",
        "head": "default",
        "tail": "default",
    }


# start is called when your Battlesnake begins a game
def start(game_state: typing.Dict):
    print("GAME START")


# end is called when your Battlesnake finishes a game
def end(game_state: typing.Dict):
    print("GAME OVER\n")


# move is called on every turn and returns your next move
# Valid moves are "up", "down", "left", or "right"
def move(game_state: typing.Dict) -> typing.Dict:
    
    # heuristic move
    next_move = choose_heuristic_move(game_state)

    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}


# start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})
