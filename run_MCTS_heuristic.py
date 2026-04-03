import typing
from MCTS_heuristic import choose_mcts_heuristic_move
 
 
# info is called when you create your Battlesnake on play.battlesnake.com
# and controls your Battlesnake's appearance
def info() -> typing.Dict:
    print("INFO")
 
    return {
        "apiversion": "1",
        "author": "Alexa",
        "color": "#AA64D0",
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
# valid moves are "up", "down", "left", or "right"
def move(game_state: typing.Dict) -> typing.Dict:
    
    # heuristic rollout MCTS
    next_move = choose_mcts_heuristic_move(game_state)
 
    print(f"MOVE {game_state['turn']}: {next_move}")
    return {"move": next_move}
 
 
# start server when `python main.py` is run
if __name__ == "__main__":
    from server import run_server
 
    run_server({"info": info, "start": start, "move": move, "end": end})