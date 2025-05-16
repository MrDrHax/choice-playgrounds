from Environment.MazeEnv import GameWrapper, multiGames
import time

steps = [
    [True, False, False, False, False, False,] 
] * 128
# ] * 1

games = multiGames(64, 64, 8, 16) # 128 instances
# games = multiGames(500, 500, 1, 1) # 1 instance

while True:
    start = time.time()
    a = games.step(
        steps
    )
    print(f"\r{time.time() - start}s             ", end="")