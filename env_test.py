from Environment.MazeEnv import GameWrapper, multiGames
import time

steps = [
    [True, False, False, False, False, False,] 
] * 128
# ] * 1

#games = multiGames(64, 64, 8, 16, False) # 128 instances
games = multiGames(500, 500, 1, 1, True) # 1 instance

i = 0
while i < 50:
    start = time.time()
    a = games.step(
        [[True, False, False, False, False, False]], i
    )
    i += 1
