from Environment.MazeEnv import GameWrapper, multiGames

# como correr multithreaded:

width = 10
height = 10

batches = 5
size = 3

games = multiGames(width, height, batches, size)

a = games.step(
    [
        [False, False, False, False, False, False,],
        [False, False, False, False, False, False,],
        [False, False, False, False, False, False,],
        [False, False, False, False, False, False,],
        [False, False, False, False, False, False,],

        [False, False, False, False, False, False,],
        [False, False, False, False, False, False,],
        [False, False, False, False, False, False,],
        [False, False, False, False, False, False,],
        [False, False, False, False, False, False,],

        [False, False, False, False, False, False,],
        [False, False, False, False, False, False,],
        [False, False, False, False, False, False,],
        [False, False, False, False, False, False,],
        [False, False, False, False, False, False,],
    ]
)

print(a)