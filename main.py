from Environment.MazeEnv import MazeGame, GameWrapper

game = GameWrapper(600, 600, True)

while True:
    game.step([
        False,
        False,
        False,
        False,
        False,
        True,
    ])

# game = MazeGame(width=800, height=600)

# if __name__ == "__main__":
#     pyglet.app.run()
