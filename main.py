from Environment.MazeEnv import MazeGame, GameWrapper
import pyglet

game = GameWrapper(600, 600, False)

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
