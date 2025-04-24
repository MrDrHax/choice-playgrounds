from Environment.MazeEnv import MazeGame, GameWrapper
import random
import torch

# Initialize the game wrapper
game = GameWrapper(600, 600, False)

step = 0
# Run the game
while True:
    #Get a set of random actions
    actions = [random.random() > 0.5 for _ in range(6)]

    #print(step, actions)
    step += 1

    # Step through the game
    obs, reward, done = game.step(actions)

    if done:
        game.reset()