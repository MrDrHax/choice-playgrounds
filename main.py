from Environment.MazeEnv import GameWrapper
from Environment.Models import BasicCNNModel

import torch

# Initialize the game wrapper
width = 640
height = 640
game = GameWrapper(width, height, False)
obs, reward, done = game.reset()

# Initialize the model
model = BasicCNNModel(image_size=(height, width))

episode = 0
step = 0
# Run the game
while episode < 100:
    step += 1

    # Get the action from the model
    action_probs = model(obs)
    actions = (action_probs > 0.5).int().squeeze().tolist()

    # Step through the game
    obs, reward, done = game.step(actions)

    print(f"Episode: {episode}, Step: {step}, Reward: {reward}")

    if done:
        game.reset()
        episode += 1
        step = 0

game.close()