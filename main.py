from Environment.MazeEnv import GameWrapper, multiGames

# como correr multithreaded:

games = multiGames(10, 10, 5, 3)

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

from Environment.Models import CNNPPOPolicy, train_ppo, test_policy, sample_actions

import torch



# Initialize the game wrapper
width = 64
height = 64
game = GameWrapper(width, height, True)

# Initialize the model
model = CNNPPOPolicy((3, height, width), 6)

"""
obs, reward, done = game.reset()
episode = 0
step = 0
# Run the game
while episode < 100:
    step += 1

    # Get the action from the model
    action_probs, _ = model(obs)

    # Convert the action probabilities to actions
    action, _, _ = sample_actions(action_probs)

    # Step through the game
    obs, reward, done = game.step(action[0].bool().tolist())

    print(f"Episode: {episode}, Step: {step}, Reward: {reward}")

    if done:
        game.reset()
        episode += 1
        step = 0

game.close()
"""

# Device to run the model on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to device
model.to(device)

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train_ppo(game, model, optimizer, epochs=100,
          steps_per_epoch=256, clip_eps=0.2)

game = GameWrapper(width, height, True)

# Test the model
test_policy(game, model, episodes=5)
