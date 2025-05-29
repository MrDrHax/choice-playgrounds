# Maze

This project implements a reinforcement learning (RL) environment and training pipeline for a 3D maze navigation task. The environment simulates a maze where an agent must learn to navigate and make decisions, such as choosing between doors with different reward probabilities.

The project leverages PyTorch, TorchRL, and custom environment wrappers to enable scalable, parallelized training using the Proximal Policy Optimization (PPO) algorithm. The training process includes data collection, advantage estimation, policy/value updates, checkpointing, and visualization of learning progress.

## Table of Contents

-   [Features](#features)
-   [Conclusions](#conclusions)
-   [Contributors](#contributors)
-   [License](#license)

## Features

-   Custom 3D Maze Environment:

    The environment provides a 3D maze with interactive elements (doors, rewards) and supports both single and batched/multithreaded execution for efficient RL training.

-   TorchRL Integration:

    The environment is wrapped as a TorchRL-compatible class, enabling seamless use with TorchRL's data collectors, transforms, and utilities.

-   PPO Training Pipeline:

    The notebook implements a full PPO training loop, including:

    -   Parallel data collection with SyncDataCollector
    -   Advantage estimation using Generalized Advantage Estimation (GAE)
    -   Policy and value updates with gradient clipping and learning rate scheduling
    -   Periodic evaluation and checkpointing

-   Custom CNN Policy:

    A convolutional neural network is used as the policy and value function, processing visual observations from the environment.

-   Logging and Visualization:

    Training and evaluation metrics are logged and visualized live using matplotlib. Feature maps from the CNN can be visualized with TensorBoard.

-   Flexible Hyperparameters:

    The notebook exposes hyperparameters for environment size, batch size, learning rate, PPO settings, and more, allowing for easy experimentation.

-   Checkpointing and Model Reloading:

    The training process saves checkpoints and logs, and supports loading trained models for further evaluation or visualization.

## Conclusions

-   Reward System

    -   If you give it a reward for entering the door that is too big, the agent enters any door without caring at all

    -   If you give it a reward for getting close to the door, the agent will learn to get close to the door, but not necessarily enter it.

    -   If you give it the same reward for entering the good room as the bad reward for entering the bad room, the agent will learn to not risk it and does not enter any door.

-   You should not use Bernoulli, and instead use Categorical to speed up training and improve model performance.

    -   Smaller effective action space

            Bernoulli: 2⁶=64 possible combinations of button‐press subsets. Most of those (e.g. pressing forwards and backwards simultaneously) are equivalent or ignored, but the agent still has to explore them.

            Categorical: exactly 6 valid one-hot choices. That’s 6 states, not 64. Much easier to explore and assign credit.

    -   Lower gradient variance

            With a 6-class softmax, you get one log‐prob gradient per step.

            With 6 independent Bernoullis, you get 6 separate log‐probs (and effectively six “yes/no” learning signals), which adds noise to PPO’s gradient estimate.

    -   Simpler entropy bonus & tuning

            A single entropy coefficient on a 6-way categorical is straightforward.

            With Bernoulli you’d need to balance 6 separate entropy terms (or one shared coefficient), making exploration/exploitation trade-offs trickier.

## Contributors

-   Mario Ignacio Frias Piña (https://github.com/MarioFriasPina)
-   Alejandro Fernández del Valle Herrera (https://github.com/MrDrHax)
-   Oswaldo Ilhuicatzi Mendizábal (https://github.com/OswaldoMen14)

## License

This project is licensed under the [MIT License](LICENSE).
