import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CNNPPOPolicy(nn.Module):
    def __init__(self, input_shape, action_dim):
        super().__init__()
        C, H, W = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(C, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            n_flat = self.conv(dummy).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(n_flat, action_dim),
            nn.Sigmoid()  # probabilities for each binary action
        )
        self.critic = nn.Linear(n_flat, 1)

    def forward(self, x):
        x = self.conv(x)
        return self.actor(x), self.critic(x)

def sample_actions(probs):
    dist = torch.distributions.Bernoulli(probs)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    return actions, log_probs.sum(dim=1), dist

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Compute Generalized Advantage Estimation (GAE) for the given rewards.

    The GAE is a way to compute the advantage of each state in a trajectory, taking
    into account the rewards and the values of the states. The GAE is computed as:

    GAE = r + gamma * (1 - done) * V(s') - V(s)

    where r is the reward, V(s) is the value of the state, gamma is the discount factor,
    and done is a boolean indicating whether the state is terminal.

    :param rewards: list of rewards
    :param values: list of values
    :param dones: list of booleans indicating whether the state is terminal
    :param gamma: discount factor
    :param lam: lambda parameter for GAE
    :return: list of GAE for each state
    """
    advantages = []
    gae = 0
    values = values + [0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages

def train_ppo(env, policy, optimizer, epochs=10, steps_per_epoch=2048, clip_eps=0.2):
    for epoch in range(epochs):
        obs_list, actions_list, log_probs_list, rewards, values, dones = [], [], [], [], [], []

        obs, _, _ = env.reset()

        for _ in range(steps_per_epoch):
            obs_tensor = obs.unsqueeze(0)
            with torch.no_grad():
                probs, value = policy(obs_tensor)
            action, log_prob, _ = sample_actions(probs)
            
            next_obs, reward, done = env.step(action[0].bool().tolist())

            obs_list.append(obs)
            actions_list.append(action.squeeze(0))
            log_probs_list.append(log_prob)
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)

            obs = env.reset() if done else next_obs

        # Convert to tensors
        obs_batch = torch.stack(obs_list)
        actions_batch = torch.stack(actions_list)
        log_probs_old = torch.stack(log_probs_list)
        returns = compute_gae(rewards, values, dones)
        advantages = torch.tensor(returns) - torch.tensor(values)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update
        for _ in range(4):  # PPO epochs
            probs, values_pred = policy(obs_batch)
            dist = torch.distributions.Bernoulli(probs)
            log_probs = dist.log_prob(actions_batch).sum(dim=1)

            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values_pred.squeeze(), torch.tensor(returns))

            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[EPOCH {epoch}] Total reward: {sum(rewards):.2f}")

def test_policy(env, policy, episodes=5):
    policy.eval()
    for ep in range(episodes):
        obs, _, _ = env.reset()
        total_reward = 0
        while True:
            obs_tensor = obs.unsqueeze(0)
            with torch.no_grad():
                probs, _ = policy(obs_tensor)
            actions = (probs > 0.5).int()[0].tolist()
            obs, reward, done = env.step(actions)
            total_reward += reward
            if done:
                break
        print(f"[TEST] Episode {ep} - Total Reward: {total_reward:.2f}")