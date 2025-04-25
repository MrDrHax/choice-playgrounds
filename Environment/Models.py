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

def train_ppo(env, policy, optimizer, epochs=10, steps_per_epoch=512, clip_eps=0.2):
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

def test_policy(env, policy, episodes=5, steps_per_episode=512):
    policy.eval()
    for ep in range(episodes):
        obs, _, _ = env.reset()
        total_reward = 0
        step = 0
        while True:
            obs_tensor = obs.unsqueeze(0)
            with torch.no_grad():
                probs, _ = policy(obs_tensor)
            actions = (probs > 0.5).int()[0].tolist()
            obs, reward, done = env.step(actions)
            total_reward += reward
            if done or step >= steps_per_episode:
                break
            step += 1
        print(f"[TEST] Episode {ep} - Total Reward: {total_reward:.2f}")


# META RL


# ======= 1. ENTORNO: RestlessBandit de 2 brazos =======
class RestlessBandit:
    def __init__(self, volatility=0.1):
        # Inicializamos p para el Brazo 0 entre 0.1 y 0.9.
        # El brazo 1 tendrá probabilidad = 1 - p.
        p = np.random.uniform(0.1, 0.9)
        self.probs = [p, 1 - p]
        self.volatility = volatility

    def pull(self, action):
        # Cada vez que se llama a pull, se actualizan las probabilidades:
        noise = np.random.randn() * self.volatility
        new_p = self.probs[0] + noise
        # Limitamos p entre 0.1 y 0.9 para evitar extremos.
        new_p = np.clip(new_p, 0.1, 0.9)
        self.probs[0] = new_p
        self.probs[1] = 1 - new_p

        # Retorna 1 con probabilidad de self.probs[action] o 0.
        return 1 if np.random.rand() < self.probs[action] else 0

# ======= 2. MODELO: Agente PPO con LSTM =======
class PPOAgent(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_actions=2):
        """
        input_size: dimensión del vector de entrada. Se compone de:
                    - Acción previa en formato one-hot (2 dimensiones)
                    - Recompensa previa (1 dimensión)
                    - Timestep normalizado (1 dimensión)
                    Total: 2+1+1 = 4.
        """
        super(PPOAgent, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.policy_head = nn.Linear(hidden_size, num_actions)  # Produce los logits
        self.value_head = nn.Linear(hidden_size, 1)             # Estima el valor

    def reset_state(self):
        self.hx = torch.zeros(1, self.hidden_size)
        self.cx = torch.zeros(1, self.hidden_size)

    def forward(self, x):
        # x es de tamaño (1, input_size)
        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        logits = self.policy_head(self.hx)
        value = self.value_head(self.hx)
        return logits, value

# ======= 3. Crear la entrada del agente =======
def get_input(last_action, last_reward, timestep, num_actions=2):
    """
    Construye un vector de entrada para el LSTM a partir de:
      - La acción previa (one-hot de dimensión 2)
      - La recompensa previa (dimensión 1)
      - El timestep normalizado (dimensión 1)
    """
    action_one_hot = F.one_hot(torch.tensor([last_action]), num_classes=num_actions).float()
    reward_tensor = torch.tensor([[last_reward]], dtype=torch.float32)
    timestep_tensor = torch.tensor([[timestep / 10.0]], dtype=torch.float32)  # Normalizamos para evitar valores altos
    x = torch.cat([action_one_hot, reward_tensor, timestep_tensor], dim=1)
    # concatena los tensores de manera horizontal (dim = 1)
    return x

# ======= 4. HIPERPARÁMETROS DE PPO =======
gamma = 0.99           # Factor de descuento
clip_epsilon = 0.2     # Parámetro de recorte PPO (clip)
ppo_epochs = 4         # Número de épocas por actualización
lr = 0.009             # Tasa de aprendizaje

agent = PPOAgent()
optimizer = optim.Adam(agent.parameters(), lr=lr)

num_episodes = 1000    # Cantidad total de episodios
episode_length = 5     # Pasos por episodio

# ======= 5. CICLO DE ENTRENAMIENTO CON PPO =======
for episode in range(num_episodes):
    # Usamos el entorno RestlessBandit con volatilidad definida (0.1 para probar en este caso)
    env = RestlessBandit(volatility=0.1)
    agent.reset_state()
    
    # Listas para almacenar la trayectoria del episodio
    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []
    
    # Inicializamos: se parte de una acción por defecto (0) y recompensa 0 para el primer paso.
    last_action = 0
    last_reward = 0.0
    
    # Recorrido del episodio
    for t in range(episode_length):
        x = get_input(last_action, last_reward, t)
        logits, value = agent(x)
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Guardamos los datos de la transición
        states.append(x)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        
        # Ejecutamos la acción en el entorno y obtenemos la recompensa.
        reward = env.pull(action.item())
        rewards.append(reward)
        
        last_action = action.item()
        last_reward = reward
    
    # ======= 5.1. Calcular los RETURNS y las VENTAJAS =======
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
    values = torch.cat(values)
    advantages = returns - values.detach()
    
    old_log_probs = torch.cat(log_probs).detach()
    
    # ======= 5.2. Actualización PPO sobre la trayectoria recogida =======
    for _ in range(ppo_epochs):
        new_log_probs = []
        new_values = []
        agent.reset_state()  # Reiniciamos el estado para reevaluar la trayectoria almacenada
        
        # Se reevalúa cada estado almacenado en la trayectoria
        for i, x in enumerate(states):
            logits, value = agent(x)
            probs = F.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs.append(dist.log_prob(actions[i]))
            new_values.append(value)
        new_log_probs = torch.cat(new_log_probs)
        new_values = torch.cat(new_values)
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(new_values, returns)
        loss = policy_loss + 0.5 * value_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (episode+1) % 100 == 0:
        total_reward = sum(rewards)
        print(f"Episode {episode+1}, Total Reward: {total_reward}, Loss: {loss.item():.4f}")

print("Entrenamiento completado.")

# ======= 6. Evaluación del agente entrenado con PPO =======
agent.eval()  # Cambia el modelo a modo evaluación
env = RestlessBandit(volatility=0.1)  # Nuevo episodio de prueba en entorno cambiante
agent.reset_state()

last_action = 0
last_reward = 0
total_reward = 0

print("Probabilidades ocultas del entorno:", env.probs)

for t in range(5):  # Evaluamos por 5 pasos
    with torch.no_grad():
        x = get_input(last_action, last_reward, t)
        logits, _ = agent(x)
        probs = F.softmax(logits, dim=1)
        action = torch.multinomial(probs, num_samples=1).item()

    reward = env.pull(action)
    total_reward += reward
    print(f"Paso {t} | Acción: {action} | Recompensa: {reward}")
    last_action = action
    last_reward = reward

print("Recompensa total:", total_reward)
