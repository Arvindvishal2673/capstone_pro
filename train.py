"""
DDQN Training Script (FAST VERSION FOR EVALUATION)

Reduced episodes for quick testing/validation of DDQN agent.
Use this to quickly test if the implementation works correctly.
"""

import os
import sys
import random
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CS780-OBELIX'))
from obelix import OBELIX

ACTIONS = ("L45", "L22", "FW", "R22", "R45")
NUM_ACTIONS = len(ACTIONS)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int = 18, n_actions: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DDQNAgent:
    def __init__(
        self,
        obs_dim: int = 18,
        n_actions: int = 5,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.99,
        target_update_freq: int = 100,
        buffer_size: int = 50000,
        batch_size: int = 64,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.q_network = QNetwork(obs_dim, n_actions, hidden_dim)
        self.target_network = QNetwork(obs_dim, n_actions, hidden_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.update_count = 0
        self.loss_history = []

    def get_action(self, state: np.ndarray, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return rng.integers(self.n_actions)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones)

        # Current Q-values
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

        # DDQN: Online network selects, target network evaluates
        with torch.no_grad():
            next_actions = self.q_network(next_states_t).argmax(dim=1)
            next_q = self.target_network(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.loss_history.append(loss.item())
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        torch.save(self.q_network.state_dict(), filepath)


def train(num_episodes: int = 100, max_steps: int = 300):
    """Quick training for testing."""
    env = OBELIX(
        scaling_factor=3,
        arena_size=500,
        max_steps=max_steps,
        wall_obstacles=False,
        difficulty=0,
        box_speed=2,
    )

    agent = DDQNAgent(
        obs_dim=18,
        n_actions=5,
        hidden_dim=128,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.99,
        target_update_freq=100,
        buffer_size=50000,
        batch_size=32,
    )

    episode_rewards = []

    print("=" * 70)
    print("DDQN TRAINING (Quick Version)")
    print("=" * 70)

    for episode in range(num_episodes):
        obs = env.reset(seed=episode)
        episode_reward = 0.0
        done = False
        step = 0

        while not done and step < max_steps:
            rng = np.random.default_rng(episode * 1000 + step)
            action_idx = agent.get_action(obs, rng)
            action = ACTIONS[action_idx]

            obs, reward, done = env.step(action, render=False)
            episode_reward += reward

            agent.store_transition(obs, action_idx, reward, obs, done)
            agent.train_step()
            step += 1

        agent.decay_epsilon()
        episode_rewards.append(episode_reward)

        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(episode_rewards[-5:])
            print(
                f"Episode {episode + 1:3d}/{num_episodes} | "
                f"Reward: {episode_reward:8.1f} | "
                f"Avg (5): {avg_reward:8.1f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    print("=" * 70)
    print(f"Training completed!")
    print("=" * 70)

    return agent, episode_rewards


if __name__ == "__main__":
    agent, rewards = train(num_episodes=100, max_steps=300)

    submission_dir = os.path.dirname(__file__)
    weights_path = os.path.join(submission_dir, "weights.pth")
    agent.save(weights_path)
    print(f"\nWeights saved to: {weights_path}")
