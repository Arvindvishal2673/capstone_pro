"""
DDQN Training Script - PHASE 3 VARIANT B: Aggressive Learning + Adaptive Exploration

Improvements from baseline:
1. Learning Rate: 1e-3 → 2e-3 (faster initial learning)
2. Epsilon Start: 1.0 → 0.8 (start with less random, more deterministic)
3. Target Update: 100 → 50 (update target network more frequently)
4. Reward Scaling: Apply reward clipping to stabilize learning
5. Episodes: 500 → 800 (much longer training)
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
OBS_DIM = 18


class QNetwork(nn.Module):
    """Q-Network with residual-like skip connection concepts."""
    def __init__(self, obs_dim: int = 18, n_actions: int = 5, hidden_dim: int = 256):
        super().__init__()
        # Wider network
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience Replay Buffer with prioritization hints."""
    def __init__(self, capacity: int = 100000):
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
    """Double Deep Q-Network Agent - Variant B (Aggressive)."""
    def __init__(
        self,
        obs_dim: int = 18,
        n_actions: int = 5,
        hidden_dim: int = 256,
        lr: float = 2e-3,  # Aggressive: higher learning rate
        gamma: float = 0.99,
        epsilon: float = 0.8,  # Start with less randomness
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.992,
        target_update_freq: int = 50,  # Update more frequently
        buffer_size: int = 200000,
        batch_size: int = 96,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.q_network = QNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_network = QNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.update_count = 0
        self.loss_history = []

    def get_action(self, state: np.ndarray, rng: np.random.Generator) -> int:
        """Epsilon-greedy action selection."""
        if rng.random() < self.epsilon:
            return rng.integers(self.n_actions)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one DDQN training step with reward clipping."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Clip rewards to [-1, 1] to stabilize learning
        rewards_t = torch.clamp(rewards_t / 1000.0, -1, 1)

        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

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

    def get_stats(self):
        if len(self.loss_history) < 100:
            return {"avg_loss": np.mean(self.loss_history)}
        return {"avg_loss": np.mean(self.loss_history[-100:])}


def train(num_episodes: int = 800, max_steps: int = 500):
    """Train DDQN agent - Variant B with aggressive learning."""
    print("\n" + "="*70)
    print("PHASE 3 VARIANT B: AGGRESSIVE LEARNING + LONGER TRAINING")
    print("="*70)
    
    env = OBELIX(
        scaling_factor=3,
        arena_size=500,
        max_steps=max_steps,
        wall_obstacles=True,
        difficulty=2,
        box_speed=2,
    )

    agent = DDQNAgent(
        obs_dim=18,
        n_actions=5,
        hidden_dim=256,
        lr=2e-3,             # Aggressive learning
        gamma=0.99,
        epsilon=0.8,         # Less random startup
        epsilon_min=0.01,
        epsilon_decay=0.992,
        target_update_freq=50,  # Update frequently
        buffer_size=200000,
        batch_size=96,
        device="cpu",
    )

    episode_rewards = []
    
    print("Improvements:")
    print("  - Learning Rate: 2e-3 (faster learning)")
    print("  - Target Update: 50 steps (frequent updates)")
    print("  - Epsilon Start: 0.8 (less random)")
    print("  - Reward Scaling: Applied")
    print("  - Wider Network: 256 layers")
    print("  - Episodes: 800 (much longer training)")
    print("="*70)

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

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            stats = agent.get_stats()
            print(
                f"Episode {episode + 1:4d}/{num_episodes} | "
                f"Reward: {episode_reward:8.1f} | "
                f"Avg (10): {avg_reward:8.1f} | "
                f"Loss: {stats['avg_loss']:7.4f} | "
                f"Epsilon: {agent.epsilon:.4f}"
            )

    print("="*70)
    print(f"Training completed! Final avg: {np.mean(episode_rewards[-10:]):.1f}")
    print(f"Best avg (100-ep window): {max([np.mean(episode_rewards[i:i+100]) for i in range(0, len(episode_rewards)-100, 1)]):.1f}")
    print("="*70)

    return agent, episode_rewards


if __name__ == "__main__":
    agent, rewards = train(num_episodes=800, max_steps=500)

    submission_dir = os.path.dirname(__file__)
    weights_path = os.path.join(submission_dir, "weights_variant_b.pth")
    agent.save(weights_path)
    print(f"\nWeights saved to: {weights_path}")
