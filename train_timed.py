"""
Quick Level 1 Training - 100 Episodes with Timing
"""
import os
import sys
import time
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CS780-OBELIX'))
from obelix import OBELIX

ACTIONS = ("L45", "L22", "FW", "R22", "R45")
NUM_ACTIONS = len(ACTIONS)

class QNetwork(nn.Module):
    def __init__(self, obs_dim=18, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DDQNAgent:
    def __init__(self, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=1e-3):
        self.q_network = QNetwork()
        self.target_network = QNetwork()
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 32
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_counter = 0
        self.target_update_freq = 100

    def get_action(self, obs, rng):
        """Get action with epsilon-greedy exploration"""
        if rng.random() < self.epsilon:
            return rng.integers(0, NUM_ACTIONS)
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            q_vals = self.q_network(obs_t)
            return q_vals.argmax(dim=1).item()

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.append((obs, action, reward, next_obs, done))

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[i] for i in batch])
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)


def train_level1(num_episodes=100):
    env = OBELIX(scaling_factor=5, arena_size=500, max_steps=300, difficulty=0)
    agent = DDQNAgent()
    
    print("\n" + "="*70)
    print(f"TRAINING LEVEL 1 - {num_episodes} EPISODES")
    print("="*70)
    print(f"Start Time: {time.strftime('%H:%M:%S')}")
    print("="*70 + "\n")
    
    start_time = time.time()
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset(seed=episode)
        episode_reward = 0.0
        step = 0
        
        while step < 300:
            rng = np.random.default_rng(episode * 1000 + step)
            action_idx = agent.get_action(obs, rng)
            action = ACTIONS[action_idx]
            
            obs, reward, done = env.step(action, render=False)
            episode_reward += reward
            
            agent.store_transition(obs, action_idx, reward, obs, done)
            agent.train_step()
            
            step += 1
            if done:
                break
        
        agent.decay_epsilon()
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed
            eta = (num_episodes - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0
            
            print(f"Episode {episode + 1:3d}/{num_episodes} | "
                  f"Reward: {episode_reward:8.1f} | "
                  f"Avg (10): {avg_reward:8.1f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"ETA: {eta:.0f}s")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    return agent, episode_rewards, total_time


if __name__ == "__main__":
    agent, rewards, total_seconds = train_level1(num_episodes=100)
    
    submission_dir = os.path.dirname(__file__)
    weights_path = os.path.join(submission_dir, "weights_trained_fresh.pth")
    agent.save(weights_path)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Total Episodes:    100")
    print(f"Total Time:        {total_seconds:.1f} seconds ({total_seconds/60:.1f} minutes)")
    print(f"Time per Episode:  {total_seconds/100:.2f} seconds")
    print(f"Episodes per Min:  {100/(total_seconds/60):.1f}")
    print(f"\nWeights saved to:  {weights_path}")
    print(f"File size:         {os.path.getsize(weights_path)/1024:.1f} KB")
    print("="*70)
