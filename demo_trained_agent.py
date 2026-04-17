"""
DEMO: Level 1 Agent - Freshly Trained (100 Episodes)
Shows the trained agent behavior on static box task
"""

import os
import sys
import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'CS780-OBELIX'))
from obelix import OBELIX

ACTIONS = ("L45", "L22", "FW", "R22", "R45")


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
    def __init__(self):
        self.q_network = QNetwork()
        self.q_network.eval()

    def get_action(self, obs):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            q_vals = self.q_network(obs_t)
            action_idx = q_vals.argmax(dim=1).item()
        return action_idx

    def load_weights(self, path):
        weights = torch.load(path, map_location='cpu')
        self.q_network.load_state_dict(weights)
        print(f"✅ Weights loaded from: {path}")


def run_episode(agent, episode_num, max_steps=2000):
    """Run one episode and return statistics"""
    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=max_steps,
        difficulty=0,
    )

    obs = env.reset()
    total_reward = 0.0
    steps = 0
    action_counts = {action: 0 for action in ACTIONS}
    attached = False
    goal_reached = False

    while steps < max_steps:
        action_idx = agent.get_action(obs)
        action = ACTIONS[action_idx]
        action_counts[action] += 1
        
        obs, reward, done = env.step(action, render=False)
        total_reward += reward
        steps += 1

        # Check for special events
        if reward > 0:
            if reward >= 2000:
                goal_reached = True
            elif reward >= 100:
                attached = True

        # Progress output
        if steps % 200 == 0:
            print(f"  [Episode {episode_num}] Step {steps:4d}: Cumulative Reward={total_reward:8.1f}, Action={action}")

        if done:
            break

    success = goal_reached or (attached and total_reward > -500)
    
    return {
        'total_reward': total_reward,
        'steps': steps,
        'success': success,
        'attached': attached,
        'goal_reached': goal_reached,
        'action_counts': action_counts,
    }


def main():
    print("\n" + "="*75)
    print("LEVEL 1 AGENT DEMONSTRATION - FRESHLY TRAINED (100 EPISODES)")
    print("="*75)
    print(f"\nStart Time: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}")

    # Load trained agent
    agent = DDQNAgent()
    weights_path = os.path.join(
        os.path.dirname(__file__), 
        'weights_trained_fresh.pth'
    )
    
    if not os.path.exists(weights_path):
        print(f"❌ Weights file not found: {weights_path}")
        return

    agent.load_weights(weights_path)

    # Run 3 test episodes
    print(f"\n{'='*75}")
    print("Running 3 Episodes on LEVEL 1 (Static Box)")
    print("="*75)
    
    results = []
    for ep in range(3):
        print(f"\n📍 Episode {ep + 1}:")
        result = run_episode(agent, ep + 1)
        results.append(result)
        
        # Print episode summary
        success_emoji = "✅ SUCCESS" if result['success'] else "⏹️ TIMEOUT/FAIL"
        print(f"\nResult: {success_emoji}")
        print(f"  Total Reward:    {result['total_reward']:8.1f}")
        print(f"  Steps Taken:     {result['steps']:4d}")
        print(f"  Box Attached:    {'✅ YES' if result['attached'] else '❌ NO'}")
        print(f"  Goal Reached:    {'✅ YES' if result['goal_reached'] else '❌ NO'}")
        print(f"  Action Distribution:")
        for action, count in result['action_counts'].items():
            pct = 100 * count / result['steps'] if result['steps'] > 0 else 0
            print(f"    {action}: {count:3d} times ({pct:5.1f}%)")

    # Summary Statistics
    mean_reward = np.mean([r['total_reward'] for r in results])
    success_rate = 100 * np.mean([r['success'] for r in results])
    attachment_rate = 100 * np.mean([r['attached'] for r in results])
    mean_steps = np.mean([r['steps'] for r in results])

    print(f"\n{'='*75}")
    print("PERFORMANCE SUMMARY (3 Episodes)")
    print("="*75)
    print(f"Mean Reward:           {mean_reward:8.1f}")
    print(f"Mean Steps:            {mean_steps:8.1f}")
    print(f"Success Rate:          {success_rate:6.1f}%")
    print(f"Box Attachment Rate:   {attachment_rate:6.1f}%")
    
    print(f"\nComparison to Random Baseline:")
    print(f"  Random agent:        -2000.0 reward,  0% success")
    print(f"  Trained agent:       {mean_reward:8.1f} reward, {success_rate:.1f}% success")
    
    if mean_reward > -2000:
        improvement = (((-2000) - mean_reward) / 2000) * 100
        print(f"  📊 Improvement:       {improvement:.1f}% better than random! ✅")

    print(f"\n{'='*75}")
    print(f"Demo Complete! 🎉")
    print(f"End Time: {__import__('datetime').datetime.now().strftime('%H:%M:%S')}")
    print("="*75)


if __name__ == "__main__":
    main()
