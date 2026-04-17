#!/usr/bin/env python
"""
LEVEL 1: Static Box - Quick Evaluation Script
Run this to test the Level 1 trained agent
"""

import os
import sys

# Add OBELIX to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CS780-OBELIX'))

import argparse
import numpy as np
import importlib.util

from obelix import OBELIX


def load_agent(agent_path: str):
    """Load agent from file"""
    spec = importlib.util.spec_from_file_location("agent", agent_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load: {agent_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.policy


def run_episode(policy, difficulty=0, seed=None, max_steps=2000):
    """Run one evaluation episode"""
    env = OBELIX(
        scaling_factor=5,
        arena_size=500,
        max_steps=max_steps,
        wall_obstacles=False,
        difficulty=difficulty,
        box_speed=2,
        seed=seed if seed is not None else np.random.randint(0, 10000),
    )
    
    obs = env.reset()
    rng = np.random.default_rng(seed)
    
    total_reward = 0.0
    steps = 0
    
    while steps < max_steps:
        action = policy(obs, rng)
        obs, reward, done = env.step(action, render=False)
        total_reward += reward
        steps += 1
        if done:
            break
    
    return total_reward, steps


def main():
    parser = argparse.ArgumentParser(description="Evaluate Level 1 Agent (Static Box)")
    parser.add_argument('--runs', type=int, default=5, help='Number of evaluation runs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Load agent
    agent_path = os.path.join(os.path.dirname(__file__), 'LEVEL_1_Static_Box', 'agent.py')
    
    print("\n" + "="*70)
    print("LEVEL 1 - STATIC BOX - EVALUATION")
    print("="*70)
    print(f"Agent: {agent_path}")
    print(f"Difficulty: 0 (Static box - no movement, no blinking)")
    print(f"Runs: {args.runs}")
    print("="*70 + "\n")
    
    if not os.path.exists(agent_path):
        print(f"❌ Agent not found: {agent_path}")
        return
    
    try:
        policy = load_agent(agent_path)
        print("✅ Agent loaded successfully!\n")
    except Exception as e:
        print(f"❌ Error loading agent: {e}")
        return
    
    rewards = []
    steps_list = []
    successes = 0
    
    for i in range(args.runs):
        reward, steps = run_episode(policy, difficulty=0, seed=args.seed + i)
        rewards.append(reward)
        steps_list.append(steps)
        
        success = reward >= 1000
        status = "✅ SUCCESS" if success else "⏹️  FAIL"
        print(f"Run {i+1:2d}: Reward={reward:>8.1f}, Steps={steps:>4d}, {status}")
        
        if success:
            successes += 1
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - LEVEL 1 (STATIC BOX)")
    print("="*70)
    print(f"Mean Reward:      {np.mean(rewards):>8.1f} ± {np.std(rewards):.1f}")
    print(f"Mean Steps:       {np.mean(steps_list):>8.1f}")
    print(f"Success Rate:     {(successes/args.runs)*100:>8.1f}%")
    print(f"Successes:        {successes}/{args.runs}")
    print("="*70)
    
    print("\n📊 Analysis:")
    if np.mean(rewards) > -1000:
        print("   ✅ Agent shows learning - reward better than -2000 baseline")
    else:
        print("   ⚠️  Performance similar to random baseline")
    
    print("\n📝 Note: Level 1 has the simplest task (static boxes).")
    print("   Higher success rates expected compared to blinking/moving levels.")


if __name__ == '__main__':
    main()
