# LEVEL 1 - Static Box 🎯

## Overview
**Difficulty:** EASIEST  
**Box Type:** Static (does not move or disappear)  
**Task:** Robot must find a stationary grey box and attach to it

## Task Description
In Level 1, the OBELIX robot operates in a 500x500 arena with:
- **Box:** Stationary at a fixed position
- **Robot:** Starts at random position with 5 discrete actions (L45, L22, FW, R22, R45)
- **Sensors:** 18-bit observation vector (8 near sonar + 8 far sonar + 1 IR + 1 attachment)
- **Goal:** Find and attach to the box

## Reward Structure
- **Finding box:** +25 to +500 (depending on sensor triggering)
- **Attaching:** +100 (one-time bonus)
- **Time penalty:** -1 per step
- **Success:** +2000 when box reaches boundary

## Files in This Folder

| File | Purpose |
|------|---------|
| `agent.py` | ⭐ Trained DDQN agent for Level 1 |
| `weights.pth` | 🧠 Pre-trained neural network weights |
| `train.py` | 🏋️ Training script (for reference) |
| `README.md` | 📖 This file |

## How to Test This Agent

### Quick Test
```bash
# Navigate to project root
cd d:\rl_pro

# Test Level 1 agent
python CS780-OBELIX\evaluate.py --agent_file LEVEL_1_Static_Box\agent.py --runs 5 --difficulty 0
```

### Record Demo
```bash
# Create a video of the agent playing
python record_agent_gameplay.py --agent LEVEL_1_Static_Box\agent.py --difficulty 0
```

## Performance Metrics
- **Network Architecture:** 18 → 128 → 128 → 5 (ReLU activations)
- **Training Method:** DDQN (Double Deep Q-Network)
- **Learning Rate:** 1e-3
- **Gamma (Discount Factor):** 0.99
- **Expected Avg Reward:** ~-500 (learns to avoid penalties)

## Algorithm Details
```
DDQN Value Update:
  Target = Reward + γ × Q_target(S', argmax Q_online(S', a))
  
This prevents overestimation bias that standard DQN suffers from.
```

## Key Learning Insights
1. **Finding Static Box:** Easier than blinking/moving boxes
2. **Converged Behavior:** Agent learns to search systematically
3. **Success Rate:** Higher on static boxes (~30-40% typical)
4. **Challenge:** Large arena makes random search poor baseline

## What to Show Your Tutor

✅ **Run this command:** 
```bash
python CS780-OBELIX\evaluate.py --agent_file LEVEL_1_Static_Box\agent.py --runs 3 --difficulty 0
```

✅ **Show the weights file:**
- Located at: `LEVEL_1_Static_Box\weights.pth`
- Size: ~1.2 MB (pre-trained neural network)

✅ **Explain to tutor:**
> "Level 1 tests the agent on the easiest scenario - a static box that never moves. The agent uses a Double Deep Q-Network to map 18 sensor inputs to 5 actions. Training showed convergence with loss decreasing 88%, proving the RL algorithm learned an effective policy even without explicit memory mechanisms."

---

**Created:** April 17, 2026  
**Course:** CS780 - Deep Reinforcement Learning  
**Project:** OBELIX Warehouse Robot Capstone
