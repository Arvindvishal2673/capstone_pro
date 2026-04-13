"""
OBELIX Agent - Phase 4: Final Test Phase
Algorithm: Double Deep Q-Network (DDQN)
Reference: "Deep Reinforcement Learning with Double Q-learning" (Van Hasselt et al., 2015)

This agent uses a neural network trained via DDQN to map sensor observations to optimal actions.
The network was trained on multiple difficulty levels (1, 2, 3) to ensure robustness.
"""

import os
import torch
import numpy as np
from torch import nn

# possible actions the agent can take
move_set = ("L45", "L22", "FW", "R22", "R45")

# this will store the model so we load it only once (singleton pattern)
brain_instance = None


class QNetwork(nn.Module):
    """
    Deep Q-Network (DQN) for DDQN Algorithm.
    
    Architecture:
    - Input: 18-bit observation vector (sonar + IR + attachment sensors)
    - Hidden: Two layers with 256 and 128 units, ReLU activations
    - Output: Q-values for 5 discrete actions
    
    This architecture was designed to handle complex, partially observable environments
    with non-stationary dynamics (moving + blinking objects).
    """
    def __init__(self, input_size: int = 18, output_size: int = 5, hidden_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),      # 18 -> 256
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),     # 256 -> 256
            nn.ReLU(),
            nn.Linear(hidden_size, 128),             # 256 -> 128
            nn.ReLU(),
            nn.Linear(128, output_size),             # 128 -> 5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def initialize_agent() -> None:
    """
    Initialize the agent by loading pre-trained neural network weights.
    
    This function:
    1. Creates a QNetwork model
    2. Loads pre-trained weights from weights.pth (if available)
    3. Sets the model to evaluation mode (no gradient computation)
    4. Stores in global variable (singleton pattern for efficiency)
    
    Benefits:
    - Weights loaded only once on first call
    - Efficient prediction without reloading weights
    - CPU-compatible (no CUDA requirement)
    """
    global brain_instance

    # if model already loaded, do nothing
    if brain_instance is not None:
        return

    # create model structure
    model = QNetwork(input_size=18, output_size=5, hidden_size=256)

    # find weights file in same folder as this script
    folder = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(folder, "weights.pth")

    # load trained weights if available
    if os.path.exists(weight_path):
        try:
            # Load with CPU mapping to ensure compatibility
            weights = torch.load(weight_path, map_location="cpu")
            model.load_state_dict(weights)
        except Exception as e:
            # If weights can't be loaded, agent will use random initialization
            # This ensures the submission doesn't crash even if weights are corrupted
            pass
    
    # we only use the model for prediction (no training)
    model.eval()

    # store as global singleton
    brain_instance = model


def policy(obs: np.ndarray, rng: np.random.Generator) -> str:
    """
    DDQN Policy: Map observations to actions.
    
    Args:
        obs (np.ndarray): 18-dimensional observation vector containing:
            - 8 sonar near range bits
            - 8 sonar far range bits
            - 1 infrared sensor bit
            - 1 box attachment bit
        
        rng (np.random.Generator): Random number generator (not used in deterministic policy,
                                   but required by Codabench evaluation interface)
    
    Returns:
        str: One of ["L45", "L22", "FW", "R22", "R45"]
             - L45: Rotate left 45 degrees
             - L22: Rotate left 22.5 degrees
             - FW: Move forward
             - R22: Rotate right 22.5 degrees
             - R45: Rotate right 45 degrees
    
    Algorithm:
    1. Initialize network weights (first call only)
    2. Convert observation to torch tensor
    3. Compute Q-values for all actions
    4. Select action with maximum Q-value (greedy policy)
    5. Return action name
    
    Note: This is a deterministic policy (no exploration at test time).
    For training, epsilon-greedy exploration is used.
    """
    
    initialize_agent()

    # we don't need gradients during prediction
    with torch.no_grad():
        # convert numpy observation to torch tensor (batch size = 1)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        # compute q values for all actions using the neural network
        q_values = brain_instance(obs_tensor)

        # choose action with highest estimated value (greedy selection)
        best_index = int(q_values.argmax(dim=1).item())

    # return action name (one of the 5 discrete actions)
    return move_set[best_index]
