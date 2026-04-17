import os
import torch
import numpy as np
from torch import nn

# possible actions the agent can take
move_set = ("L45", "L22", "FW", "R22", "R45")

# this will store the model so we load it only once
brain_instnce = None


class Move_Predictor(nn.Module):
    # this class defines the neural network used by the agent
    def __init__(self, input_size=18, output_size=5, hidden_size=128):
        super().__init__()

        # simple feed forward neural network
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


def initialize_agent():
    
    #this function creates the model and loads weights if available
    global brain_instnce

    # if model already loaded, do nothing
    if brain_instnce is not None:
        return

    # create model structure
    model = Move_Predictor(input_size=18, output_size=5, hidden_size=128)

    # find weights file in same folder
    folder = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(folder, "weights.pth")

    if os.path.exists(weight_path):
        try:
            # load trained weights
            weights = torch.load(weight_path, map_location="cpu")
            model.load_state_dict(weights)
            print("pretrained weights loaded successfully")
        except Exception as e:
            print("could not load weights:", e)
    else:
        print("weights file not found, agent will act randomly")

    # we only use the model for prediction
    model.eval()

    brain_instnce = model


def policy(obs: np.ndarray, rng) -> str:
    #this function decides the next move of the agent
    
    initialize_agent()

    # we don't need gradients during prediction
    with torch.no_grad():

        # convert numpy observation to torch tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        # get q values for all actions
        q_values = brain_instnce(obs_tensor)

        # choose action with highest value
        best_index = int(q_values.argmax(dim=1).item())

    # return action name
    return move_set[best_index]
