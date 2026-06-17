# Import:
# -------
import random
import torch.nn as nn
import torch.nn.functional as F


# Deep Q-Network:
# ---------------
class Qnet(nn.Module):
    # successful arch (experiment 3):
    # def __init__(self, dim_actions, dim_states): # i can decide my own architecture 
    #     super(Qnet, self).__init__()
    #     self.fc1 = nn.Linear(dim_states, 64)
    #     self.fc2 = nn.Linear(64, 64)
    #     self.fc3 = nn.Linear(64, dim_actions)

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
        
    #     return x
    
    # successful arch (experiment 4):
    def __init__(self, dim_actions, dim_states): # i can decide my own architecture 
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(dim_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, dim_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
    
    def sample_action(self, observation, epsilon, dim_actions):
        a = self.forward(observation)
        
        #! Exploration
        if random.random() < epsilon:
            return random.randint(0, dim_actions - 1)
        
        #! Exploitation
        else : 
            return a.argmax().item()
