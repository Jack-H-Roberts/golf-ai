import torch
import torch.nn as nn

class GolfModel(nn.Module):
    def __init__(self, input_dim=737, output_dim=10):
        super(GolfModel, self).__init__()
        
        # The "Shared" brain that understands the game state
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # The Actor: Outputs "Logits" for our 10 possible moves
        self.actor = nn.Linear(256, output_dim)
        
        # The Critic: Outputs a single value (Win Probability)
        self.critic = nn.Linear(256, 1)

    def forward(self, x, mask=None):
        shared_out = self.shared_layers(x)
        
        # Move probabilities
        logits = self.actor(shared_out)
        
        # Apply the action mask by setting illegal moves to a very large negative number
        if mask is not None:
            logits = logits + (mask.log()) # Moves where mask=0 become -inf
            
        # Win probability (clamped between 0 and 1)
        value = torch.sigmoid(self.critic(shared_out))
        
        return logits, value