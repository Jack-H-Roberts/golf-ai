import torch
import torch.nn as nn

class GolfModel(nn.Module):
    def __init__(self, input_dim=737, output_dim=10):
        super(GolfModel, self).__init__()
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Actor: Outputs Logits for actions
        self.actor = nn.Linear(256, output_dim)
        
        # Critic: Outputs linear Value (Score estimation)
        # CRITICAL FIX: Removed Sigmoid so it can predict negative/large scores
        self.critic = nn.Linear(256, 1)

    def forward(self, x, mask=None):
        shared_out = self.shared_layers(x)
        
        logits = self.actor(shared_out)
        
        if mask is not None:
            # CRITICAL FIX: Use -1e9 instead of -inf (log(0))
            # PyTorch Categorical can crash with actual -inf values
            logits = logits.masked_fill(mask == 0, -1e9)
            
        value = self.critic(shared_out)
        
        return logits, value