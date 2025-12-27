import torch
from environment import GolfEnv
from model import GolfModel

def simple_train():
    device = "cuda"
    env = GolfEnv(num_envs=1024, device=device)
    model = GolfModel().to(device)
    
    # Reset the world
    state = env.reset()
    
    # Get the "Actor" to make a guess
    mask = env.get_action_mask()
    logits, value = model(state, mask=mask)
    
    # Pick the move with the highest score
    actions = torch.argmax(logits, dim=1)
    
    print(f"Model successfully processed {len(actions)} games!")
    print(f"Sample actions from first 5 games: {actions[:5]}")

if __name__ == "__main__":
    simple_train()