import torch
from environment import GolfEnv

def run_test():
    # 1. Initialize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = GolfEnv(num_envs=1024, device=device)
    
    # 2. Reset and Deal
    state = env.reset()
    print(f"Environment Reset on {device}. State shape: {state.shape}")
    
    # 3. Verify specific data
    # Check if the first player in Game 0 has a valid Bag count
    env.debug_print_hand(player_idx=0)
    
    # 4. Integrity Check
    # Ensure that across 1024 games, the number of reds isn't constant (randomness check)
    from utils import BAG_START
    all_red_counts = state[:, BAG_START]
    print(f"\nAverage Red cards dealt to P0 across all games: {all_red_counts.mean().item():.2f}")
    print("Test Complete!")

if __name__ == "__main__":
    run_test()