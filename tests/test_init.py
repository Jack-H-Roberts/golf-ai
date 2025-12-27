import torch
from environment import GolfEnv
from utils import STAGE_START, TABLE_START, SLOT_SIZE

def test_initialization():
    env = GolfEnv(num_envs=1024, device="cuda")
    state = env.reset()
    
    print("--- Step 0: Reset (Arrange Phase) ---")
    env.debug_print_hand(0)
    
    # Step 1: Arrange
    # We create dummy logits favoring slots 0-3
    actions = torch.randn(1024, 10, device="cuda") 
    actions[:, 0:4] += 10.0 # Force high preference for top-left
    env.step(actions)
    
    # Step 2: Flip 1
    print("\n--- Step 1: After Arrange (Flip 1 Phase) ---")
    # We want to flip slot 0
    actions = torch.zeros(1024, 10, device="cuda")
    actions[:, 0] = 100.0
    env.step(actions)
    
    # Step 3: Flip 2
    print("\n--- Step 2: After First Flip (Flip 2 Phase) ---")
    # We want to flip slot 1
    actions = torch.zeros(1024, 10, device="cuda")
    actions[:, 1] = 100.0
    env.step(actions)
    
    print("\n--- Step 3: Initialization Complete (Play Phase) ---")
    env.debug_print_hand(0)
    
    # Verify we are in Stage 3 (Play 2.1)
    is_play = state[0, STAGE_START + 3] == 1.0
    print(f"\nSuccessfully reached Play Phase? {is_play.item()}")

if __name__ == "__main__":
    test_initialization()