import torch
from environment import GolfEnv

def test_scoring():
    env = GolfEnv(num_envs=1) # Single game for clarity
    env.reset()
    
    # Cheat: Set P0's grid manually to:
    # 5  1  2
    # 5  8  3
    # 5  9  4
    # Column 0 is matching (5,5,5). Score should be 0 for that col.
    # Col 1 sum: 1+8+9 = 18.
    # Col 2 sum: 2+3+4 = 9.
    # Total Score should be 0 + 18 + 9 = 27.
    
    grid = torch.tensor([
        [5, 1, 2],
        [5, 8, 3],
        [5, 9, 4]
    ])
    # Flatten to [9]
    flat_grid = grid.flatten()
    
    # Update grid_values
    env.grid_values[0, 0, :] = flat_grid
    
    # Calculate
    scores = env._calculate_final_scores()
    p0_score = scores[0, 0].item()
    
    print("--- Scoring Test ---")
    print("Grid:\n", grid)
    print(f"Calculated Score: {p0_score}")
    
    if p0_score == 27.0:
        print(">> SUCCESS: Column cancellation works.")
    else:
        print(f">> FAILURE: Expected 27.0, got {p0_score}")

if __name__ == "__main__":
    test_scoring()