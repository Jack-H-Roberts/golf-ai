import torch
from environment import GolfEnv
from utils import (
    STAGE_START, TABLE_START, SLOT_SIZE, 
    DISCARD_START, PLAYER_GRID_SIZE
)

def test_play_logic():
    # 1. Setup
    env = GolfEnv(num_envs=1024, device="cuda")
    state = env.reset()
    
    print("--- Test Start: Play Phase Logic ---")
    
    # 2. Cheat: Force Game 0 into 'Play 2.1' (Take/Draw)
    # We skip Arrange/Flip for this test
    state[:, STAGE_START : STAGE_START + 5] = 0.0
    state[:, STAGE_START + 3] = 1.0 
    
    # 3. Cheat: Force specific cards for Game 0 so we can track them
    # Force Grid Slot 0 to be a Hidden King (13)
    env.grid_values[0, 0, 0] = 13 # Player 0, Slot 0
    # Update State bits to show it is Hidden (Face bits 0)
    slot0_idx = TABLE_START
    state[0, slot0_idx : slot0_idx + SLOT_SIZE] = 0.0
    state[0, slot0_idx + 1] = 1.0 # Blue (King is Blue)
    state[0, slot0_idx + 14] = 0.0 # Unknown? No, wait. 
    # Your logic: If face bits are 0, it is unknown.
    # So we leave bits 2-14 as 0. Correct.
    
    # Force Discard to be a Red 5
    state[0, DISCARD_START : DISCARD_START + 14] = 0.0
    state[0, DISCARD_START] = 1.0 # Red
    state[0, DISCARD_START + 5] = 1.0 # Face 5
    
    print("\n[Before Swap]")
    print("Grid Slot 0: [ ? (Blue) ] (Hidden King)")
    print("Top Discard: [ 5 (Red) ]")
    
    # 4. Action: Player 0 chooses Index 0 (Swap Discard with Slot 0)
    actions = torch.zeros((1024, 10), device="cuda")
    actions[:, 0] = 100.0 # High logit for index 0
    
    env.step(actions)
    
    print("\n[After Swap]")
    
    # 5. Verify Results
    # A. Grid Slot 0 should now be the Red 5 (Revealed)
    slot0_bits = state[0, slot0_idx : slot0_idx + SLOT_SIZE]
    is_red = slot0_bits[0] == 1.0
    face_val_idx = torch.argmax(slot0_bits[2:]).item()
    face_val = face_val_idx + 1 if slot0_bits[2:].sum() > 0 else 0
    
    print(f"Grid Slot 0: [ {face_val} ({'Red' if is_red else 'Blue'}) ]")
    
    if face_val == 5 and is_red:
        print(">> SUCCESS: Grid updated correctly.")
    else:
        print(f">> FAILURE: Expected 5 (Red), got {face_val}")

    # B. Discard Slot should now be the King (Revealed)
    discard_bits = state[0, DISCARD_START : DISCARD_START + 14]
    disc_red = discard_bits[0] == 1.0
    disc_face = torch.argmax(discard_bits[1:]).item() + 1
    
    print(f"Top Discard: [ {disc_face} ({'Red' if disc_red else 'Blue'}) ]")
    
    if disc_face == 13:
        print(">> SUCCESS: Hidden King was revealed in Discard.")
    else:
        print(f">> FAILURE: Expected 13 (Blue), got {disc_face}")

if __name__ == "__main__":
    test_play_logic()