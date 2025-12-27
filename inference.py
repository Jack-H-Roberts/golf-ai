import torch
import numpy as np
import os
import sys

from environment import GolfEnv
from model import GolfModel
from utils import (
    TABLE_START, SLOT_SIZE, PLAYER_GRID_SIZE,
    DISCARD_START, STAGE_START, POINT_MAP
)

def render_game(env, human_idx=0):
    """
    Prints a beautiful text representation of the table.
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    state = env.state
    print(f"=== GOLF AI ARENA (You are Player {human_idx}) ===")
    
    # 1. Print Scores (Normalized in state, so we multiply by 100 for display)
    # Note: These are simplified running totals.
    scores = env.game_scores[0].cpu().numpy()
    score_str = " | ".join([f"P{i}: {int(s)}" for i, s in enumerate(scores)])
    print(f"SCORES: [ {score_str} ]")
    print("-" * 60)

    # 2. Print Top Discard
    disc_bits = state[0, DISCARD_START : DISCARD_START+14]
    if disc_bits.sum() == 0:
        print("DISCARD PILE: [ Empty ]")
    else:
        is_blue = (disc_bits[0] == 1.0)
        rank = torch.argmax(disc_bits[1:]).item() + 1
        color = "BLUE" if is_blue else "RED"
        print(f"DISCARD PILE: [ {rank} ({color}) ]")
        
    print("-" * 60)

    # 3. Print Each Player's Hand
    for p in range(5):
        is_me = (p == human_idx)
        prefix = "YOU" if is_me else f"CPU {p}"
        
        # Get bits for this player
        p_start = TABLE_START + (p * PLAYER_GRID_SIZE)
        print(f"{prefix}:")
        
        for row in range(3):
            row_str = "   "
            for col in range(3):
                slot_idx = row * 3 + col
                base = p_start + (slot_idx * SLOT_SIZE)
                bits = state[0, base : base + SLOT_SIZE]
                
                # Decode
                is_red_back = (bits[0] == 1.0)
                is_blue_back = (bits[1] == 1.0) # or just check bits[0]
                
                # Check Face (sum of bits 2-14)
                face_bits = bits[2:]
                if face_bits.sum() == 0:
                    # Hidden
                    back_char = "B" if is_blue_back else "R"
                    val_str = "?"
                    row_str += f"[{val_str}{back_char}] "
                else:
                    # Revealed
                    rank = torch.argmax(face_bits).item() + 1
                    back_char = "B" if is_blue_back else "R"
                    row_str += f"[{rank:2}{back_char}] "
            print(row_str)
        print("")

def get_human_action(mask):
    """Asks user for a valid action."""
    valid_indices = torch.nonzero(mask[0]).flatten().cpu().numpy()
    print(f"Valid Options: {valid_indices}")
    
    while True:
        try:
            choice = input("Enter Action ID: ")
            val = int(choice)
            if val in valid_indices:
                return val
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a number.")

def play_game(model_path="latest_model.pt"):
    device = torch.device("cpu") # Inference is fine on CPU
    
    # 1. Setup Environment (1 game only)
    env = GolfEnv(num_envs=1, device=device)
    agent = GolfModel().to(device)
    
    # 2. Load Brain
    try:
        checkpoint = torch.load(model_path, map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from update {checkpoint.get('update', '?')}")
    except FileNotFoundError:
        print("Model file not found! Please download 'latest_model.pt' first.")
        return

    # 3. Start
    obs = env.reset()
    done = False
    
    # We fix the Human as Player 0 for simplicity.
    # The environment rotates 'current_player' internally.
    # We check env.current_player to see whose turn it is.
    
    while not done:
        render_game(env)
        
        # Who is acting?
        # current_player is a tensor [1]
        active_p = env.current_player[0].item()
        
        mask = env.get_action_mask()
        
        if active_p == 0:
            # --- HUMAN TURN ---
            print("\n>>> YOUR TURN <<<")
            stage_bits = env.state[0, STAGE_START:STAGE_START+5]
            if stage_bits[0] == 1: print("Phase: ARRANGE (Enter 0-8 to toggle preference)")
            elif stage_bits[1] == 1: print("Phase: FLIP 1 (Choose 0-8 to flip)")
            elif stage_bits[2] == 1: print("Phase: FLIP 2 (Choose 0-8 to flip)")
            elif stage_bits[3] == 1: print("Phase: PLAY (0-8=Swap Discard, 9=Draw)")
            elif stage_bits[4] == 1: print("Phase: DRAWN (0-8=Swap Drawn, 9=Discard Drawn)")
            
            action_idx = get_human_action(mask)
            action_tensor = torch.zeros((1, 10), device=device)
            action_tensor[0, action_idx] = 100.0 # High logit
            
        else:
            # --- AI TURN ---
            # print(f"\n--- CPU {active_p} is thinking... ---")
            with torch.no_grad():
                logits, _ = agent(obs, mask=mask)
                dist = torch.distributions.Categorical(logits=logits)
                action_idx = dist.sample()
                
                action_tensor = torch.zeros((1, 10), device=device)
                action_tensor.scatter_(1, action_idx.unsqueeze(1), 100.0)
                
        # Step
        obs, reward, dones, info = env.step(action_tensor)
        
        if dones[0]:
            render_game(env)
            print("\n!!! GAME OVER !!!")
            print(f"Final Reward signal: {reward[0].item()}")
            break
            
if __name__ == "__main__":
    play_game()