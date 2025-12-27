import torch
import numpy as np
import os
import sys

from environment import GolfEnv
from model import GolfModel
from utils import (
    TABLE_START, SLOT_SIZE, PLAYER_GRID_SIZE,
    DISCARD_START, DRAW_START, STAGE_START, POINT_MAP
)

def render_game(env, human_idx=0):
    os.system('cls' if os.name == 'nt' else 'clear')
    state = env.state
    dealer = env.dealer_pos[0].item()
    curr = env.current_player[0].item()
    
    print(f"=== GOLF AI ARENA ===")
    print(f"You are P{human_idx}. Dealer is P{dealer}. Current Turn: P{curr}")
    
    scores = env.game_scores[0].cpu().numpy()
    score_str = " | ".join([f"P{i}: {int(s)}" for i, s in enumerate(scores)])
    print(f"SCORES: [ {score_str} ]")
    print("-" * 60)

    # DRAW PILE
    draw_bit = state[0, DRAW_START]
    draw_str = "BLUE" if draw_bit == 1.0 else "RED"
    
    # DISCARD PILE
    disc_bits = state[0, DISCARD_START : DISCARD_START+14]
    if disc_bits.sum() == 0:
        disc_str = "[ Empty ]"
    else:
        is_blue = (disc_bits[0] == 1.0)
        rank = torch.argmax(disc_bits[1:]).item() + 1
        color = "BLUE" if is_blue else "RED"
        disc_str = f"[ {rank} ({color}) ]"
        
    print(f"DRAW PILE: [ ? ({draw_str}) ]    DISCARD PILE: {disc_str}")
    print("-" * 60)

    for p in range(5):
        prefix = f"YOU (P{p})" if p == human_idx else f"CPU {p}"
        if p == dealer: prefix += " [D]"
        if p == curr: prefix += " <--"
        
        p_start = TABLE_START + (p * PLAYER_GRID_SIZE)
        print(f"{prefix}:")
        
        for row in range(3):
            row_str = "   "
            for col in range(3):
                slot_idx = row * 3 + col
                base = p_start + (slot_idx * SLOT_SIZE)
                bits = state[0, base : base + SLOT_SIZE]
                
                is_blue_back = (bits[1] == 1.0) 
                face_bits = bits[2:]
                
                back_char = "B" if is_blue_back else "R"
                
                if face_bits.sum() == 0:
                    val_str = str(slot_idx) 
                    row_str += f"[{val_str}{back_char}] "
                else:
                    rank = torch.argmax(face_bits).item() + 1
                    row_str += f"[{rank:2}{back_char}] "
            print(row_str)
        print("")

def get_human_arrange_action(num_reds):
    print(f"\nARRANGE PHASE: You have {num_reds} RED cards.")
    print("Enter the indices (0-8) where you want to place RED cards.")
    print("Example: '0 4 8' will put red cards at top-left, center, bottom-right.")
    
    while True:
        try:
            inp = input("Indices: ")
            parts = [int(x) for x in inp.split()]
            if len(parts) != num_reds:
                print(f"Please enter exactly {num_reds} numbers.")
                continue
            if any(x < 0 or x > 8 for x in parts):
                print("Indices must be 0-8.")
                continue
            if len(set(parts)) != num_reds:
                print("Indices must be unique.")
                continue
            
            action = torch.zeros(10)
            action[:] = -100.0
            for idx in parts:
                action[idx] = 100.0
            return action
        except ValueError:
            print("Invalid input.")

def get_human_play_action(mask):
    valid_indices = torch.nonzero(mask[0]).flatten().cpu().numpy()
    print(f"Valid Options: {valid_indices}")
    while True:
        try:
            val = int(input("Enter Action ID: "))
            if val in valid_indices:
                return val
            print("Invalid choice.")
        except ValueError:
            print("Number please.")

def play_game(model_path="latest_model.pt"):
    device = torch.device("cpu")
    env = GolfEnv(num_envs=1, device=device)
    agent = GolfModel().to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded.")
    except:
        print("Model not found. Using random agent.")

    obs = env.reset()
    done = False
    HUMAN_SEAT = 0 # Fixed seat for you, but dealer rotates
    
    while not done:
        render_game(env, human_idx=HUMAN_SEAT)
        
        curr = env.current_player[0].item()
        mask = env.get_action_mask()
        stage_bits = env.state[0, STAGE_START:STAGE_START+5]
        
        action_tensor = torch.zeros((1, 10), device=device)

        if curr == HUMAN_SEAT:
            print("\n>>> YOUR TURN <<<")
            
            if stage_bits[0] == 1: # ARRANGE
                backs = env.grid_backs[0, HUMAN_SEAT, :]
                num_reds = (backs == 0).sum().item()
                logits = get_human_arrange_action(num_reds)
                action_tensor[0] = logits
                
            else:
                if stage_bits[1] == 1: print("Phase: FLIP 1 (Choose 0-8)")
                elif stage_bits[2] == 1: print("Phase: FLIP 2 (Choose 0-8)")
                elif stage_bits[3] == 1: print("Phase: PLAY (0-8=Swap Discard, 9=Draw)")
                elif stage_bits[4] == 1: print("Phase: DRAWN (0-8=Swap Drawn, 9=Discard Drawn)")
                
                idx = get_human_play_action(mask)
                action_tensor[0, idx] = 100.0
        else:
            with torch.no_grad():
                logits, _ = agent(obs, mask=mask)
                if stage_bits[0] == 1:
                    action_tensor[0] = logits[0]
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    idx = dist.sample()
                    action_tensor.scatter_(1, idx.unsqueeze(1), 100.0)

        obs, reward, dones, info = env.step(action_tensor)
        
        if dones[0]:
            render_game(env, human_idx=HUMAN_SEAT)
            print("GAME OVER")
            break

if __name__ == "__main__":
    play_game()