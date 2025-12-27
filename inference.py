import torch
import numpy as np
import os
import time

from environment import GolfEnv
from model import GolfModel
from utils import (
    TABLE_START, SLOT_SIZE, PLAYER_GRID_SIZE,
    DISCARD_START, DRAW_START, STAGE_START, POINT_MAP
)

# --- VISUALIZATION HELPERS ---
RANK_MAP = {
    1: 'A', 10: '10', 11: 'J', 12: 'Q', 13: 'K'
}

def get_card_str(rank_idx, is_blue_back, is_face_up):
    # Show Back Color for face-down cards (e.g., ?R or ?B)
    color_char = "B" if is_blue_back else "R"
    
    if not is_face_up:
        return f"[ ?{color_char} ]" 
    
    # Show Rank + Original Back Color for face-up cards
    rank_str = RANK_MAP.get(rank_idx, str(rank_idx))
    return f"[{rank_str:>2}{color_char}]"

def render_game(env, human_idx=0):
    # Clear screen (optional, can comment out for debugging)
    # os.system('cls' if os.name == 'nt' else 'clear')
    
    state = env.state
    dealer = env.dealer_pos[0].item()
    curr = env.current_player[0].item()
    
    print("\n" + "="*60)
    print(f"=== GOLF AI ARENA ===")
    print(f"You are P{human_idx}. Dealer is P{dealer}. Current Turn: P{curr}")
    
    scores = env.game_scores[0].cpu().numpy()
    score_str = " | ".join([f"P{i}: {int(s):3}" for i, s in enumerate(scores)])
    print(f"SCORES: [ {score_str} ]")
    print("-" * 60)

    # DRAW PILE
    draw_bit = state[0, DRAW_START]
    draw_str = "BLUE" if draw_bit == 1.0 else "RED"
    
    # DISCARD PILE
    disc_bits = state[0, DISCARD_START : DISCARD_START+14]
    
    # Debug: Check if discard bits are actually set
    if disc_bits.sum() == 0:
        disc_str = "[ Empty (Bug?) ]"
    else:
        is_blue = (disc_bits[0] == 1.0)
        # argmax of bits 1-13 gives 0-12, so +1 to get Rank
        # We assume bits 1..13 contain the face.
        face_vec = disc_bits[1:]
        if face_vec.sum() == 0:
             disc_str = "[ Empty? ]"
        else:
            rank = torch.argmax(face_vec).item() + 1
            color = "B" if is_blue else "R"
            r_str = RANK_MAP.get(rank, str(rank))
            disc_str = f"[ {r_str}{color} ]"
        
    print(f"DRAW PILE: [ ?{draw_str[0]} ]       DISCARD PILE: {disc_str}")
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
                
                # Decode bits
                # Bit 0 = Red Back, Bit 1 = Blue Back
                # Bit 2..14 = Rank (Face Value)
                
                # Note: bits[1] == 1.0 means Blue. 
                is_blue_back = (bits[1] == 1.0)
                
                face_bits = bits[2:]
                is_face_up = (face_bits.sum() > 0)
                
                rank = 0
                if is_face_up:
                    rank = torch.argmax(face_bits).item() + 1
                
                c_str = get_card_str(rank, is_blue_back, is_face_up)
                row_str += f"{slot_idx}:{c_str}  "
            print(row_str)
        print("")

def get_input(prompt, valid_range, count=1):
    while True:
        try:
            inp = input(f"{prompt}: ").strip()
            parts = [int(x) for x in inp.split()]
            
            if len(parts) != count:
                print(f"Please enter exactly {count} number(s).")
                continue
                
            if any(x not in valid_range for x in parts):
                print(f"Numbers must be in {valid_range}.")
                continue
                
            if count > 1 and len(set(parts)) != count:
                print("Numbers must be unique.")
                continue
                
            return parts[0] if count == 1 else parts
        except ValueError:
            print("Invalid input. Please enter numbers.")

def play_game(model_path="latest_model.pt"):
    device = torch.device("cpu")
    # Initialize 1 environment
    env = GolfEnv(num_envs=1, device=device)
    agent = GolfModel().to(device)
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Handle both full checkpoint dict and direct state_dict
            if 'model_state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['model_state_dict'])
            else:
                agent.load_state_dict(checkpoint)
            print(f"Loaded AI from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using Random Agent.")
    else:
        print("No model found. Playing against Random Bot.")

    obs = env.reset()
    done = False
    HUMAN_SEAT = 0 
    
    # Debug: Force a render immediately after reset to verify Discard Pile
    print("DEBUG: Initial State (Check Discard Pile)")
    render_game(env, human_idx=HUMAN_SEAT)
    input("Press Enter to start...")

    while not done:
        curr = env.current_player[0].item()
        mask = env.get_action_mask()
        stage_bits = env.state[0, STAGE_START:STAGE_START+5]
        
        # RENDER only when it changes player or phase
        render_game(env, human_idx=HUMAN_SEAT)

        action_tensor = torch.zeros((1, 10), device=device)

        if curr == HUMAN_SEAT:
            print("\n>>> YOUR TURN <<<")
            
            # 1. ARRANGE
            if stage_bits[0] == 1: 
                backs = env.grid_backs[0, HUMAN_SEAT, :]
                num_reds = (backs == 0).sum().item()
                print(f"ARRANGE PHASE: You have {num_reds} Red-backed cards.")
                print("The rest will be Blue.")
                if num_reds > 0:
                    indices = get_input("Enter indices for RED cards", range(9), count=num_reds)
                else:
                    print("No red cards! Auto-arranging.")
                    indices = []
                
                # Logic: We set the chosen indices to high value
                action_tensor[:] = -100.0
                if isinstance(indices, int): indices = [indices]
                for idx in indices:
                    action_tensor[0, idx] = 100.0
            
            # 2. FLIP 1 & 3. FLIP 2
            elif stage_bits[1] == 1 or stage_bits[2] == 1:
                phase_name = "FLIP 1" if stage_bits[1] == 1 else "FLIP 2"
                print(f"{phase_name}: Choose a face-down card to reveal.")
                valid = torch.nonzero(mask[0]).flatten().tolist()
                idx = get_input(f"Choose card index {valid}", valid, count=1)
                action_tensor[0, idx] = 100.0
            
            # 4. PLAY (Top Discard vs Draw)
            elif stage_bits[3] == 1:
                print("PLAY PHASE: Top Discard is available.")
                print("Enter 0-8 to SWAP that card with your card.")
                print("Enter 9 to PASS (and draw from deck).")
                idx = get_input("Choice", range(10), count=1)
                action_tensor[0, idx] = 100.0
                
            # 5. PLAY (Drawn Card)
            elif stage_bits[4] == 1:
                print("DECISION PHASE: You drew a card.")
                print("Enter 0-8 to SWAP drawn card with your card.")
                print("Enter 9 to DISCARD drawn card.")
                idx = get_input("Choice", range(10), count=1)
                action_tensor[0, idx] = 100.0

        else:
            # AI TURN
            print(f"CPU {curr} is thinking...")
            time.sleep(0.5) 
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
            # Determine winner
            scores = env.game_scores[0].cpu().numpy()
            min_score = scores.min()
            
            winner_idx = np.argmin(scores)
            if winner_idx == HUMAN_SEAT:
                 print("VICTORY! You won.")
            else:
                 print(f"DEFEAT. P{winner_idx} won.")
            break

if __name__ == "__main__":
    play_game()