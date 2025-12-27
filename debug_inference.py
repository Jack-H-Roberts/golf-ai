import torch
import numpy as np
import os
import time
import sys

# Import your existing environment
from environment import GolfEnv
from model import GolfModel
from utils import (
    TABLE_START, SLOT_SIZE, PLAYER_GRID_SIZE,
    DISCARD_START, DRAW_START, STAGE_START, 
    POINT_MAP, SEAT_ID_START
)

# --- VISUALIZATION HELPERS ---
RANK_MAP = { 1: 'A', 10: '10', 11: 'J', 12: 'Q', 13: 'K' }

def get_card_str(rank_idx, is_blue_back, is_face_up):
    color_char = "B" if is_blue_back else "R"
    if not is_face_up: return f"[ ?{color_char} ]" 
    rank_str = RANK_MAP.get(rank_idx, str(rank_idx))
    return f"[{rank_str:>2}{color_char}]"

def debug_print_state(env):
    """Prints raw debug info about the state vector."""
    state = env.state[0]
    
    # 1. Check Stage
    stage_bits = state[STAGE_START:STAGE_START+5].cpu().numpy()
    stage_names = ["ARRANGE", "FLIP 1", "FLIP 2", "PLAY (Disc)", "PLAY (Draw)"]
    active_stages = [name for i, name in enumerate(stage_names) if stage_bits[i] == 1.0]
    print(f"DEBUG: Stage Bits: {stage_bits} -> {active_stages}")

    # 2. Check Player
    curr = env.current_player[0].item()
    print(f"DEBUG: Current Player Logic: {curr}")

    # 3. Check Discard
    disc_bits = state[DISCARD_START:DISCARD_START+14]
    print(f"DEBUG: Discard Bits Sum: {disc_bits.sum().item()}")

def force_fix_discard(env):
    """Injects a random card into discard if it is empty."""
    if env.state[0, DISCARD_START : DISCARD_START+14].sum() == 0:
        print("!!! WARNING: Discard pile was empty. Force-fixing it... !!!")
        # Set Blue King (Color=1, Face=13)
        # Bit 0 = 1.0 (Blue)
        # Bit 13 (Index 13) = 1.0 (King)
        env.state[0, DISCARD_START] = 1.0 # Blue
        env.state[0, DISCARD_START + 13] = 1.0 # King
        print("!!! Force-fix complete. Discard is now [ K (Blue) ] !!!")

def render_game(env, human_idx=0):
    # REMOVED CLS to prevent hiding debug info
    state = env.state
    dealer = env.dealer_pos[0].item()
    curr = env.current_player[0].item()
    
    print("\n" + "="*60)
    print(f"=== GOLF AI ARENA (DEBUG MODE) ===")
    print(f"You are P{human_idx}. Dealer is P{dealer}. Current Turn: P{curr}")
    
    # Check Discard Display
    disc_bits = state[0, DISCARD_START : DISCARD_START+14]
    if disc_bits.sum() == 0:
        disc_str = "[ EMPTY ERROR ]"
    else:
        is_blue = (disc_bits[0] == 1.0)
        face_vec = disc_bits[1:]
        if face_vec.sum() == 0:
             disc_str = "[ COLOR ONLY? ]"
        else:
            rank = torch.argmax(face_vec).item() + 1
            color = "B" if is_blue else "R"
            r_str = RANK_MAP.get(rank, str(rank))
            disc_str = f"[ {r_str}{color} ]"
        
    print(f"DISCARD PILE: {disc_str}")
    print("-" * 60)

    for p in range(5):
        prefix = f"P{p}"
        if p == human_idx: prefix = f"YOU ({prefix})"
        if p == curr: prefix += " <--"
        
        p_start = TABLE_START + (p * PLAYER_GRID_SIZE)
        
        row_str = f"{prefix}: "
        for i in range(9):
            if i % 3 == 0 and i > 0: row_str += " | "
            
            base = p_start + (i * SLOT_SIZE)
            bits = state[0, base : base + SLOT_SIZE]
            is_blue_back = (bits[1] == 1.0)
            face_bits = bits[2:]
            is_face_up = (face_bits.sum() > 0)
            
            rank = 0
            if is_face_up: rank = torch.argmax(face_bits).item() + 1
            
            row_str += f"{i}:{get_card_str(rank, is_blue_back, is_face_up)} "
        print(row_str)

def get_input(prompt, valid_range, count=1):
    while True:
        try:
            inp = input(f"{prompt}: ").strip()
            parts = [int(x) for x in inp.split()]
            if len(parts) != count:
                print(f"Need exactly {count} numbers.")
                continue
            if any(x not in valid_range for x in parts):
                print(f"Range {valid_range}.")
                continue
            return parts[0] if count == 1 else parts
        except ValueError:
            print("Numbers only.")

def play_game(model_path="latest_model.pt"):
    device = torch.device("cpu")
    env = GolfEnv(num_envs=1, device=device)
    agent = GolfModel().to(device)
    
    if os.path.exists(model_path):
        try:
            ckpt = torch.load(model_path, map_location=device)
            agent.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
            print(f"Loaded {model_path}")
        except:
            print("Model load failed. Using Random.")
    
    obs = env.reset()
    
    # --- DIAGNOSTIC START ---
    force_fix_discard(env)
    debug_print_state(env)
    # ------------------------

    done = False
    HUMAN_SEAT = 0 
    
    while not done:
        curr = env.current_player[0].item()
        stage_bits = env.state[0, STAGE_START:STAGE_START+5]
        
        render_game(env, human_idx=HUMAN_SEAT)
        debug_print_state(env) # See what stage we are really in

        action_tensor = torch.zeros((1, 10), device=device)

        if curr == HUMAN_SEAT:
            print("\n>>> YOUR TURN <<<")
            
            # ARRANGE
            if stage_bits[0] == 1: 
                backs = env.grid_backs[0, HUMAN_SEAT, :]
                num_reds = (backs == 0).sum().item()
                print(f"ARRANGE: Place {num_reds} Red Cards (Indices 0-8)")
                if num_reds > 0:
                    indices = get_input("Indices", range(9), count=num_reds)
                else:
                    print("No reds. Auto-arranging.")
                    indices = []
                
                action_tensor[:] = -100.0
                if isinstance(indices, int): indices = [indices]
                for idx in indices: action_tensor[0, idx] = 100.0
            
            # FLIP 1 & 2
            elif stage_bits[1] == 1 or stage_bits[2] == 1:
                print("FLIP: Choose a card to reveal.")
                mask = env.get_action_mask()
                valid = torch.nonzero(mask[0]).flatten().tolist()
                print(f"Valid options (Face Down): {valid}")
                idx = get_input("Index", valid, count=1)
                action_tensor[0, idx] = 100.0
            
            # PLAY
            elif stage_bits[3] == 1:
                print("PLAY: 0-8 Swap, 9 Pass/Draw")
                idx = get_input("Choice", range(10), count=1)
                action_tensor[0, idx] = 100.0
                
            # DRAWN
            elif stage_bits[4] == 1:
                print("DRAWN: 0-8 Swap, 9 Discard")
                idx = get_input("Choice", range(10), count=1)
                action_tensor[0, idx] = 100.0
            
            else:
                print("!!! ERROR: Human turn but NO STAGE BIT SET !!!")
                print("This is the bug. Manually forcing Flip 1...")
                env.state[0, STAGE_START+1] = 1.0
                continue

        else:
            # AI
            print(f"CPU {curr} acting...")
            time.sleep(0.2)
            with torch.no_grad():
                logits, _ = agent(obs) # Ignore mask for AI simple check
                if stage_bits[0] == 1:
                    action_tensor[0] = logits[0]
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    idx = dist.sample()
                    action_tensor.scatter_(1, idx.unsqueeze(1), 100.0)

        obs, reward, dones, info = env.step(action_tensor)
        
        if dones[0]:
            render_game(env)
            print("GAME OVER")
            break

if __name__ == "__main__":
    play_game()