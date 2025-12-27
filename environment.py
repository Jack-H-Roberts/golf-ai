import torch
from utils import (
    TABLE_START, SLOT_SIZE, PLAYER_GRID_SIZE, 
    BAG_START, get_card_color, STAGE_START
)

class GolfEnv:
    def __init__(self, num_envs=1024, device="cuda"):
        self.num_envs = num_envs
        self.device = device
        
        # [Batch, 737] - The master state vector
        self.state = torch.zeros((num_envs, 737), device=self.device)
        
        # [Batch, 104] - The actual card faces (A=1, K=13) hidden from AI
        self.full_decks = torch.zeros((num_envs, 104), device=self.device, dtype=torch.long)
        
        # 2 decks: (A-K) x 8 suits total
        single_deck = torch.arange(1, 14).repeat(8) 
        self.deck_template = single_deck.to(self.device)

    def reset(self):
        """Prepares 1024 games: Shuffles, deals, and sets initial state."""
        self.state.fill_(0)
        
        # 1. Shuffle all 1024 decks
        random_indices = torch.rand((self.num_envs, 104), device=self.device).argsort(dim=1)
        self.full_decks = self.deck_template.expand(self.num_envs, -1).gather(1, random_indices)
        
        # 2. Deal hands and populate color bits
        self.deal_initial_hands()
        
        # 3. Calculate and set Initial Red Counts (The 'Bags')
        for p in range(5):
            p_start = TABLE_START + (p * PLAYER_GRID_SIZE)
            # Step through every 15th bit (the Red bit) for all 9 cards
            red_bits = self.state[:, p_start : p_start + PLAYER_GRID_SIZE : SLOT_SIZE]
            self.state[:, BAG_START + p] = red_bits.sum(dim=1)
            
        # 4. Set Stage Flag to 'Arrange' [1, 0, 0, 0, 0]
        self.state[:, STAGE_START] = 1.0 
        
        return self.state

    def step(self, actions):
        """
        Stage 1: Arrange
        'actions' is a tensor of shape [Batch, 10] containing move preferences.
        """

        # 1. Identify which games are in 'Arrange' mode (Stage index 0 is 1.0)
        is_arrange = self.state[:, STAGE_START] == 1.0
        
        if is_arrange.any():
            # Get only the games currently arranging
            arrange_state = self.state[is_arrange]
            num_arranging = arrange_state.size(0)
            
            # Get preferences for grid slots 0-8
            prefs = actions[is_arrange, :9]
            
            # Get the number of reds each player has to place
            red_counts = arrange_state[:, BAG_START].long()
            
            # Logic: We create a 'Rank' for each slot [Batch, 9]
            # topk gives us the indices of the highest N preferences
            # However, since N varies, we'll use a mask-based approach or a fast loop
            for i in range(num_arranging):
                k = red_counts[i].item()
                _, top_indices = torch.topk(prefs[i], k=k)
                
                # Clear existing color bits for this player
                p_idx = TABLE_START # Player 0
                self.state[i, p_idx : p_idx + PLAYER_GRID_SIZE] = 0.0
                
                # Set the 'Unknown' bit for all slots
                for slot in range(9):
                    self.state[i, p_idx + (slot * SLOT_SIZE) + 14] = 1.0
                    
                # Place Reds at top_indices (Bit 0) and Blues everywhere else (Bit 1)
                for slot in range(9):
                    if slot in top_indices:
                        self.state[i, p_idx + (slot * SLOT_SIZE)] = 1.0 # Red
                    else:
                        self.state[i, p_idx + (slot * SLOT_SIZE) + 1] = 1.0 # Blue

            # Update Stage: Move from 'Arrange' to 'Flip 1'
            self.state[is_arrange, STAGE_START] = 0.0
            self.state[is_arrange, STAGE_START + 1] = 1.0

    def deal_initial_hands(self):
        """Deals 9 cards to each player and sets color + unknown bits."""
        # Slice the first 45 cards and reshape to [Batch, Player, Card]
        hands = self.full_decks[:, :45].view(self.num_envs, 5, 9)
        
        for p in range(5):
            for c in range(9):
                idx = TABLE_START + (p * PLAYER_GRID_SIZE) + (c * SLOT_SIZE)
                val = hands[:, p, c]
                
                # Set 'Unknown' bit (index 14 in slot)
                self.state[:, idx + 14] = 1.0
                
                # Set color bits (0 for Red, 1 for Blue)
                is_red = (val >= 2) & (val <= 8)
                self.state[is_red, idx] = 1.0
                self.state[~is_red, idx + 1] = 1.0

    def get_action_mask(self):
        """
        Returns a binary mask of shape [1024, 10].
        Based on the current Stage Flag in the 737-vector.
        """
        from utils import STAGE_START
        mask = torch.zeros((self.num_envs, 10), device=self.device)
        
        # Get current stage for all games [Batch, 5]
        stages = self.state[:, STAGE_START : STAGE_START + 5]
        
        # Stage 1: Arrange (Index 0)
        # Legal: Neurons 0-8 (Grid), Illegal: Neuron 9 (Draw/Discard)
        arrange_mask = stages[:, 0] == 1.0
        mask[arrange_mask, 0:9] = 1.0
        
        # Stage 2 & 3: Flip 1 & Flip 2 (Indices 1 & 2)
        # Legal: Neurons 0-8, Illegal: Neuron 9
        flip_mask = (stages[:, 1] == 1.0) | (stages[:, 2] == 1.0)
        mask[flip_mask, 0:9] = 1.0
        
        # --- Crucial: Mask cards already flipped ---
        # We don't want the AI to flip the same card twice.
        # This will be implemented fully once we have the 'Flip' logic.

        # Stage 4 & 5: Play Phase (Indices 3 & 4)
        # Legal: All neurons 0-9
        play_mask = (stages[:, 3] == 1.0) | (stages[:, 4] == 1.0)
        mask[play_mask, 0:10] = 1.0
        
        return mask

    def debug_print_hand(self, player_idx=0):
        """Prints a 3x3 grid of the first game's hand for verification."""
        start = TABLE_START + (player_idx * PLAYER_GRID_SIZE)
        hand_bits = self.state[0, start : start + PLAYER_GRID_SIZE]
        bag_count = self.state[0, BAG_START + player_idx].item()
        
        print(f"\n--- Player {player_idx} Hand (Game 0) | Starting Reds: {int(bag_count)} ---")
        for row in range(3):
            row_str = ""
            for col in range(3):
                s_start = (row * 3 + col) * SLOT_SIZE
                slot = hand_bits[s_start : s_start + SLOT_SIZE]
                color = "Red" if slot[0] == 1.0 else "Blue"
                row_str += f"[ ? ({color}) ] "
            print(row_str)