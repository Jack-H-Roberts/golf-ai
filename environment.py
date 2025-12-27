import torch
from utils import (
    TABLE_START, SLOT_SIZE, PLAYER_GRID_SIZE, 
    BAG_START, get_card_color, STAGE_START
)

class GolfEnv:
    def __init__(self, num_envs=1024, device="cuda"):
        self.num_envs = num_envs
        self.device = device
        
        # [Batch, 737] - The master state vector (AI Observation)
        self.state = torch.zeros((num_envs, 737), device=self.device)
        
        # [Batch, 104] - The deck sources (Hidden)
        self.full_decks = torch.zeros((num_envs, 104), device=self.device, dtype=torch.long)
        
        # [Batch, 5, 9] - The TRUE face values of cards in the grid (Hidden)
        # We need this to know what card we are picking up if it's currently face-down.
        self.grid_values = torch.zeros((num_envs, 5, 9), device=self.device, dtype=torch.long)
        
        # Deck pointers and template
        single_deck = torch.arange(1, 14).repeat(8) 
        self.deck_template = single_deck.to(self.device)
        self.deck_pointers = torch.full((num_envs,), 45, device=self.device, dtype=torch.long)

    def reset(self):
        """Prepares 1024 games: Shuffles, deals, and sets initial state."""
        self.state.fill_(0)
        
        # 1. Shuffle all 1024 decks
        random_indices = torch.rand((self.num_envs, 104), device=self.device).argsort(dim=1)
        self.full_decks = self.deck_template.expand(self.num_envs, -1).gather(1, random_indices)
        
        # 2. Deal hands (Indices 0-44)
        self.deal_initial_hands()
        
        # 3. Deal the Initial Discard Card (Index 45)
        # We need to set the bits in DISCARD_START
        from utils import DISCARD_START
        
        # Get card 45 for all games
        discard_cards = self.full_decks[:, 45]
        
        # Update Deck Pointers (Next card is 46)
        self.deck_pointers[:] = 46
        
        # Set Discard Bits
        for i in range(self.num_envs):
            val = discard_cards[i].item()
            is_red = (val >= 2) and (val <= 8)
            
            # Set Color
            self.state[i, DISCARD_START] = 1.0 if is_red else 0.0
            # Set Face (Bit 1-13)
            self.state[i, DISCARD_START + val] = 1.0

        # 4. Calculate Red Counts
        for p in range(5):
            p_start = TABLE_START + (p * PLAYER_GRID_SIZE)
            red_bits = self.state[:, p_start : p_start + PLAYER_GRID_SIZE : SLOT_SIZE]
            self.state[:, BAG_START + p] = red_bits.sum(dim=1)
            
        # 5. Set Stage Flag to 'Arrange'
        self.state[:, STAGE_START] = 1.0 
        
        return self.state

    def step(self, actions):
        """
        Handles Arrange, Flip 1, and Flip 2 logic.
        """
        # 1. Identify which games are in 'Arrange' mode
        is_arrange = self.state[:, STAGE_START] == 1.0
        
        if is_arrange.any():
            # Get only the games currently arranging
            arrange_state = self.state[is_arrange]
            num_arranging = arrange_state.size(0)
            
            # Get preferences for grid slots 0-8
            prefs = actions[is_arrange, :9]
            
            # Get the number of reds each player has to place
            red_counts = arrange_state[:, BAG_START].long()
            
            for i in range(num_arranging):
                k = red_counts[i].item()
                _, top_indices = torch.topk(prefs[i], k=k)
                
                # We work on Player 0's grid
                p_idx = TABLE_START 
                
                # Reset this player's grid to all zeros first
                self.state[is_arrange][i, p_idx : p_idx + PLAYER_GRID_SIZE] = 0.0
                
                # Now place Red/Blue bits. 
                # IMPORTANT: Face bits (2-14) remain 0, indicating "Unknown".
                for slot in range(9):
                    slot_base = p_idx + (slot * SLOT_SIZE)
                    if slot in top_indices:
                        self.state[is_arrange][i, slot_base] = 1.0 # Bit 0 (Red)
                    else:
                        self.state[is_arrange][i, slot_base + 1] = 1.0 # Bit 1 (Blue)

            # Update Stage: Move from 'Arrange' to 'Flip 1'
            # We use boolean indexing on the main state to update only relevant rows
            self.state[is_arrange, STAGE_START] = 0.0
            self.state[is_arrange, STAGE_START + 1] = 1.0

        # --- LOGIC FOR FLIP 1 & FLIP 2 ---
        is_flip1 = self.state[:, STAGE_START + 1] == 1.0
        is_flip2 = self.state[:, STAGE_START + 2] == 1.0
        is_flipping = is_flip1 | is_flip2
        
        if is_flipping.any():
            # Get the single best action (0-8) for flipping
            flip_choices = torch.argmax(actions[is_flipping, :9], dim=1)
            
            # Get the real face values from the decks
            # P0 cards map to deck indices 0-8
            current_decks = self.full_decks[is_flipping]
            revealed_faces = current_decks.gather(1, flip_choices.unsqueeze(1)).squeeze(1)
            
            # Map updates back to master state
            flip_game_indices = torch.nonzero(is_flipping).squeeze(1)
            
            for i in range(len(flip_game_indices)):
                game_idx = flip_game_indices[i]
                slot_idx = flip_choices[i].item() # 0-8
                face_val = revealed_faces[i].item() # 1-13
                
                # Calculate start of this slot
                base = TABLE_START + (slot_idx * SLOT_SIZE)
                
                # Set the Face bit.
                # Index 0=Red, 1=Blue. 
                # Face 1 (Ace) -> Index 2. Face 13 (King) -> Index 14.
                # So we add (1 + face_val) to base.
                self.state[game_idx, base + 1 + face_val] = 1.0
            
            # Stage Transition
            # Flip 1 -> Flip 2
            flip1_indices = is_flip1 
            self.state[flip1_indices, STAGE_START + 1] = 0.0
            self.state[flip1_indices, STAGE_START + 2] = 1.0
            
            # Flip 2 -> Play 2.1
            # Note: We use the ORIGINAL is_flip2 mask
            self.state[is_flip2, STAGE_START + 2] = 0.0
            self.state[is_flip2, STAGE_START + 3] = 1.0

        # --- LOGIC FOR PLAY PHASE (Stages 3 & 4) ---
        # Stage 3 = Play 2.1 (Take/Draw)
        # Stage 4 = Play 2.2 (Place/Discard)
        is_p2_1 = self.state[:, STAGE_START + 3] == 1.0
        is_p2_2 = self.state[:, STAGE_START + 4] == 1.0
        is_playing = is_p2_1 | is_p2_2
        
        if is_playing.any():
            from utils import DISCARD_START, DISCARD_SIZE, GRAVEYARD_START
            
            # Get actions for playing games
            play_actions = torch.argmax(actions[is_playing], dim=1) # 0-9
            
            # --- SUB-PHASE 2.1: DECIDE (Take Discard vs Draw) ---
            # Action 0-8: Take Discard & Swap with Slot X
            # Action 9: Draw from Deck (Move to 2.2)
            if is_p2_1.any():
                # Identify games in 2.1
                active_2_1 = is_p2_1 
                acts_2_1 = torch.argmax(actions[active_2_1], dim=1)
                
                # Case A: Draw from Deck (Action 9)
                draw_mask = (acts_2_1 == 9)
                if draw_mask.any():
                    # We need to target the specific games that chose '9'
                    # We map this back to the main 'state' index
                    draw_indices = torch.nonzero(active_2_1)[draw_mask].squeeze(1)
                    
                    # 1. Get next card from deck
                    ptrs = self.deck_pointers[draw_indices]
                    new_cards = self.full_decks[draw_indices, ptrs]
                    self.deck_pointers[draw_indices] += 1
                    
                    # 2. Update DISCARD_START with this new card
                    # (Conceptually, we are 'holding' it, but we put it in the Market slot for the NN to see)
                    # First, we must push the *current* Discard to Graveyard (if it exists)
                    # NOTE: In Golf, you don't bury the discard when you draw; 
                    # you compare the drawn card to the discard. 
                    # BUT for our NN input, we defined "Active Card" as the Discard Slot.
                    # So we temporarily overwrite the Discard slot with the Drawn card for 2.2.
                    # We DO NOT update Graveyard yet because the old discard is still there physically.
                    
                    # Clear current Discard slot logic...
                    # (For simplicity in this snippet, let's just write the new card to Discard Slot)
                    base_d = DISCARD_START
                    self.state[draw_indices, base_d : base_d + DISCARD_SIZE] = 0.0
                    
                    # Encode new card (Color + Face)
                    # This helper logic would ideally be a function, but inlined for now:
                    for i, idx in enumerate(draw_indices):
                        val = new_cards[i].item()
                        is_red = (val >= 2) and (val <= 8)
                        # Set Color
                        self.state[idx, base_d] = 1.0 if is_red else 0.0
                        # Set Face (Bit 1-13)
                        # Our Discard Vector is 14 bits: [Color, Ace...King]
                        # Face 1 (Ace) -> Index 1. Face 13 -> Index 13.
                        self.state[idx, base_d + val] = 1.0
                    
                    # Transition to 2.2
                    self.state[draw_indices, STAGE_START + 3] = 0.0
                    self.state[draw_indices, STAGE_START + 4] = 1.0

                # Case B: Swap Discard with Grid (Action 0-8)
                swap_mask = ~draw_mask
                if swap_mask.any():
                    swap_indices = torch.nonzero(active_2_1)[swap_mask].squeeze(1)
                    slots = acts_2_1[swap_mask] # 0-8
                    
                    # Perform Swap Logic (Function call below)
                    self._perform_swap(swap_indices, slots)
                    
                    # Turn ends -> Next Player
                    self._next_turn(swap_indices)


            # --- SUB-PHASE 2.2: PLACE (Swap Drawn vs Discard Drawn) ---
            # Action 0-8: Swap Drawn (currently in Discard Slot) with Slot X
            # Action 9: Discard the Drawn Card (It stays on top)
            if is_p2_2.any():
                active_2_2 = is_p2_2
                acts_2_2 = torch.argmax(actions[active_2_2], dim=1)
                
                # Case A: Swap (Action 0-8)
                swap_mask = (acts_2_2 != 9)
                if swap_mask.any():
                    swap_indices = torch.nonzero(active_2_2)[swap_mask].squeeze(1)
                    slots = acts_2_2[swap_mask]
                    self._perform_swap(swap_indices, slots)
                    self._next_turn(swap_indices)
                    
                # Case B: Discard (Action 9)
                discard_mask = ~swap_mask
                if discard_mask.any():
                    # The card is already in the Discard Slot (from 2.1).
                    # We just move to next turn.
                    # BUT: We need to update Graveyard with the *card that was previously there*?
                    # No, if we draw and discard, the previous top discard is now buried.
                    # We need to handle that Graveyard update.
                    
                    # For now, let's just rotate turn.
                    disc_indices = torch.nonzero(active_2_2)[discard_mask].squeeze(1)
                    self._next_turn(disc_indices)
        
        # --- CHECK FOR DONE ---
        # 1. Check if P0 is finished
        dones = self._check_round_over()
        
        # 2. Calculate Rewards
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        if dones.any():
            # Calculate final scores
            final_scores = self._calculate_final_scores()
            
            # Reward = -1 * Your Score (Lower is better)
            # Normalize? If score is 50, reward is -50.
            # Ideally, we want (Opponent Avg Score - My Score).
            
            my_score = final_scores[:, 0]
            others_score = final_scores[:, 1:].mean(dim=1)
            
            # Simple Reward: Did I win? (My score < Avg Others)
            # Or simplified: Reward = (Others - Mine)
            # If I have 10, Others have 20 -> Reward +10.
            # If I have 30, Others have 20 -> Reward -10.
            
            rewards[dones] = (others_score[dones] - my_score[dones]) / 10.0 # Scale down
            
            # Auto-reset finished games? 
            # Usually in vectorized envs, we reset dones automatically.
            # For now, let's just return the done flags and let the training loop handle reset.
            
        return self.state, rewards, dones, {}

    def deal_initial_hands(self):
        """Deals 9 cards to each player. Sets Red/Blue bits. Face bits = 0."""
        # Get the first 45 cards
        hands = self.full_decks[:, :45].view(self.num_envs, 5, 9)
        
        # Store the TRUTH
        self.grid_values = hands.clone()
        
        # Update the OBSERVATION (State Bits)
        for p in range(5):
            for c in range(9):
                idx = TABLE_START + (p * PLAYER_GRID_SIZE) + (c * SLOT_SIZE)
                val = hands[:, p, c]
                
                # Determine Color
                is_red = (val >= 2) & (val <= 8)
                self.state[is_red, idx] = 1.0     # Bit 0
                self.state[~is_red, idx + 1] = 1.0 # Bit 1
                
                # Face bits remain 0 (Unknown)

    def get_action_mask(self):
        """
        Returns binary mask [1024, 10]. 
        1=Legal, 0=Illegal.
        """
        mask = torch.zeros((self.num_envs, 10), device=self.device)
        
        stages = self.state[:, STAGE_START : STAGE_START + 5]
        
        # --- ARRANGE ---
        arrange_mask = stages[:, 0] == 1.0
        mask[arrange_mask, 0:9] = 1.0
        
        # --- FLIPS ---
        flip_mask = (stages[:, 1] == 1.0) | (stages[:, 2] == 1.0)
        
        if flip_mask.any():
            # Check P0's grid for cards that have NO face bits set
            # For each slot, we sum bits 2-14. If sum == 0, it's hidden.
            p0_start = TABLE_START
            
            for slot in range(9):
                slot_start = p0_start + (slot * SLOT_SIZE)
                # Slice bits 2-15 (non-inclusive of 15, so 2..14)
                face_bits = self.state[:, slot_start + 2 : slot_start + 15]
                
                # If sum of face bits is 0, the card is hidden/valid to flip
                is_hidden = face_bits.sum(dim=1) == 0.0
                
                valid_slot = flip_mask & is_hidden
                mask[valid_slot, slot] = 1.0

        # --- PLAY ---
        play_mask = (stages[:, 3] == 1.0) | (stages[:, 4] == 1.0)
        mask[play_mask, 0:10] = 1.0
        
        return mask
    
    def _perform_swap(self, game_indices, slot_indices):
        """
        Swaps the card at DISCARD with the card at [CurrentPlayer, Slot].
        """
        from utils import DISCARD_START, TABLE_START, SLOT_SIZE, PLAYER_GRID_SIZE
        
        # We assume Current Player is always P0 relative to the decision maker
        # But we need to check SEAT_ID to know which absolute player grid to touch?
        # For simplicity in this training setup, let's assume the state is already rotated
        # so that P0 is always the active player. (If not, we need SEAT_ID logic).
        # Let's stick to P0 = Active for now.
        
        p_idx = 0 
        
        # 1. Read the value of the card currently in the Grid (The one leaving)
        # We use advanced indexing: [Batch, Player=0, Slot=Indices]
        old_grid_vals = self.grid_values[game_indices, p_idx, slot_indices]
        
        # 2. Read the value of the card currently in Discard (The one entering)
        # We need to extract this from the state bits or track it separately?
        # State bits are reliable for Discard since it's always Face Up.
        # Let's decode the state bits to get the integer value.
        discard_bits = self.state[game_indices, DISCARD_START : DISCARD_START+14]
        # Discard format: [Color, Ace...King]
        # We want the index of the Face bit (1-13). 
        # Argmax of bits[1:] gives 0..12. Add 1 -> 1..13.
        new_grid_vals = torch.argmax(discard_bits[:, 1:], dim=1) + 1
        
        # 3. SWAP THE TRUTH (grid_values)
        self.grid_values[game_indices, p_idx, slot_indices] = new_grid_vals
        
        # 4. UPDATE THE OBSERVATION (State Bits)
        
        # A. Update Grid Slot (It now holds the card from Discard)
        for i, game_idx in enumerate(game_indices):
            slot = slot_indices[i].item()
            new_val = new_grid_vals[i].item()
            
            base = TABLE_START + (slot * SLOT_SIZE)
            
            # Reset slot bits
            self.state[game_idx, base : base + SLOT_SIZE] = 0.0
            
            # Set Color
            is_red = (2 <= new_val <= 8)
            self.state[game_idx, base] = 1.0 if is_red else 0.0
            self.state[game_idx, base + 1] = 0.0 if is_red else 1.0
            
            # Set Face (It is now KNOWN)
            self.state[game_idx, base + 1 + new_val] = 1.0 # +1 skips Color bits, +val maps to Face bit

        # B. Update Discard Slot (It now holds the old grid card)
        # This card is now REVEALED to everyone
        for i, game_idx in enumerate(game_indices):
            old_val = old_grid_vals[i].item()
            
            # Reset Discard bits
            self.state[game_idx, DISCARD_START : DISCARD_START + 14] = 0.0
            
            # Set Color
            is_red = (2 <= old_val <= 8)
            self.state[game_idx, DISCARD_START] = 1.0 if is_red else 0.0
            
            # Set Face (Bit 1-13)
            self.state[game_idx, DISCARD_START + old_val] = 1.0

    def _next_turn(self, game_indices):
        """
        Advances the turn for the specified games.
        1. Rotates SEAT_ID (P0->P1, P4->P0).
        2. Sets Stage to Play 2.1 (Index 3).
        """
        from utils import SEAT_ID_START, STAGE_START
        
        # 1. Rotate Seat ID
        # Current one-hot: [0, 1, 0, 0, 0] -> Index 1
        # New one-hot: [0, 0, 1, 0, 0] -> Index 2
        current_seats = self.state[game_indices, SEAT_ID_START : SEAT_ID_START + 5]
        current_indices = torch.argmax(current_seats, dim=1)
        next_indices = (current_indices + 1) % 5
        
        # Clear old seat bits
        self.state[game_indices, SEAT_ID_START : SEAT_ID_START + 5] = 0.0
        # Set new seat bits
        # We need a scatter or loop. Loop is fine for clarity here.
        for i, game_idx in enumerate(game_indices):
            next_seat = next_indices[i].item()
            self.state[game_idx, SEAT_ID_START + next_seat] = 1.0
            
        # 2. Reset Stage to Play 2.1
        self.state[game_indices, STAGE_START : STAGE_START + 5] = 0.0
        self.state[game_indices, STAGE_START + 3] = 1.0 # Index 3 is Play 2.1

    def _calculate_final_scores(self):
        """
        Calculates the score for all players in all games.
        Returns a tensor [1024, 5] of scores.
        """
        # grid_values is [Batch, 5, 9]
        # We need to reshape to [Batch, 5, 3, 3] (3 rows, 3 cols) to check columns
        # However, our slots 0-8 are row-major (0,1,2 is row 0). 
        # So columns are: (0,3,6), (1,4,7), (2,5,8).
        
        scores = torch.zeros((self.num_envs, 5), device=self.device)
        
        for p in range(5):
            # Get this player's grid [Batch, 9]
            g = self.grid_values[:, p, :]
            
            # Extract columns
            col0 = torch.stack([g[:, 0], g[:, 3], g[:, 6]], dim=1)
            col1 = torch.stack([g[:, 1], g[:, 4], g[:, 7]], dim=1)
            col2 = torch.stack([g[:, 2], g[:, 5], g[:, 8]], dim=1)
            
            total_score = torch.zeros(self.num_envs, device=self.device)
            
            # Helper to score a column
            # If all 3 values are equal, score is 0. Else sum.
            for col in [col0, col1, col2]:
                # Check equality: (row0 == row1) & (row1 == row2)
                # Note: Golf usually requires them to be matching ranks. 
                # Our values 1-13 are ranks.
                match = (col[:, 0] == col[:, 1]) & (col[:, 1] == col[:, 2])
                
                # Sum the column
                col_sum = col.sum(dim=1).float()
                
                # If match, score is 0. Else col_sum.
                # We also assume Kings (13) might be 0 points in some rules, 
                # but let's stick to standard "Face Value" except K=0? 
                # Let's keep it simple: Face Value 1-13. 
                # (You can tweak this later: e.g., King=0, Ace=1).
                
                # Apply cancellation
                col_score = torch.where(match, torch.zeros_like(col_sum), col_sum)
                total_score += col_score
            
            scores[:, p] = total_score
            
        return scores

    def _check_round_over(self):
        """
        Returns a boolean tensor [1024] indicating which games have ended.
        A round ends if ANY player has revealed all 9 cards.
        """
        from utils import TABLE_START, SLOT_SIZE, PLAYER_GRID_SIZE
        
        # We check the OBSERVATION state. 
        # If a card is hidden, its Face Bits (2-14) are all 0.
        # We want to find games where for at least one player, ALL slots have >0 sum in Face Bits.
        
        # This is expensive to loop. Let's do a vectorized check on P0 (Active Player) first?
        # Actually, in Golf, ANY player ending the game triggers the final turn.
        # For this version 1.0, we will end the game IMMEDIATELY when P0 reveals their last card.
        # (Implementing the 'one last turn' logic is complex; let's start with 'First to finish ends it'.)
        
        # Check P0 grid
        p0_start = TABLE_START
        # We need to sum face bits for all 9 slots
        # Reshape to [Batch, 9, 15]
        p0_grid = self.state[:, p0_start : p0_start + PLAYER_GRID_SIZE].view(self.num_envs, 9, SLOT_SIZE)
        
        # Sum bits 2-14 for each slot
        face_sums = p0_grid[:, :, 2:].sum(dim=2) # [Batch, 9]
        
        # A slot is revealed if sum > 0.
        # A player is done if ALL 9 slots are revealed.
        cards_revealed = (face_sums > 0).sum(dim=1) # [Batch]
        
        is_done = (cards_revealed == 9)
        return is_done

    def reset_indices(self, indices):
        """
        Resets specific games (given by boolean mask) to the starting state.
        Required for continuous training.
        """
        if not indices.any():
            return

        # 1. Select the games that need resetting
        # We perform a partial reset by generating new data for just these indices
        n_reset = indices.sum().item()
        
        # 2. Reshuffle decks for these games
        rand_idx = torch.rand((n_reset, 104), device=self.device).argsort(dim=1)
        new_decks = self.deck_template.expand(n_reset, -1).gather(1, rand_idx)
        self.full_decks[indices] = new_decks
        
        # 3. Deal Initial Hands (Indices 0-44)
        hands = new_decks[:, :45].view(n_reset, 5, 9)
        self.grid_values[indices] = hands.clone()
        
        # We need to map these hands to the 737-vector state.
        # Since vectorizing partial updates is complex, we'll do a focused update.
        # Create a temporary sub-state for the reset games
        sub_state = torch.zeros((n_reset, 737), device=self.device)
        
        from utils import TABLE_START, SLOT_SIZE, PLAYER_GRID_SIZE, BAG_START, STAGE_START, DISCARD_START
        
        # A. Populate Grid Bits
        for p in range(5):
            for c in range(9):
                slot_idx = TABLE_START + (p * PLAYER_GRID_SIZE) + (c * SLOT_SIZE)
                val = hands[:, p, c]
                
                is_red = (val >= 2) & (val <= 8)
                # Set Red (Bit 0) or Blue (Bit 1)
                sub_state[:, slot_idx] = torch.where(is_red, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
                sub_state[:, slot_idx+1] = torch.where(is_red, torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device))
                # Face bits remain 0 (Unknown)

        # B. Deal Discard (Card 45)
        discard_cards = new_decks[:, 45]
        is_red_d = (discard_cards >= 2) & (discard_cards <= 8)
        
        # Set Color
        sub_state[:, DISCARD_START] = torch.where(is_red_d, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        # Set Face
        sub_state.scatter_(1, (DISCARD_START + discard_cards).unsqueeze(1), 1.0)
        
        # C. Red Counts
        for p in range(5):
            p_start = TABLE_START + (p * PLAYER_GRID_SIZE)
            red_bits = sub_state[:, p_start : p_start + PLAYER_GRID_SIZE : SLOT_SIZE]
            sub_state[:, BAG_START + p] = red_bits.sum(dim=1)
            
        # D. Reset Pointers & Stage
        self.deck_pointers[indices] = 46
        sub_state[:, STAGE_START] = 1.0 # Arrange Phase
        
        # Apply the update to the master state
        self.state[indices] = sub_state

    def debug_print_hand(self, player_idx=0):
        """Prints a 3x3 grid using the new 15-bit logic."""
        start = TABLE_START + (player_idx * PLAYER_GRID_SIZE)
        hand_bits = self.state[0, start : start + PLAYER_GRID_SIZE]
        bag_count = self.state[0, BAG_START + player_idx].item()
        
        print(f"\n--- Player {player_idx} Hand (Game 0) | Starting Reds: {int(bag_count)} ---")
        for row in range(3):
            row_str = ""
            for col in range(3):
                s_start = (row * 3 + col) * SLOT_SIZE
                slot = hand_bits[s_start : s_start + SLOT_SIZE]
                
                # Check Face Bits (2-14)
                face_sum = slot[2:].sum().item()
                is_red = slot[0].item() == 1.0
                color_str = "Red" if is_red else "Blue"
                
                if face_sum == 0:
                    row_str += f"[ ? ({color_str}) ] "
                else:
                    # Find which bit is set. 
                    # slot[2] is Ace (1), slot[14] is King (13)
                    # argmax on the slice gives 0..12. Add 1 to get face value.
                    face_val = torch.argmax(slot[2:]).item() + 1
                    row_str += f"[ {face_val} ({color_str}) ] "
            print(row_str)