import torch
from utils import (
    TABLE_START, SLOT_SIZE, PLAYER_GRID_SIZE, 
    BAG_START, STAGE_START, DISCARD_START, 
    GRAVEYARD_START, DRAW_START, SCORES_START,
    SEAT_ID_START, TRIGGER_START, DISCARD_SIZE, 
    POINT_MAP, GRAVEYARD_SIZE
)

class GolfEnv:
    def __init__(self, num_envs=1024, device="cuda"):
        self.num_envs = num_envs
        self.device = device
        
        # [Batch, 737] - The master state vector
        self.state = torch.zeros((num_envs, 737), device=self.device)
        
        # --- HIDDEN STATE ---
        # The Deck: 104 Cards. 
        # IDs 1-52 = Red Back (Hearts/Diams/Clubs/Spades 1-13)
        # IDs 53-104 = Blue Back
        self.full_decks = torch.zeros((num_envs, 104), device=self.device, dtype=torch.long)
        self.deck_pointers = torch.zeros((num_envs,), device=self.device, dtype=torch.long)
        
        # Dealer Tracker (0-4)
        self.dealer_pos = torch.zeros((num_envs,), device=self.device, dtype=torch.long)
        
        # Current Player (0-4) - Absolute Index
        self.current_player = torch.zeros((num_envs,), device=self.device, dtype=torch.long)
        
        # True Grid Values [Batch, 5, 9] (Rank 1-13)
        self.grid_values = torch.zeros((num_envs, 5, 9), device=self.device, dtype=torch.long)
        # True Grid Backs [Batch, 5, 9] (0=Red, 1=Blue)
        self.grid_backs = torch.zeros((num_envs, 5, 9), device=self.device, dtype=torch.long)
        
        # Game Cumulative Scores
        self.game_scores = torch.zeros((num_envs, 5), device=self.device, dtype=torch.float)
        
        # Deck Templates
        # Ranks: 1-13 repeated 4 times (Red), then 1-13 repeated 4 times (Blue)
        self.rank_template = torch.cat([
            torch.arange(1, 14).repeat(4), 
            torch.arange(1, 14).repeat(4)
        ]).to(self.device)
        
        # Backs: 52 Red (0), 52 Blue (1)
        self.back_template = torch.cat([
            torch.zeros(52), 
            torch.ones(52)
        ]).long().to(self.device)

        # Temporary Buffers for Shuffling
        self.shuffled_ranks = torch.zeros((num_envs, 104), device=self.device, dtype=torch.long)
        self.shuffled_backs = torch.zeros((num_envs, 104), device=self.device, dtype=torch.long)

    def reset(self):
        """Starts a NEW GAME (Scores = 0)."""
        self.state.fill_(0)
        self.game_scores.fill_(0)
        
        # Randomize Dealer for first round
        self.dealer_pos = torch.randint(0, 5, (self.num_envs,), device=self.device)
        
        # Start the first round
        self._start_new_round(initial=True)
        return self.state

    def _start_new_round(self, initial=False, indices=None):
        """Resets cards, deals hands, keeps scores. Handles dealer rotation."""
        if indices is None:
            # All envs (usually just for initial reset)
            indices = torch.arange(self.num_envs, device=self.device)
            
        n = len(indices)
        if n == 0: return

        if not initial:
            # Rotate Dealer Left
            self.dealer_pos[indices] = (self.dealer_pos[indices] + 1) % 5
            
            # Reset Round-Specific State bits (keep Scores)
            scores_backup = self.state[indices, SCORES_START : SCORES_START + 5].clone()
            self.state[indices] = 0.0
            self.state[indices, SCORES_START : SCORES_START + 5] = scores_backup

        # 1. Shuffle
        rand_idx = torch.rand((n, 104), device=self.device).argsort(dim=1)
        
        self.shuffled_ranks[indices] = self.rank_template.expand(n, -1).gather(1, rand_idx)
        self.shuffled_backs[indices] = self.back_template.expand(n, -1).gather(1, rand_idx)
        
        # 2. Deal 9 cards to 5 players (Indices 0-44)
        self.grid_values[indices] = self.shuffled_ranks[indices, :45].view(n, 5, 9)
        self.grid_backs[indices] = self.shuffled_backs[indices, :45].view(n, 5, 9)
        
        # 3. Setup Initial Observation (Arrange Phase)
        # Populate BAG_START (Count of Red Backs)
        for p in range(5):
            p_backs = self.grid_backs[indices, p, :] 
            num_blues = p_backs.sum(dim=1)
            num_reds = 9 - num_blues
            self.state[indices, BAG_START + p] = num_reds.float()
        
        # 4. Set Initial Discard (Card 45)
        self.deck_pointers[indices] = 46
        
        discard_val = self.shuffled_ranks[indices, 45]
        discard_back = self.shuffled_backs[indices, 45]
        self._update_discard_obs(discard_val, discard_back, indices=indices)
        
        # 5. Set Draw Pile Color (Card 46)
        self._update_draw_obs(indices=indices)
        
        # 6. Set Stage to Arrange (1.1)
        self.state[indices, STAGE_START] = 1.0 
        
        # 7. Set Starting Player (Left of Dealer)
        self.current_player[indices] = (self.dealer_pos[indices] + 1) % 5
        self._update_seat_obs(indices)

    def step(self, actions):
        """Main transition function."""
        # --- STAGE HANDLERS ---
        
        # 1. ARRANGE
        is_arrange = self.state[:, STAGE_START] == 1.0
        if is_arrange.any():
            self._handle_arrange(actions, is_arrange)

        # 2. FLIP 1
        is_flip1 = self.state[:, STAGE_START + 1] == 1.0
        if is_flip1.any():
            self._handle_flip(actions, is_flip1, stage_idx=1)
            
        # 3. FLIP 2
        is_flip2 = self.state[:, STAGE_START + 2] == 1.0
        if is_flip2.any():
            self._handle_flip(actions, is_flip2, stage_idx=2)

        # 4. PLAY PHASE (2.1 & 2.2)
        is_p2_1 = self.state[:, STAGE_START + 3] == 1.0
        is_p2_2 = self.state[:, STAGE_START + 4] == 1.0
        is_playing = is_p2_1 | is_p2_2
        
        if is_playing.any():
            self._handle_play(actions, is_playing)

        # --- CHECK ROUND END ---
        round_over_mask = self._check_round_over()
        
        rewards = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        if round_over_mask.any():
            # 1. Calculate Round Scores
            round_scores = self._calculate_round_scores()
            
            # 2. Add to Game Scores (masked)
            self.game_scores[round_over_mask] += round_scores[round_over_mask]
            
            # 3. Update Observation of Scores (Normalized)
            self.state[round_over_mask, SCORES_START : SCORES_START+5] = self.game_scores[round_over_mask] / 100.0
            
            # 4. Check Game Over (>100)
            max_scores, _ = self.game_scores.max(dim=1)
            game_over = (max_scores >= 100) & round_over_mask
            
            if game_over.any():
                dones[game_over] = True
                # Reward: (Avg Opponent Score - My Score) / 10
                final_s = self.game_scores[game_over]
                my_score = final_s[:, 0] 
                opp_score = final_s[:, 1:].mean(dim=1)
                rewards[game_over] = (opp_score - my_score) / 10.0
            
            # 5. Start New Round for continuous games (where game is NOT over)
            reset_round_mask = round_over_mask & (~game_over)
            if reset_round_mask.any():
                indices = torch.nonzero(reset_round_mask).squeeze(1)
                self._start_new_round(initial=False, indices=indices)

        return self.state, rewards, dones, {}

    # --- HELPERS ---

    def _handle_arrange(self, actions, mask):
        """Places Red Backs where requested, shuffling hidden values to match."""
        indices = torch.nonzero(mask).squeeze(1)
        
        # P0 is the active agent.
        # We need to swap the TRUE grid values so that the visual placement matches the hidden reality.
        
        for i in indices:
            # 1. Get current P0 hand
            current_vals = self.grid_values[i, 0, :]
            current_backs = self.grid_backs[i, 0, :]
            
            # 2. Identify Red vs Blue indices in the current hand
            red_indices = (current_backs == 0).nonzero().flatten()
            blue_indices = (current_backs == 1).nonzero().flatten()
            num_reds = len(red_indices)
            
            # 3. Get User Preferences for Red Slots
            prefs = actions[i, :9]
            _, top_k = torch.topk(prefs, k=num_reds)
            
            # 4. Create new arrangement buffers
            new_vals = torch.zeros(9, dtype=torch.long, device=self.device)
            new_backs = torch.zeros(9, dtype=torch.long, device=self.device)
            
            # 5. Fill Red Slots (using the values that had Red Backs)
            new_vals[top_k] = current_vals[red_indices]
            new_backs[top_k] = 0 # Red
            
            # 6. Fill Blue Slots (using values that had Blue Backs)
            # Identify which slots were NOT chosen
            is_red_slot = torch.zeros(9, dtype=torch.bool, device=self.device)
            is_red_slot[top_k] = True
            blue_slots = (~is_red_slot).nonzero().flatten()
            
            new_vals[blue_slots] = current_vals[blue_indices]
            new_backs[blue_slots] = 1 # Blue
            
            # 7. Update Truth
            self.grid_values[i, 0, :] = new_vals
            self.grid_backs[i, 0, :] = new_backs
            
            # 8. Update Observation (Visuals only, Values hidden)
            p0_start = TABLE_START
            self.state[i, p0_start : p0_start + PLAYER_GRID_SIZE] = 0.0 # Clear
            
            for slot in range(9):
                idx = p0_start + (slot * SLOT_SIZE)
                if slot in top_k:
                    self.state[i, idx] = 1.0 # Red Back
                else:
                    self.state[i, idx+1] = 1.0 # Blue Back
                    
        # Transition -> Flip 1
        self.state[mask, STAGE_START] = 0.0
        self.state[mask, STAGE_START+1] = 1.0

    def _handle_flip(self, actions, mask, stage_idx):
        indices = torch.nonzero(mask).squeeze(1)
        choices = torch.argmax(actions[mask, :9], dim=1)
        
        for idx, i in enumerate(indices):
            slot = choices[idx].item()
            rank = self.grid_values[i, 0, slot].item()
            base = TABLE_START + (slot * SLOT_SIZE)
            # Reveal Face Bit (Rank 1 -> Index 2 ... Rank 13 -> Index 14)
            self.state[i, base + 1 + rank] = 1.0
            
        current_stage = STAGE_START + stage_idx
        self.state[mask, current_stage] = 0.0
        self.state[mask, current_stage + 1] = 1.0

    def _handle_play(self, actions, mask):
        act_inds = torch.argmax(actions[mask], dim=1)
        indices = torch.nonzero(mask).squeeze(1)
        
        for idx, i in enumerate(indices):
            action = act_inds[idx].item()
            
            # --- PHASE 2.1 (Decide on Top Discard) ---
            if self.state[i, STAGE_START + 3] == 1.0:
                if action == 9: 
                    # DRAW
                    ptr = self.deck_pointers[i]
                    if ptr >= 104: ptr = 0 
                    
                    drawn_rank = self.shuffled_ranks[i, ptr].item()
                    drawn_back = self.shuffled_backs[i, ptr].item()
                    self.deck_pointers[i] += 1
                    
                    # Store the OLD discard (the one we are covering up) to Graveyard
                    self._add_current_discard_to_graveyard(i)
                    
                    # Show Drawn Card in Discard Slot
                    self._update_discard_obs(
                        torch.tensor([drawn_rank], device=self.device), 
                        torch.tensor([drawn_back], device=self.device),
                        indices=[i]
                    )
                    
                    # Update Draw Pile Obs (peek next)
                    self._update_draw_obs(indices=[i])

                    # Transition -> 2.2
                    self.state[i, STAGE_START + 3] = 0.0
                    self.state[i, STAGE_START + 4] = 1.0
                    
                else: 
                    # SWAP Discard (0-8)
                    self._swap_card(i, slot=action)
                    self._next_turn(i)

            # --- PHASE 2.2 (Decide on Drawn Card) ---
            elif self.state[i, STAGE_START + 4] == 1.0:
                if action == 9:
                    # DISCARD Drawn Card (It stays on top)
                    # We pass turn. The 'drawn' card is currently at Discard Obs.
                    # Previous discard was already buried in 2.1.
                    self._next_turn(i)
                else:
                    # SWAP Drawn Card (0-8)
                    # Drawn card goes to grid. Old grid card becomes top discard.
                    self._swap_card(i, slot=action)
                    self._next_turn(i)

    def _swap_card(self, game_idx, slot):
        # 1. Get Old Grid Card
        old_rank = self.grid_values[game_idx, 0, slot].item()
        old_back = self.grid_backs[game_idx, 0, slot].item()
        
        # 2. Get New Card (from Discard Obs)
        disc_bits = self.state[game_idx, DISCARD_START:DISCARD_START+14]
        new_back = 0 if disc_bits[0] == 0.0 else 1
        new_rank = torch.argmax(disc_bits[1:]).item() + 1
        
        # 3. Update Truth
        self.grid_values[game_idx, 0, slot] = new_rank
        self.grid_backs[game_idx, 0, slot] = new_back
        
        # 4. Update Grid Obs
        base = TABLE_START + (slot * SLOT_SIZE)
        self.state[game_idx, base : base+SLOT_SIZE] = 0.0
        if new_back == 0:
            self.state[game_idx, base] = 1.0 
            self.state[game_idx, base+1] = 0.0
        else:
            self.state[game_idx, base] = 0.0
            self.state[game_idx, base+1] = 1.0
        # Set Face
        self.state[game_idx, base + 1 + new_rank] = 1.0
        
        # 5. Old Grid Card becomes Top Discard
        # If we came from 2.1 Swap, we need to bury the previous discard first.
        # If we came from 2.2 Swap, the previous discard was buried in 2.1.
        if self.state[game_idx, STAGE_START+3] == 1.0: # If we are in 2.1
             self._add_current_discard_to_graveyard(game_idx)
             
        self._update_discard_obs(
            torch.tensor([old_rank], device=self.device),
            torch.tensor([old_back], device=self.device),
            indices=[game_idx]
        )

    def _add_current_discard_to_graveyard(self, game_idx):
        """Reads the current Discard Obs and increments Graveyard."""
        disc_bits = self.state[game_idx, DISCARD_START:DISCARD_START+14]
        
        # Check if there is actually a card there (Sum > 0)
        if disc_bits.sum() == 0: return

        # Decode
        is_blue = (disc_bits[0] == 1.0)
        face_idx = torch.argmax(disc_bits[1:]).item() # 0-12 (Ace-King)
        
        # Graveyard Index: Red(0-12), Blue(13-25)
        gy_idx = face_idx + (13 if is_blue else 0)
        
        self.state[game_idx, GRAVEYARD_START + gy_idx] += 1.0

    def _next_turn(self, game_idx):
        self.current_player[game_idx] = (self.current_player[game_idx] + 1) % 5
        self._update_seat_obs(indices=[game_idx])
        self.state[game_idx, STAGE_START : STAGE_START+5] = 0.0
        self.state[game_idx, STAGE_START + 3] = 1.0

    def _update_discard_obs(self, val, back, indices=None):
        if indices is None: indices = slice(None)
        
        # 1. Clear Bits
        self.state[indices, DISCARD_START : DISCARD_START+14] = 0.0
        
        # 2. Set Color Bit
        is_blue = (back == 1)
        # Use simple assignment with broadcast
        self.state[indices, DISCARD_START] = is_blue.float()
        
        # 3. Set Face Bits
        # FIXED: Use direct advanced indexing instead of scatter_
        offset = DISCARD_START + val
        self.state[indices, offset] = 1.0

    def _update_draw_obs(self, indices=None):
        """Peek at next card's back color."""
        if indices is None: indices = torch.arange(self.num_envs, device=self.device)
        if len(indices) == 0: return
        
        ptrs = self.deck_pointers[indices]
        # Safety clamp
        ptrs = torch.clamp(ptrs, max=103)
        
        next_backs = self.shuffled_backs[indices, ptrs]
        is_blue = (next_backs == 1)
        
        self.state[indices, DRAW_START] = is_blue.float()

    def _update_seat_obs(self, indices):
        if indices is None: indices = slice(None)
        
        # FIXED: Use direct assignment
        self.state[indices, SEAT_ID_START : SEAT_ID_START+5] = 0.0
        cols = SEAT_ID_START + self.current_player[indices]
        self.state[indices, cols] = 1.0

    def _check_round_over(self):
        # Check P0 revealed count
        p0_start = TABLE_START
        p0_grid = self.state[:, p0_start : p0_start + PLAYER_GRID_SIZE].view(self.num_envs, 9, SLOT_SIZE)
        # Sum bits 2-14 (Face bits)
        face_sums = p0_grid[:, :, 2:].sum(dim=2)
        revealed_count = (face_sums > 0).sum(dim=1)
        return revealed_count == 9

    def _calculate_round_scores(self):
        scores = torch.zeros((self.num_envs, 5), device=self.device)
        for p in range(5):
            g = self.grid_values[:, p, :] 
            c0 = torch.stack([g[:,0], g[:,3], g[:,6]], dim=1)
            c1 = torch.stack([g[:,1], g[:,4], g[:,7]], dim=1)
            c2 = torch.stack([g[:,2], g[:,5], g[:,8]], dim=1)
            
            p_score = torch.zeros(self.num_envs, device=self.device)
            for col in [c0, c1, c2]:
                match = (col[:,0] == col[:,1]) & (col[:,1] == col[:,2])
                
                col_points = col.clone().float()
                # Apply Map
                for r, pts in POINT_MAP.items():
                    col_points[col == r] = pts
                    
                col_sum = col_points.sum(dim=1)
                col_sum = torch.where(match, torch.zeros_like(col_sum), col_sum)
                p_score += col_sum
            scores[:, p] = p_score
        return scores

    def get_action_mask(self):
        """Standard mask logic."""
        mask = torch.zeros((self.num_envs, 10), device=self.device)
        stages = self.state[:, STAGE_START : STAGE_START + 5]
        
        # Arrange
        mask[stages[:, 0] == 1.0, 0:9] = 1.0
        
        # Flips
        flip_mask = (stages[:, 1] == 1.0) | (stages[:, 2] == 1.0)
        if flip_mask.any():
            p0_start = TABLE_START
            # Allow flipping only hidden cards
            for slot in range(9):
                slot_start = p0_start + (slot * SLOT_SIZE)
                face_bits = self.state[:, slot_start + 2 : slot_start + 15]
                is_hidden = face_bits.sum(dim=1) == 0.0
                valid = flip_mask & is_hidden
                mask[valid, slot] = 1.0
                
        # Play
        play_mask = (stages[:, 3] == 1.0) | (stages[:, 4] == 1.0)
        mask[play_mask, 0:10] = 1.0
        
        return mask
    
    def reset_indices(self, indices):
        # Full game reset for specific indices
        self._start_new_round(initial=True, indices=torch.nonzero(indices).squeeze(1))