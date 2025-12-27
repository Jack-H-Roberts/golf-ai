import torch
from utils import (
    TABLE_START, SLOT_SIZE, PLAYER_GRID_SIZE, 
    BAG_START, STAGE_START, DISCARD_START, 
    GRAVEYARD_START, DRAW_START, SCORES_START,
    SEAT_ID_START, DISCARD_SIZE, 
    POINT_MAP, GRAVEYARD_SIZE
)

class GolfEnv:
    def __init__(self, num_envs=1024, device="cuda"):
        self.num_envs = num_envs
        self.device = device
        
        # [Batch, 737] - The master state vector
        self.state = torch.zeros((num_envs, 737), device=self.device)
        
        # --- HIDDEN STATE ---
        self.full_decks = torch.zeros((num_envs, 104), device=self.device, dtype=torch.long)
        self.deck_pointers = torch.zeros((num_envs,), device=self.device, dtype=torch.long)
        self.dealer_pos = torch.zeros((num_envs,), device=self.device, dtype=torch.long)
        self.current_player = torch.zeros((num_envs,), device=self.device, dtype=torch.long)
        
        # Initialization Tracker (0 to 5)
        self.init_count = torch.zeros((num_envs,), device=self.device, dtype=torch.long)
        
        self.grid_values = torch.zeros((num_envs, 5, 9), device=self.device, dtype=torch.long)
        self.grid_backs = torch.zeros((num_envs, 5, 9), device=self.device, dtype=torch.long)
        self.game_scores = torch.zeros((num_envs, 5), device=self.device, dtype=torch.float)
        
        # Templates
        self.rank_template = torch.cat([torch.arange(1, 14).repeat(4), torch.arange(1, 14).repeat(4)]).to(self.device)
        self.back_template = torch.cat([torch.zeros(52), torch.ones(52)]).long().to(self.device)
        self.shuffled_ranks = torch.zeros((num_envs, 104), device=self.device, dtype=torch.long)
        self.shuffled_backs = torch.zeros((num_envs, 104), device=self.device, dtype=torch.long)

    def reset(self):
        self.state.fill_(0)
        self.game_scores.fill_(0)
        self.dealer_pos = torch.randint(0, 5, (self.num_envs,), device=self.device)
        self._start_new_round(initial=True)
        return self.state

    def _start_new_round(self, initial=False, indices=None):
        if indices is None: indices = torch.arange(self.num_envs, device=self.device)
        n = len(indices)
        if n == 0: return

        if not initial:
            self.dealer_pos[indices] = (self.dealer_pos[indices] + 1) % 5
            scores_backup = self.state[indices, SCORES_START : SCORES_START + 5].clone()
            self.state[indices] = 0.0
            self.state[indices, SCORES_START : SCORES_START + 5] = scores_backup

        # 1. Shuffle
        rand_idx = torch.rand((n, 104), device=self.device).argsort(dim=1)
        self.shuffled_ranks[indices] = self.rank_template.expand(n, -1).gather(1, rand_idx)
        self.shuffled_backs[indices] = self.back_template.expand(n, -1).gather(1, rand_idx)
        
        # 2. Deal 9 cards (Indices 0-44)
        self.grid_values[indices] = self.shuffled_ranks[indices, :45].view(n, 5, 9)
        self.grid_backs[indices] = self.shuffled_backs[indices, :45].view(n, 5, 9)
        
        # 3. Populate Initial Red Bags (Visible immediately)
        for p in range(5):
            p_backs = self.grid_backs[indices, p, :] 
            num_blues = p_backs.sum(dim=1)
            num_reds = 9 - num_blues
            self.state[indices, BAG_START + p] = num_reds.float()
        
        # 4. REVEAL Discard (Card 45) -- BEFORE choices
        d_ranks = self.shuffled_ranks[indices, 45]
        d_backs = self.shuffled_backs[indices, 45]
        self._update_discard_obs(d_ranks, d_backs, indices=indices)
        
        # 5. REVEAL Draw Color (Card 46) -- BEFORE choices
        self.deck_pointers[indices] = 46
        self._update_draw_obs(indices=indices)
        
        # 6. Setup First Player (Left of Dealer)
        self.init_count[indices] = 0
        self.current_player[indices] = (self.dealer_pos[indices] + 1) % 5
        self._update_seat_obs(indices)
        
        # 7. Set Stage to Arrange (1.1)
        self.state[indices, STAGE_START] = 1.0 

    def step(self, actions):
        # 1. ARRANGE (P1 -> P2 -> ... P5)
        is_arrange = self.state[:, STAGE_START] == 1.0
        if is_arrange.any():
            self._handle_arrange(actions, is_arrange)

        # 2. FLIP 1
        is_flip1 = self.state[:, STAGE_START + 1] == 1.0
        if is_flip1.any():
            self._handle_flip(actions, is_flip1, stage_idx=1)
            
        # 3. FLIP 2 (End of Init Turn for Player X)
        is_flip2 = self.state[:, STAGE_START + 2] == 1.0
        if is_flip2.any():
            self._handle_flip(actions, is_flip2, stage_idx=2)
            self._advance_init_phase(is_flip2)

        # 4. PLAY PHASE
        is_p2_1 = self.state[:, STAGE_START + 3] == 1.0
        is_p2_2 = self.state[:, STAGE_START + 4] == 1.0
        is_playing = is_p2_1 | is_p2_2
        
        if is_playing.any():
            self._handle_play(actions, is_playing)

        # --- CHECK ROUND END ---
        # Only check if initialization is complete
        game_started = (self.init_count == 5)
        round_over_mask = self._check_round_over() & game_started
        
        rewards = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        if round_over_mask.any():
            round_scores = self._calculate_round_scores()
            self.game_scores[round_over_mask] += round_scores[round_over_mask]
            self.state[round_over_mask, SCORES_START : SCORES_START+5] = self.game_scores[round_over_mask] / 100.0
            
            max_scores, _ = self.game_scores.max(dim=1)
            game_over = (max_scores >= 100) & round_over_mask
            
            if game_over.any():
                dones[game_over] = True
                final_s = self.game_scores[game_over]
                rewards[game_over] = (final_s[:, 1:].mean(dim=1) - final_s[:, 0]) / 10.0
            
            reset_round_mask = round_over_mask & (~game_over)
            if reset_round_mask.any():
                indices = torch.nonzero(reset_round_mask).squeeze(1)
                self._start_new_round(initial=False, indices=indices)

        return self.state, rewards, dones, {}

    # --- HELPERS ---

    def _handle_arrange(self, actions, mask):
        indices = torch.nonzero(mask).squeeze(1)
        for i in indices:
            p = self.current_player[i].item() # Active Player
            
            current_vals = self.grid_values[i, p, :]
            current_backs = self.grid_backs[i, p, :]
            
            red_indices = (current_backs == 0).nonzero().flatten()
            blue_indices = (current_backs == 1).nonzero().flatten()
            num_reds = len(red_indices)
            
            prefs = actions[i, :9]
            _, top_k = torch.topk(prefs, k=num_reds)
            
            new_vals = torch.zeros(9, dtype=torch.long, device=self.device)
            new_backs = torch.zeros(9, dtype=torch.long, device=self.device)
            
            new_vals[top_k] = current_vals[red_indices]
            new_backs[top_k] = 0 # Red
            
            is_red_slot = torch.zeros(9, dtype=torch.bool, device=self.device)
            is_red_slot[top_k] = True
            blue_slots = (~is_red_slot).nonzero().flatten()
            
            new_vals[blue_slots] = current_vals[blue_indices]
            new_backs[blue_slots] = 1 # Blue
            
            self.grid_values[i, p, :] = new_vals
            self.grid_backs[i, p, :] = new_backs
            
            # Update Observation (Visuals)
            p_start = TABLE_START + (p * PLAYER_GRID_SIZE)
            self.state[i, p_start : p_start + PLAYER_GRID_SIZE] = 0.0 
            
            for slot in range(9):
                idx = p_start + (slot * SLOT_SIZE)
                if slot in top_k:
                    self.state[i, idx] = 1.0 
                else:
                    self.state[i, idx+1] = 1.0 

        self.state[mask, STAGE_START] = 0.0
        self.state[mask, STAGE_START+1] = 1.0

    def _handle_flip(self, actions, mask, stage_idx):
        indices = torch.nonzero(mask).squeeze(1)
        choices = torch.argmax(actions[mask, :9], dim=1)
        
        for idx, i in enumerate(indices):
            p = self.current_player[i].item()
            slot = choices[idx].item()
            rank = self.grid_values[i, p, slot].item()
            
            base = TABLE_START + (p * PLAYER_GRID_SIZE) + (slot * SLOT_SIZE)
            self.state[i, base + 1 + rank] = 1.0
            
        current_stage = STAGE_START + stage_idx
        self.state[mask, current_stage] = 0.0
        self.state[mask, current_stage + 1] = 1.0

    def _advance_init_phase(self, mask):
        """Called after Flip 2. Rotates player. If loop done, starts game."""
        indices = torch.nonzero(mask).squeeze(1)
        self.init_count[indices] += 1
        
        # 1. CONTINUE INIT (Count < 5)
        cont_mask = (self.init_count[indices] < 5)
        if cont_mask.any():
            cont_indices = indices[cont_mask]
            # Next Player
            self.current_player[cont_indices] = (self.current_player[cont_indices] + 1) % 5
            self._update_seat_obs(cont_indices)
            # Reset to Arrange
            self.state[cont_indices, STAGE_START : STAGE_START+5] = 0.0
            self.state[cont_indices, STAGE_START] = 1.0
            
        # 2. FINISH INIT (Count == 5)
        start_mask = (self.init_count[indices] == 5)
        if start_mask.any():
            start_indices = indices[start_mask]
            
            # Reset Player to P1 (Left of Dealer)
            self.current_player[start_indices] = (self.dealer_pos[start_indices] + 1) % 5
            self._update_seat_obs(start_indices)
            
            # Set Stage to Play 2.1
            self.state[start_indices, STAGE_START : STAGE_START+5] = 0.0
            self.state[start_indices, STAGE_START+3] = 1.0

    def _handle_play(self, actions, mask):
        act_inds = torch.argmax(actions[mask], dim=1)
        indices = torch.nonzero(mask).squeeze(1)
        
        for idx, i in enumerate(indices):
            action = act_inds[idx].item()
            p = self.current_player[i].item()

            # --- PHASE 2.1 (Decide on Top Discard) ---
            if self.state[i, STAGE_START + 3] == 1.0:
                if action == 9: # PASS (Draw)
                    ptr = self.deck_pointers[i]
                    if ptr >= 104: ptr = 0 
                    
                    drawn_rank = self.shuffled_ranks[i, ptr].item()
                    drawn_back = self.shuffled_backs[i, ptr].item()
                    self.deck_pointers[i] += 1
                    
                    # Pass means current discard goes to Graveyard
                    self._add_current_discard_to_graveyard(i)
                    
                    # Reveal Drawn Card in Discard Slot
                    self._update_discard_obs(torch.tensor([drawn_rank], device=self.device), torch.tensor([drawn_back], device=self.device), indices=[i])
                    self._update_draw_obs(indices=[i])

                    self.state[i, STAGE_START + 3] = 0.0
                    self.state[i, STAGE_START + 4] = 1.0
                else: # SWAP Discard
                    self._swap_card(i, p, slot=action)
                    self._next_turn(i)

            # --- PHASE 2.2 (Decide on Drawn Card) ---
            elif self.state[i, STAGE_START + 4] == 1.0:
                if action == 9: # PASS (Discard Drawn)
                    self._next_turn(i)
                else: # SWAP Drawn
                    self._swap_card(i, p, slot=action)
                    self._next_turn(i)

    def _swap_card(self, game_idx, player_idx, slot):
        old_rank = self.grid_values[game_idx, player_idx, slot].item()
        old_back = self.grid_backs[game_idx, player_idx, slot].item()
        
        disc_bits = self.state[game_idx, DISCARD_START:DISCARD_START+14]
        new_back = 0 if disc_bits[0] == 0.0 else 1
        new_rank = torch.argmax(disc_bits[1:]).item() + 1
        
        self.grid_values[game_idx, player_idx, slot] = new_rank
        self.grid_backs[game_idx, player_idx, slot] = new_back
        
        base = TABLE_START + (player_idx * PLAYER_GRID_SIZE) + (slot * SLOT_SIZE)
        self.state[game_idx, base : base+SLOT_SIZE] = 0.0
        if new_back == 0:
            self.state[game_idx, base] = 1.0 
            self.state[game_idx, base+1] = 0.0
        else:
            self.state[game_idx, base] = 0.0
            self.state[game_idx, base+1] = 1.0
        self.state[game_idx, base + 1 + new_rank] = 1.0
        
        # If we swapped in 2.1, the previous discard wasn't buried yet. Bury it now.
        if self.state[game_idx, STAGE_START+3] == 1.0:
             self._add_current_discard_to_graveyard(game_idx)
             
        self._update_discard_obs(torch.tensor([old_rank], device=self.device), torch.tensor([old_back], device=self.device), indices=[game_idx])

    def _add_current_discard_to_graveyard(self, game_idx):
        disc_bits = self.state[game_idx, DISCARD_START:DISCARD_START+14]
        if disc_bits.sum() == 0: return
        is_blue = (disc_bits[0] == 1.0)
        face_idx = torch.argmax(disc_bits[1:]).item() 
        gy_idx = face_idx + (13 if is_blue else 0)
        self.state[game_idx, GRAVEYARD_START + gy_idx] += 1.0

    def _next_turn(self, game_idx):
        self.current_player[game_idx] = (self.current_player[game_idx] + 1) % 5
        self._update_seat_obs(indices=[game_idx])
        self.state[game_idx, STAGE_START : STAGE_START+5] = 0.0
        self.state[game_idx, STAGE_START + 3] = 1.0

    def _update_discard_obs(self, val, back, indices=None):
        if indices is None: indices = slice(None)
        self.state[indices, DISCARD_START : DISCARD_START+14] = 0.0
        is_blue = (back == 1)
        self.state[indices, DISCARD_START] = is_blue.float()
        offset = DISCARD_START + val
        self.state[indices, offset] = 1.0

    def _update_draw_obs(self, indices=None):
        if indices is None: indices = torch.arange(self.num_envs, device=self.device)
        if len(indices) == 0: return
        ptrs = self.deck_pointers[indices]
        ptrs = torch.clamp(ptrs, max=103)
        next_backs = self.shuffled_backs[indices, ptrs]
        is_blue = (next_backs == 1)
        self.state[indices, DRAW_START] = is_blue.float()

    def _update_seat_obs(self, indices):
        if indices is None: indices = slice(None)
        self.state[indices, SEAT_ID_START : SEAT_ID_START+5] = 0.0
        cols = SEAT_ID_START + self.current_player[indices]
        self.state[indices, cols] = 1.0

    def _check_round_over(self):
        any_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for p in range(5):
             p_start = TABLE_START + (p * PLAYER_GRID_SIZE)
             p_grid = self.state[:, p_start : p_start + PLAYER_GRID_SIZE].view(self.num_envs, 9, SLOT_SIZE)
             face_sums = p_grid[:, :, 2:].sum(dim=2)
             cnt = (face_sums > 0).sum(dim=1)
             any_done |= (cnt == 9)
        return any_done

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
                for r, pts in POINT_MAP.items():
                    col_points[col == r] = pts
                col_sum = col_points.sum(dim=1)
                col_sum = torch.where(match, torch.zeros_like(col_sum), col_sum)
                p_score += col_sum
            scores[:, p] = p_score
        return scores

    def get_action_mask(self):
        mask = torch.zeros((self.num_envs, 10), device=self.device)
        stages = self.state[:, STAGE_START : STAGE_START + 5]
        mask[stages[:, 0] == 1.0, 0:9] = 1.0 # Arrange
        
        flip_mask = (stages[:, 1] == 1.0) | (stages[:, 2] == 1.0)
        if flip_mask.any():
            indices = torch.nonzero(flip_mask).squeeze(1)
            for i in indices:
                p = self.current_player[i].item()
                p_start = TABLE_START + (p * PLAYER_GRID_SIZE)
                for slot in range(9):
                    slot_start = p_start + (slot * SLOT_SIZE)
                    face_bits = self.state[i, slot_start + 2 : slot_start + 15]
                    if face_bits.sum() == 0:
                        mask[i, slot] = 1.0
                
        play_mask = (stages[:, 3] == 1.0) | (stages[:, 4] == 1.0)
        mask[play_mask, 0:10] = 1.0
        return mask
    
    def reset_indices(self, indices):
        self._start_new_round(initial=True, indices=torch.nonzero(indices).squeeze(1))