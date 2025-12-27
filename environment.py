import torch
from utils import (
    TABLE_START, SLOT_SIZE, PLAYER_GRID_SIZE, 
    BAG_START, STAGE_START, DISCARD_START, 
    GRAVEYARD_START, DRAW_START, SCORES_START,
    SEAT_ID_START, DISCARD_SIZE, TRIGGER_START,
    POINT_MAP, GRAVEYARD_SIZE
)

class GolfEnv:
    def __init__(self, num_envs=4096, device="cuda"):
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
        
        # Finisher Tracker (-1 means no one has finished yet)
        self.finisher = torch.full((num_envs,), -1, device=self.device, dtype=torch.long)
        
        self.grid_values = torch.zeros((num_envs, 5, 9), device=self.device, dtype=torch.long)
        self.grid_backs = torch.zeros((num_envs, 5, 9), device=self.device, dtype=torch.long)
        self.game_scores = torch.zeros((num_envs, 5), device=self.device, dtype=torch.float)
        
        # Templates for fast shuffling
        self.rank_template = torch.cat([torch.arange(1, 14).repeat(4), torch.arange(1, 14).repeat(4)]).to(self.device)
        self.back_template = torch.cat([torch.zeros(52), torch.ones(52)]).long().to(self.device)
        self.shuffled_ranks = torch.zeros((num_envs, 104), device=self.device, dtype=torch.long)
        self.shuffled_backs = torch.zeros((num_envs, 104), device=self.device, dtype=torch.long)
        
        # Constant helpers for vectorization
        self.batch_indices = torch.arange(self.num_envs, device=self.device)
        self.slot_arange = torch.arange(9, device=self.device).expand(self.num_envs, 9)

    def reset(self):
        self.state.fill_(0)
        self.game_scores.fill_(0)
        self.dealer_pos = torch.randint(0, 5, (self.num_envs,), device=self.device)
        self._start_new_round(initial=True, indices=self.batch_indices)
        return self.state

    def _start_new_round(self, initial=False, indices=None):
        if indices is None: indices = self.batch_indices
        n = len(indices)
        if n == 0: return

        if not initial:
            self.dealer_pos[indices] = (self.dealer_pos[indices] + 1) % 5
            scores_backup = self.state[indices, SCORES_START : SCORES_START + 5].clone()
            self.state[indices] = 0.0
            self.state[indices, SCORES_START : SCORES_START + 5] = scores_backup

        # Reset Finisher
        self.finisher[indices] = -1

        # 1. Shuffle
        rand_idx = torch.rand((n, 104), device=self.device).argsort(dim=1)
        self.shuffled_ranks[indices] = self.rank_template.expand(n, -1).gather(1, rand_idx)
        self.shuffled_backs[indices] = self.back_template.expand(n, -1).gather(1, rand_idx)
        
        # 2. Deal
        self.grid_values[indices] = self.shuffled_ranks[indices, :45].view(n, 5, 9)
        self.grid_backs[indices] = self.shuffled_backs[indices, :45].view(n, 5, 9)
        
        # 3. Red Bags
        num_blues = self.grid_backs[indices].sum(dim=2) 
        num_reds = 9 - num_blues
        self.state[indices, BAG_START:BAG_START+5] = num_reds.float()
        
        # 4. Reveal Discard
        # VERIFIED: Uses random card from shuffle (Index 45)
        d_ranks = self.shuffled_ranks[indices, 45]
        d_backs = self.shuffled_backs[indices, 45]
        self._update_discard_obs(d_ranks, d_backs, indices=indices)
        
        # 5. Reveal Draw Color
        self.deck_pointers[indices] = 46
        self._update_draw_obs(indices=indices)
        
        # 6. Setup First Player
        self.init_count[indices] = 0
        self.current_player[indices] = (self.dealer_pos[indices] + 1) % 5
        self._update_seat_obs(indices)
        
        # 7. Set Stage
        self.state[indices, STAGE_START] = 1.0 

    def step(self, actions):
        processed_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # 1. ARRANGE
        is_arrange = (self.state[:, STAGE_START] == 1.0) & (~processed_mask)
        if is_arrange.any():
            self._handle_arrange(actions, is_arrange)
            processed_mask[is_arrange] = True

        # 2. FLIP 1
        is_flip1 = (self.state[:, STAGE_START + 1] == 1.0) & (~processed_mask)
        if is_flip1.any():
            self._handle_flip(actions, is_flip1, stage_idx=1)
            processed_mask[is_flip1] = True
            
        # 3. FLIP 2
        is_flip2 = (self.state[:, STAGE_START + 2] == 1.0) & (~processed_mask)
        if is_flip2.any():
            self._handle_flip(actions, is_flip2, stage_idx=2)
            self._advance_init_phase(is_flip2)
            processed_mask[is_flip2] = True

        # 4. PLAY PHASE
        is_p2_1 = self.state[:, STAGE_START + 3] == 1.0
        is_p2_2 = self.state[:, STAGE_START + 4] == 1.0
        is_playing = (is_p2_1 | is_p2_2) & (~processed_mask)
        
        if is_playing.any():
            self._handle_play(actions, is_playing)
            processed_mask[is_playing] = True

        # --- UPDATE FINISHER STATUS ---
        table = self.state[:, TABLE_START:TABLE_START+675].view(self.num_envs, 5, 9, 15)
        face_sums = table[:, :, :, 2:].sum(dim=3) 
        revealed_counts = (face_sums > 0).sum(dim=2)
        has_nine = (revealed_counts == 9)
        
        update_mask = (self.finisher == -1) & has_nine.any(dim=1)
        if update_mask.any():
            finisher_idx = torch.argmax(has_nine[update_mask].float(), dim=1)
            self.finisher[update_mask] = finisher_idx
            self.state[update_mask, TRIGGER_START] = 1.0

        # --- CHECK ROUND OVER ---
        game_started = (self.init_count == 5)
        round_over_mask = (self.current_player == self.finisher) & (self.finisher != -1) & game_started
        
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

    # --- HANDLERS ---
    def _handle_arrange(self, actions, mask):
        indices = torch.nonzero(mask).squeeze(1)
        if len(indices) == 0: return

        p = self.current_player[indices]
        current_backs = self.grid_backs[indices, p, :] 
        current_vals = self.grid_values[indices, p, :]
        num_reds = (current_backs == 0).sum(dim=1) 
        
        prefs = actions[indices, :9]
        _, sorted_indices = torch.sort(prefs, descending=True, dim=1)
        
        rank_grid = self.slot_arange[:len(indices)] 
        is_red_rank = rank_grid < num_reds.unsqueeze(1)
        is_red_slot = torch.zeros_like(is_red_rank)
        is_red_slot.scatter_(1, sorted_indices, is_red_rank)
        
        back_sort_idx = torch.argsort(current_backs, dim=1) 
        sorted_vals = current_vals.gather(1, back_sort_idx) 
        sorted_backs = current_backs.gather(1, back_sort_idx)

        new_vals = torch.zeros_like(current_vals)
        new_vals.scatter_(1, sorted_indices, sorted_vals)
        new_backs = torch.zeros_like(current_backs)
        new_backs.scatter_(1, sorted_indices, sorted_backs)
        
        self.grid_values[indices, p, :] = new_vals
        self.grid_backs[indices, p, :] = new_backs
        
        p_start = TABLE_START + (p * PLAYER_GRID_SIZE)
        base_indices = p_start.unsqueeze(1) + (self.slot_arange[:len(indices)] * SLOT_SIZE)
        
        flat_bases = base_indices.flatten()
        flat_reds = new_backs.flatten() == 0
        row_idx = indices.unsqueeze(1).expand(-1, 9).flatten()
        
        self.state[row_idx, flat_bases] = 0.0
        self.state[row_idx, flat_bases + 1] = 0.0
        self.state[row_idx[flat_reds], flat_bases[flat_reds]] = 1.0     
        self.state[row_idx[~flat_reds], flat_bases[~flat_reds] + 1] = 1.0 
        
        self.state[mask, STAGE_START] = 0.0
        self.state[mask, STAGE_START+1] = 1.0

    def _handle_flip(self, actions, mask, stage_idx):
        indices = torch.nonzero(mask).squeeze(1)
        if len(indices) == 0: return

        choices = torch.argmax(actions[indices, :9], dim=1) 
        p = self.current_player[indices]
        ranks = self.grid_values[indices, p, choices] 
        bases = TABLE_START + (p * PLAYER_GRID_SIZE) + (choices * SLOT_SIZE)
        target_bits = bases + 1 + ranks
        self.state[indices, target_bits] = 1.0
        
        curr_s = STAGE_START + stage_idx
        self.state[mask, curr_s] = 0.0
        self.state[mask, curr_s + 1] = 1.0

    def _advance_init_phase(self, mask):
        indices = torch.nonzero(mask).squeeze(1)
        if len(indices) == 0: return

        self.init_count[indices] += 1
        
        cont_mask = (self.init_count[indices] < 5)
        if cont_mask.any():
            c_idx = indices[cont_mask]
            self.current_player[c_idx] = (self.current_player[c_idx] + 1) % 5
            self._update_seat_obs(c_idx)
            self.state[c_idx, STAGE_START : STAGE_START+5] = 0.0
            self.state[c_idx, STAGE_START] = 1.0
            
        start_mask = (self.init_count[indices] == 5)
        if start_mask.any():
            s_idx = indices[start_mask]
            self.current_player[s_idx] = (self.dealer_pos[s_idx] + 1) % 5
            self._update_seat_obs(s_idx)
            self.state[s_idx, STAGE_START : STAGE_START+5] = 0.0
            self.state[s_idx, STAGE_START+3] = 1.0

    def _handle_play(self, actions, mask):
        all_indices = torch.nonzero(mask).squeeze(1)
        if len(all_indices) == 0: return

        act_inds = torch.argmax(actions[all_indices], dim=1)
        is_2_1 = (self.state[all_indices, STAGE_START + 3] == 1.0)
        is_2_2 = ~is_2_1
        
        if is_2_1.any():
            idx_21 = all_indices[is_2_1]
            acts_21 = act_inds[is_2_1]
            draw_mask = (acts_21 == 9)
            if draw_mask.any():
                # PASS: Discard old, Draw new
                d_idx = idx_21[draw_mask]
                ptr = self.deck_pointers[d_idx]
                ptr = torch.remainder(ptr, 104) 
                
                drawn_rank = self.shuffled_ranks[d_idx, ptr]
                drawn_back = self.shuffled_backs[d_idx, ptr]
                self.deck_pointers[d_idx] += 1
                
                # CORRECT: Bury the *current* discard card before replacing it
                self._add_current_discard_to_graveyard(d_idx)
                self._update_discard_obs(drawn_rank, drawn_back, indices=d_idx)
                self._update_draw_obs(indices=d_idx)
                
                self.state[d_idx, STAGE_START + 3] = 0.0
                self.state[d_idx, STAGE_START + 4] = 1.0
                
            swap_mask = ~draw_mask
            if swap_mask.any():
                s_idx = idx_21[swap_mask]
                slots = acts_21[swap_mask]
                p = self.current_player[s_idx]
                self._swap_card_vectorized(s_idx, p, slots)
                self._next_turn_vectorized(s_idx)
        
        if is_2_2.any():
            idx_22 = all_indices[is_2_2]
            acts_22 = act_inds[is_2_2]
            pass_mask = (acts_22 == 9)
            if pass_mask.any():
                p_idx = idx_22[pass_mask]
                self._next_turn_vectorized(p_idx)
                
            swap_mask = ~pass_mask
            if swap_mask.any():
                s_idx = idx_22[swap_mask]
                slots = acts_22[swap_mask]
                p = self.current_player[s_idx]
                self._swap_card_vectorized(s_idx, p, slots)
                self._next_turn_vectorized(s_idx)

    def _swap_card_vectorized(self, indices, p, slots):
        old_rank = self.grid_values[indices, p, slots]
        old_back = self.grid_backs[indices, p, slots]
        
        disc_vec = self.state[indices, DISCARD_START:DISCARD_START+14]
        new_back = (disc_vec[:, 0] == 1.0).long()
        new_rank = torch.argmax(disc_vec[:, 1:], dim=1) + 1
        
        self.grid_values[indices, p, slots] = new_rank
        self.grid_backs[indices, p, slots] = new_back
        
        bases = TABLE_START + (p * PLAYER_GRID_SIZE) + (slots * SLOT_SIZE)
        row_idx = indices
        
        for k in range(15):
            self.state[row_idx, bases + k] = 0.0
            
        self.state[row_idx, bases] = (new_back == 0).float() 
        self.state[row_idx, bases + 1] = (new_back == 1).float() 
        self.state[row_idx, bases + 1 + new_rank] = 1.0
        
        # BUG FIX: Removed Graveyard update here.
        # Swapping (Taking the card) does NOT bury it.
            
        self._update_discard_obs(old_rank, old_back, indices=indices)

    def _add_current_discard_to_graveyard(self, indices):
        disc_vec = self.state[indices, DISCARD_START:DISCARD_START+14]
        has_card = (disc_vec.sum(dim=1) > 0)
        valid_idx = indices[has_card]
        if len(valid_idx) == 0: return
        valid_vec = disc_vec[has_card]
        is_blue = (valid_vec[:, 0] == 1.0)
        face_idx = torch.argmax(valid_vec[:, 1:], dim=1) 
        gy_idx = face_idx + (13 * is_blue.long())
        flat_gy_idx = GRAVEYARD_START + gy_idx
        self.state[valid_idx, flat_gy_idx] += 1.0

    def _next_turn_vectorized(self, indices):
        self.current_player[indices] = (self.current_player[indices] + 1) % 5
        self._update_seat_obs(indices)
        self.state[indices, STAGE_START : STAGE_START+5] = 0.0
        self.state[indices, STAGE_START + 3] = 1.0

    def _update_discard_obs(self, val, back, indices=None):
        if indices is None: indices = self.batch_indices
        self.state[indices, DISCARD_START : DISCARD_START+14] = 0.0
        is_blue = (back == 1)
        self.state[indices, DISCARD_START] = is_blue.float()
        flat_idx = DISCARD_START + val
        self.state[indices, flat_idx] = 1.0

    def _update_draw_obs(self, indices=None):
        if indices is None: indices = self.batch_indices
        if len(indices) == 0: return
        ptrs = self.deck_pointers[indices]
        ptrs = torch.clamp(ptrs, max=103)
        next_backs = self.shuffled_backs[indices, ptrs]
        is_blue = (next_backs == 1)
        self.state[indices, DRAW_START] = is_blue.float()

    def _update_seat_obs(self, indices):
        if indices is None: indices = self.batch_indices
        self.state[indices, SEAT_ID_START : SEAT_ID_START+5] = 0.0
        cols = SEAT_ID_START + self.current_player[indices]
        self.state[indices, cols] = 1.0

    def _check_round_over(self):
        table = self.state[:, TABLE_START:TABLE_START+675].view(self.num_envs, 5, 9, 15)
        face_sums = table[:, :, :, 2:].sum(dim=3)
        revealed_counts = (face_sums > 0).sum(dim=2)
        any_done = (revealed_counts == 9).any(dim=1) 
        return any_done

    def _calculate_round_scores(self):
        scores = torch.zeros((self.num_envs, 5), device=self.device)
        for c_idx, cols in enumerate([(0,3,6), (1,4,7), (2,5,8)]):
            col_stack = torch.stack([self.grid_values[:, :, i] for i in cols], dim=2)
            match = (col_stack[:, :, 0] == col_stack[:, :, 1]) & (col_stack[:, :, 1] == col_stack[:, :, 2])
            
            if not hasattr(self, 'point_lookup'):
                self.point_lookup = torch.zeros(14, device=self.device)
                for k,v in POINT_MAP.items():
                    self.point_lookup[k] = v
            
            pts = self.point_lookup[col_stack.long()] 
            col_sum = pts.sum(dim=2)
            col_sum = torch.where(match, torch.zeros_like(col_sum), col_sum)
            scores += col_sum
            
        return scores

    def get_action_mask(self):
        mask = torch.zeros((self.num_envs, 10), device=self.device)
        stages = self.state[:, STAGE_START : STAGE_START + 5]
        
        mask[stages[:, 0] == 1.0, 0:9] = 1.0 
        
        flip_mask = (stages[:, 1] == 1.0) | (stages[:, 2] == 1.0)
        if flip_mask.any():
            indices = torch.nonzero(flip_mask).squeeze(1)
            p = self.current_player[indices]
            
            table = self.state[indices, TABLE_START:TABLE_START+675].view(len(indices), 5, 9, 15)
            p_ex = p.view(-1, 1, 1, 1).expand(-1, 1, 9, 15)
            p_grid = table.gather(1, p_ex).squeeze(1) 
            
            face_sum = p_grid[:, :, 2:].sum(dim=2) 
            is_hidden = (face_sum == 0)
            mask[indices, 0:9] = is_hidden.float()
            
        play_mask = (stages[:, 3] == 1.0) | (stages[:, 4] == 1.0)
        mask[play_mask, 0:10] = 1.0
        
        return mask
    
    def reset_indices(self, indices):
        idx = torch.nonzero(indices).squeeze(1)
        self._start_new_round(initial=True, indices=idx)