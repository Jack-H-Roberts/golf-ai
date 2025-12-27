import torch
from utils import (
    TABLE_START, SLOT_SIZE, PLAYER_GRID_SIZE, 
    BAG_START, STAGE_START, DISCARD_START, 
    GRAVEYARD_START, DRAW_START, SCORES_START,
    SEAT_ID_START, DISCARD_SIZE, 
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
        self._start_new_round(initial=True)
        return self.state

    def _start_new_round(self, initial=False, indices=None):
        if indices is None: indices = self.batch_indices
        n = len(indices)
        if n == 0: return

        if not initial:
            self.dealer_pos[indices] = (self.dealer_pos[indices] + 1) % 5
            # Zero out state but preserve scores
            scores_backup = self.state[indices, SCORES_START : SCORES_START + 5].clone()
            self.state[indices] = 0.0
            self.state[indices, SCORES_START : SCORES_START + 5] = scores_backup

        # 1. Shuffle
        # Create random sort keys
        rand_idx = torch.rand((n, 104), device=self.device).argsort(dim=1)
        
        # Gather using advanced indexing
        self.shuffled_ranks[indices] = self.rank_template.expand(n, -1).gather(1, rand_idx)
        self.shuffled_backs[indices] = self.back_template.expand(n, -1).gather(1, rand_idx)
        
        # 2. Deal 9 cards (Indices 0-44)
        self.grid_values[indices] = self.shuffled_ranks[indices, :45].view(n, 5, 9)
        self.grid_backs[indices] = self.shuffled_backs[indices, :45].view(n, 5, 9)
        
        # 3. Populate Initial Red Bags (Visible immediately)
        # Vectorized sum over the 5 players
        # grid_backs is [N, 5, 9]. Sum over last dim gives blues.
        num_blues = self.grid_backs[indices].sum(dim=2) # [N, 5]
        num_reds = 9 - num_blues
        # Flatten to put into state
        self.state[indices, BAG_START:BAG_START+5] = num_reds.float()
        
        # 4. REVEAL Discard (Card 45)
        d_ranks = self.shuffled_ranks[indices, 45]
        d_backs = self.shuffled_backs[indices, 45]
        self._update_discard_obs(d_ranks, d_backs, indices=indices)
        
        # 5. REVEAL Draw Color (Card 46)
        self.deck_pointers[indices] = 46
        self._update_draw_obs(indices=indices)
        
        # 6. Setup First Player
        self.init_count[indices] = 0
        self.current_player[indices] = (self.dealer_pos[indices] + 1) % 5
        self._update_seat_obs(indices)
        
        # 7. Set Stage to Arrange
        self.state[indices, STAGE_START] = 1.0 

    def step(self, actions):
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
            self._advance_init_phase(is_flip2)

        # 4. PLAY PHASE
        is_p2_1 = self.state[:, STAGE_START + 3] == 1.0
        is_p2_2 = self.state[:, STAGE_START + 4] == 1.0
        is_playing = is_p2_1 | is_p2_2
        
        if is_playing.any():
            self._handle_play(actions, is_playing)

        # --- CHECK ROUND END ---
        # Vectorized logic:
        # A game is started if init_count == 5.
        game_started = (self.init_count == 5)
        
        # Check round over
        round_over_mask = self._check_round_over() & game_started
        
        rewards = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        if round_over_mask.any():
            round_scores = self._calculate_round_scores()
            self.game_scores[round_over_mask] += round_scores[round_over_mask]
            self.state[round_over_mask, SCORES_START : SCORES_START+5] = self.game_scores[round_over_mask] / 100.0
            
            # Check for >100
            max_scores, _ = self.game_scores.max(dim=1)
            game_over = (max_scores >= 100) & round_over_mask
            
            if game_over.any():
                dones[game_over] = True
                final_s = self.game_scores[game_over]
                rewards[game_over] = (final_s[:, 1:].mean(dim=1) - final_s[:, 0]) / 10.0
            
            # Reset rounds that ended but game isn't over
            reset_round_mask = round_over_mask & (~game_over)
            if reset_round_mask.any():
                indices = torch.nonzero(reset_round_mask).squeeze(1)
                self._start_new_round(initial=False, indices=indices)

        return self.state, rewards, dones, {}

    # --- VECTORIZED HANDLERS ---

    def _handle_arrange(self, actions, mask):
        indices = torch.nonzero(mask).squeeze(1)
        if len(indices) == 0: return

        p = self.current_player[indices] # [N]
        
        # Get Current Backs for active player: [N, 9]
        # We need advanced indexing: [indices, p, :]
        current_backs = self.grid_backs[indices, p, :] 
        current_vals = self.grid_values[indices, p, :]
        
        # Count reds per player: (backs==0)
        num_reds = (current_backs == 0).sum(dim=1) # [N]
        
        # Get Preferences: [N, 9]
        prefs = actions[indices, :9]
        
        # Sort preferences to determine which slots get the reds
        # argsort descending: index 0 is the highest pref slot
        _, sorted_indices = torch.sort(prefs, descending=True, dim=1)
        
        # Determine Red/Blue assignment for each rank (0..8)
        # If rank < num_reds, it gets a Red card.
        # We create a mask of "is_red_rank" [N, 9]
        rank_grid = self.slot_arange[:len(indices)] # [N, 9] 0..8
        is_red_rank = rank_grid < num_reds.unsqueeze(1) # [N, 9] Bool
        
        # Now we map this back to SLOTS using sorted_indices.
        # We want a tensor 'is_red_slot' [N, 9]
        # We can use scatter_.
        is_red_slot = torch.zeros_like(is_red_rank)
        # If sorted_indices[b, 0] is the slot ID for highest pref...
        # and is_red_rank[b, 0] is True...
        # we want is_red_slot[b, slot_ID] = True.
        is_red_slot.scatter_(1, sorted_indices, is_red_rank)
        
        # Now we have the TARGET arrangement mask.
        # We need to pull the actual values.
        # We split current values into Reds and Blues.
        # This is tricky vectorized because order matters? 
        # Actually, "shuffle hidden values" implies order doesn't matter among same color.
        # We gather all Reds and all Blues, then scatter them into the new red/blue slots.
        
        # 1. Sort current values so Reds (0) come first, then Blues (1)
        # We sort by back color.
        back_sort_idx = torch.argsort(current_backs, dim=1) # Reds first
        sorted_vals = current_vals.gather(1, back_sort_idx) # [N, 9] with Reds on left
        
        # 2. Now we place these sorted values into the target slots.
        # The target slots for Reds are where is_red_slot is True.
        # The target slots for Blues are where is_red_slot is False.
        # We can construct a "destination index" map.
        # We want the first Red from sorted_vals to go to the first True in is_red_slot.
        
        # Helper: We just need to know which slot gets the 0th sorted_val, which gets 1st...
        # is_red_slot tells us the pattern, e.g. [0, 1, 1, 0...]
        # sorted_vals is [R, R, B, B...]
        # We need to invert the mapping.
        # Actually, if we just assign the red slots to 0 (Red) and blue to 1 (Blue),
        # And assign the values similarly...
        
        # Optimization: Just overwrite grid_backs based on the decision.
        new_backs = torch.where(is_red_slot, torch.zeros_like(current_backs), torch.ones_like(current_backs))
        
        # For Values: logic requires us to move the specific card values.
        # Since we sorted `sorted_vals` to have Reds first, and `is_red_rank` has Trues first,
        # we can just scatter `sorted_vals` back into position!
        # wait: is_red_rank is [T, T, F, F]. sorted_vals is [R, R, B, B].
        # We scattered is_red_rank to created is_red_slot.
        # If we scatter sorted_vals using the SAME indices...
        # No, sorted_indices was based on PREFS.
        # We want to place `sorted_vals[0]` (a red card) into `sorted_indices[0]`.
        # We want to place `sorted_vals[k]` into `sorted_indices[k]`.
        
        new_vals = torch.zeros_like(current_vals)
        new_vals.scatter_(1, sorted_indices, sorted_vals)
        
        # We also need to scatter the backs (which are 0 for Reds, 1 for Blues)
        # sorted_backs should correspond to sorted_vals (0s then 1s)
        sorted_backs = current_backs.gather(1, back_sort_idx)
        new_backs = torch.zeros_like(current_backs)
        new_backs.scatter_(1, sorted_indices, sorted_backs)
        
        # Update Truth
        self.grid_values[indices, p, :] = new_vals
        self.grid_backs[indices, p, :] = new_backs
        
        # Update Obs
        p_start = TABLE_START + (p * PLAYER_GRID_SIZE)
        
        # Create a view of the state to batch-update
        # We need to construct the 135-len vector for each player
        # Or just update Red/Blue bits
        
        # Base indices for the player's grid in flat state
        # shape [N, 9]
        base_indices = p_start.unsqueeze(1) + (self.slot_arange[:len(indices)] * SLOT_SIZE)
        
        # Clear Red/Blue/Face bits? 
        # Easier to just zero out the whole grid area
        # We can't easily slice arbitrary bits in flat vector without loop or complex scatter.
        # BUT, since we are only setting Red/Blue bits (Face is 0), we can just set specific bits.
        
        # Zero out the Red/Blue bits (Offsets 0 and 1)
        # We can use scatter on the state
        # Actually, simpler: 
        # state[idx, base + 0] = is_red
        # state[idx, base + 1] = ~is_red
        
        # Flatten for scatter
        flat_bases = base_indices.flatten() # [N*9]
        flat_reds = new_backs.flatten() == 0
        
        # We need row indices for scatter
        row_idx = indices.unsqueeze(1).expand(-1, 9).flatten()
        
        # Reset bits 0 and 1
        self.state[row_idx, flat_bases] = 0.0
        self.state[row_idx, flat_bases + 1] = 0.0
        
        # Set new bits
        # red_idx = flat_bases where flat_reds is True
        self.state[row_idx[flat_reds], flat_bases[flat_reds]] = 1.0     # Bit 0
        self.state[row_idx[~flat_reds], flat_bases[~flat_reds] + 1] = 1.0 # Bit 1
        
        # Advance Stage
        self.state[mask, STAGE_START] = 0.0
        self.state[mask, STAGE_START+1] = 1.0

    def _handle_flip(self, actions, mask, stage_idx):
        indices = torch.nonzero(mask).squeeze(1)
        if len(indices) == 0: return

        choices = torch.argmax(actions[indices, :9], dim=1) # [N]
        p = self.current_player[indices]
        
        # Get rank: grid_values[indices, p, choices]
        # Advanced indexing
        ranks = self.grid_values[indices, p, choices] # [N]
        
        # Calculate State Index to update
        # Base = TableStart + PlayerOffset + SlotOffset
        bases = TABLE_START + (p * PLAYER_GRID_SIZE) + (choices * SLOT_SIZE)
        
        # Bit to set = Base + 1 (Skip Color) + Rank
        target_bits = bases + 1 + ranks
        
        self.state[indices, target_bits] = 1.0
        
        # Advance
        curr_s = STAGE_START + stage_idx
        self.state[mask, curr_s] = 0.0
        self.state[mask, curr_s + 1] = 1.0

    def _advance_init_phase(self, mask):
        indices = torch.nonzero(mask).squeeze(1)
        if len(indices) == 0: return

        self.init_count[indices] += 1
        
        # 1. Continue Init
        cont_mask = (self.init_count[indices] < 5)
        if cont_mask.any():
            c_idx = indices[cont_mask]
            self.current_player[c_idx] = (self.current_player[c_idx] + 1) % 5
            self._update_seat_obs(c_idx)
            self.state[c_idx, STAGE_START : STAGE_START+5] = 0.0
            self.state[c_idx, STAGE_START] = 1.0
            
        # 2. Finish Init
        start_mask = (self.init_count[indices] == 5)
        if start_mask.any():
            s_idx = indices[start_mask]
            self.current_player[s_idx] = (self.dealer_pos[s_idx] + 1) % 5
            self._update_seat_obs(s_idx)
            self.state[s_idx, STAGE_START : STAGE_START+5] = 0.0
            self.state[s_idx, STAGE_START+3] = 1.0

    def _handle_play(self, actions, mask):
        # Vectorized Play Handler
        all_indices = torch.nonzero(mask).squeeze(1)
        if len(all_indices) == 0: return

        act_inds = torch.argmax(actions[all_indices], dim=1)
        
        # Mask for 2.1 vs 2.2
        is_2_1 = (self.state[all_indices, STAGE_START + 3] == 1.0)
        is_2_2 = ~is_2_1
        
        # --- PHASE 2.1 ---
        if is_2_1.any():
            idx_21 = all_indices[is_2_1]
            acts_21 = act_inds[is_2_1]
            
            # Action 9: Draw
            draw_mask = (acts_21 == 9)
            if draw_mask.any():
                d_idx = idx_21[draw_mask]
                
                # Logic: ptr++, update obs
                ptr = self.deck_pointers[d_idx]
                # Wrap safe
                ptr = torch.remainder(ptr, 104) 
                
                drawn_rank = self.shuffled_ranks[d_idx, ptr]
                drawn_back = self.shuffled_backs[d_idx, ptr]
                self.deck_pointers[d_idx] += 1
                
                self._add_current_discard_to_graveyard(d_idx)
                self._update_discard_obs(drawn_rank, drawn_back, indices=d_idx)
                self._update_draw_obs(indices=d_idx)
                
                # Advance to 2.2
                self.state[d_idx, STAGE_START + 3] = 0.0
                self.state[d_idx, STAGE_START + 4] = 1.0
                
            # Action 0-8: Swap Discard
            swap_mask = ~draw_mask
            if swap_mask.any():
                s_idx = idx_21[swap_mask]
                slots = acts_21[swap_mask]
                p = self.current_player[s_idx]
                
                self._swap_card_vectorized(s_idx, p, slots)
                self._next_turn_vectorized(s_idx)
        
        # --- PHASE 2.2 ---
        if is_2_2.any():
            idx_22 = all_indices[is_2_2]
            acts_22 = act_inds[is_2_2]
            
            # Action 9: Discard Drawn (Pass)
            pass_mask = (acts_22 == 9)
            if pass_mask.any():
                p_idx = idx_22[pass_mask]
                self._next_turn_vectorized(p_idx)
                
            # Action 0-8: Swap Drawn
            swap_mask = ~pass_mask
            if swap_mask.any():
                s_idx = idx_22[swap_mask]
                slots = acts_22[swap_mask]
                p = self.current_player[s_idx]
                
                self._swap_card_vectorized(s_idx, p, slots)
                self._next_turn_vectorized(s_idx)

    def _swap_card_vectorized(self, indices, p, slots):
        # Gather Old Values
        old_rank = self.grid_values[indices, p, slots]
        old_back = self.grid_backs[indices, p, slots]
        
        # Decode Discard Obs
        # Discard start is [N, 14]
        disc_vec = self.state[indices, DISCARD_START:DISCARD_START+14]
        new_back = (disc_vec[:, 0] == 0.0).long() # 0 if Red (bit 0 is 0?), wait.
        # Utils says: Bit 0 is Color (0=Red?? No).
        # _update_discard_obs says: is_blue.float() -> state[0].
        # So Bit 0 = 1.0 means Blue. 
        # new_back should be 1 if Blue.
        new_back = (disc_vec[:, 0] == 1.0).long()
        
        # Decode Rank
        # bits 1..13. argmax gives 0..12. +1 gives 1..13.
        new_rank = torch.argmax(disc_vec[:, 1:], dim=1) + 1
        
        # Update Grid Truth
        self.grid_values[indices, p, slots] = new_rank
        self.grid_backs[indices, p, slots] = new_back
        
        # Update Grid Obs
        bases = TABLE_START + (p * PLAYER_GRID_SIZE) + (slots * SLOT_SIZE)
        
        # Zero out
        # We need row indices
        row_idx = indices
        # We can't use simple slicing for variable offsets.
        # Use scatter or loop? scatter is better.
        # But we are just setting specific bits.
        
        # Actually, simpler: Zero out the 15 bits for these slots.
        # Since SLOT_SIZE=15 is small, we can loop 15 times to zero?
        # Or construct a mask.
        # For speed: Just zero out the Red/Blue/Rank bits we know are likely set.
        # We can just write 0 to all 15 indices?
        for k in range(15):
            self.state[row_idx, bases + k] = 0.0
            
        # Set New Bits
        # Color
        self.state[row_idx, bases] = (new_back == 0).float() # Red
        self.state[row_idx, bases + 1] = (new_back == 1).float() # Blue
        # Face
        self.state[row_idx, bases + 1 + new_rank] = 1.0
        
        # Check Bury Condition (If coming from 2.1)
        # We need to know if we are in 2.1. We can check indices state.
        is_21 = (self.state[indices, STAGE_START + 3] == 1.0)
        if is_21.any():
            self._add_current_discard_to_graveyard(indices[is_21])
            
        # Old Card -> Discard Obs
        self._update_discard_obs(old_rank, old_back, indices=indices)

    def _add_current_discard_to_graveyard(self, indices):
        disc_vec = self.state[indices, DISCARD_START:DISCARD_START+14]
        has_card = (disc_vec.sum(dim=1) > 0)
        
        valid_idx = indices[has_card]
        if len(valid_idx) == 0: return
        
        valid_vec = disc_vec[has_card]
        is_blue = (valid_vec[:, 0] == 1.0)
        face_idx = torch.argmax(valid_vec[:, 1:], dim=1) # 0-12
        
        gy_idx = face_idx + (13 * is_blue.long())
        
        # Update Graveyard (Indices are valid_idx)
        # We use scatter_add_ or index add.
        # state[idx, GRAVEYARD_START + gy_idx] += 1
        
        # Linear index in state
        flat_gy_idx = GRAVEYARD_START + gy_idx
        # We can use specialized scatter if indices are unique, but here they are unique games.
        # Just loop? No, batch index.
        # self.state[valid_idx, flat_gy_idx] += 1.0 # This doesn't work directly if shape mismatch
        
        # Correct way:
        self.state[valid_idx, flat_gy_idx] += 1.0

    def _next_turn_vectorized(self, indices):
        self.current_player[indices] = (self.current_player[indices] + 1) % 5
        self._update_seat_obs(indices)
        self.state[indices, STAGE_START : STAGE_START+5] = 0.0
        self.state[indices, STAGE_START + 3] = 1.0

    def _update_discard_obs(self, val, back, indices=None):
        if indices is None: indices = self.batch_indices
        # Zero out
        self.state[indices, DISCARD_START : DISCARD_START+14] = 0.0
        
        # Color
        is_blue = (back == 1)
        self.state[indices, DISCARD_START] = is_blue.float()
        
        # Face
        # offset = val (1..13)
        # scatter 1.0 at index + val
        flat_idx = DISCARD_START + val
        self.state[indices].scatter_(1, flat_idx.unsqueeze(1), 1.0)

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
        self.state[indices].scatter_(1, cols.unsqueeze(1), 1.0)

    def _check_round_over(self):
        # Vectorized check for all 5 players
        # Reshape grid area to [N, 5, 9, 15]
        # TABLE_START = 62. Size = 675.
        table = self.state[:, TABLE_START:TABLE_START+675].view(self.num_envs, 5, 9, 15)
        # Face bits are 2..14
        face_sums = table[:, :, :, 2:].sum(dim=3) # [N, 5, 9]
        revealed_counts = (face_sums > 0).sum(dim=2) # [N, 5]
        
        any_done = (revealed_counts == 9).any(dim=1) # [N]
        return any_done

    def _calculate_round_scores(self):
        scores = torch.zeros((self.num_envs, 5), device=self.device)
        # Vectorized scoring
        # grid_values [N, 5, 9]
        for c_idx, cols in enumerate([(0,3,6), (1,4,7), (2,5,8)]):
            # Form column stack [N, 5, 3]
            col_stack = torch.stack([self.grid_values[:, :, i] for i in cols], dim=2)
            
            # Match check: all 3 equal
            match = (col_stack[:, :, 0] == col_stack[:, :, 1]) & (col_stack[:, :, 1] == col_stack[:, :, 2])
            
            # Map points (Vectorized map is hard, use gather)
            # POINT_MAP is dict. Convert to tensor.
            # 0..13 (Index 0 is unused)
            # Create lookup tensor once
            if not hasattr(self, 'point_lookup'):
                self.point_lookup = torch.zeros(14, device=self.device)
                for k,v in POINT_MAP.items():
                    self.point_lookup[k] = v
            
            # Lookup points
            pts = self.point_lookup[col_stack.long()] # [N, 5, 3]
            col_sum = pts.sum(dim=2)
            
            # Apply match rule (0 points)
            col_sum = torch.where(match, torch.zeros_like(col_sum), col_sum)
            
            scores += col_sum
            
        return scores

    def get_action_mask(self):
        mask = torch.zeros((self.num_envs, 10), device=self.device)
        stages = self.state[:, STAGE_START : STAGE_START + 5]
        
        # 1. Arrange
        mask[stages[:, 0] == 1.0, 0:9] = 1.0 
        
        # 2. Flips
        flip_mask = (stages[:, 1] == 1.0) | (stages[:, 2] == 1.0)
        if flip_mask.any():
            # We need to find hidden slots for CURRENT PLAYER
            indices = torch.nonzero(flip_mask).squeeze(1)
            p = self.current_player[indices]
            
            # Get table slice for current players
            # We can't easily slice [N, 1, 135] with variable offsets efficiently without gather.
            # But we can reconstruct the check.
            # Hidden check: Face sum == 0.
            
            # grid_values is [N, 5, 9]. We don't track revealed status in grid_values.
            # We track it in state.
            
            # View state table as [N, 5, 9, 15]
            table = self.state[indices, TABLE_START:TABLE_START+675].view(len(indices), 5, 9, 15)
            # Gather player p
            # p is [N]. Expand to [N, 1, 9, 15]
            p_ex = p.view(-1, 1, 1, 1).expand(-1, 1, 9, 15)
            p_grid = table.gather(1, p_ex).squeeze(1) # [N, 9, 15]
            
            face_sum = p_grid[:, :, 2:].sum(dim=2) # [N, 9]
            is_hidden = (face_sum == 0)
            
            # Update mask
            # mask[indices, 0..8] = is_hidden
            mask[indices, 0:9] = is_hidden.float()
            
        # 3. Play
        play_mask = (stages[:, 3] == 1.0) | (stages[:, 4] == 1.0)
        mask[play_mask, 0:10] = 1.0
        
        return mask
    
    def reset_indices(self, indices):
        # Indices is bool mask
        idx = torch.nonzero(indices).squeeze(1)
        self._start_new_round(initial=True, indices=idx)