import torch
from utils import TABLE_START, SLOT_SIZE, PLAYER_GRID_SIZE, get_card_color

class GolfEnv:
    def __init__(self, num_envs=1024, device="cuda"):
        self.num_envs = num_envs
        self.device = device
        
        # [Batch, 737] - The master state we established
        self.state = torch.zeros((num_envs, 737), device=self.device)
        
        # [Batch, 104] - The actual card faces (A=1, K=13) for 2 decks
        # These are hidden from the AI
        self.full_decks = torch.zeros((num_envs, 104), device=self.device, dtype=torch.long)
        
        # Define 2 decks: (A-K is 1-13) x 4 suits x 2 decks = 104 cards
        # We use torch.arange to create a single 104-card deck template
        single_deck = torch.arange(1, 14).repeat(8) # 13 cards * 8 = 104
        self.deck_template = single_deck.to(self.device)

    def reset(self):
        """Prepares 1024 games: Shuffles, deals, and sets initial state."""
        self.state.fill_(0)
        
        # 1. Shuffle all 1024 decks at once
        # torch.rand generates random values; argsort gives us a random permutation
        random_indices = torch.rand((self.num_envs, 104), device=self.device).argsort(dim=1)
        self.full_decks = self.deck_template.expand(self.num_envs, -1).gather(1, random_indices)
        
        # 2. Assign the "Initial Bag" (First player to get 9 cards)
        # For simplicity, we'll deal the first 45 cards (9 per player)
        # TODO: Assign cards to player grids and update the 737-vector
        
        # 3. Set Stage Flag to 'Arrange' [1, 0, 0, 0, 0]
        self.state[:, 0] = 1.0 
        
        return self.state

    def deal_initial_hands(self):
        """
        Deals 9 cards to each of the 5 players for all 1024 games.
        This uses indices 0-44 from the shuffled full_decks.
        """
        # Step 1: Slice the first 45 cards from the deck [1024, 45]
        initial_pool = self.full_decks[:, :45]
        
        # Step 2: Reshape into [1024 games, 5 players, 9 cards each]
        hands = initial_pool.view(self.num_envs, 5, 9)
        
        # Step 3: Populate the 737-vector 'Table' segment
        # Recall our Table starts at a specific index in the 737-vector
        # Let's say the Table starts at index 62 (Seat5 + Stage3 + Score5 + Triggers2 + Discard14 + Draw1 + Bags5 + Memory26)
        # Note: You'll want to define these 'Start Indices' as constants for clarity.
        TABLE_START = 62
        
        # Determine colors and initial state bits
        for p in range(5):
            for c in range(9):
                # Calculate start bit for this slot
                idx = TABLE_START + (p * PLAYER_GRID_SIZE) + (c * SLOT_SIZE)
                
                # Get card value from shuffled deck
                val = hands[:, p, c] # Tensor of 1024 values
                
                # Set the 'Unknown' bit (the 15th bit, index 14)
                self.state[:, idx + 14] = 1.0
                
                # Vectorized color assignment
                # If value is 2-8, set bit 0 (Red). Else, set bit 1 (Blue).
                is_red = (val >= 2) & (val <= 8)
                self.state[is_red, idx] = 1.0     # Set Red bit
                self.state[~is_red, idx + 1] = 1.0 # Set Blue bit

    def step(self, actions):
        """
        Takes a tensor of 1024 actions (integers 0-9).
        Updates the 737-vector and returns (new_state, reward, done).
        """
        # actions is a tensor of shape [1024]
        # Logic for each stage (Arrange, Flip, Play) goes here
        pass

    def get_action_mask(self):
        """
        Returns a binary mask of shape [1024, 10].
        1 = Legal move, 0 = Illegal move.
        """
        mask = torch.ones((self.num_envs, 10), device=self.device)
        # TODO: Use Stage Flags from the 737-vector to set illegal moves to 0
        return mask