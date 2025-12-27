# Feature indices (Stage1-5)
STAGE_START = 0
STAGE_SIZE = 5

# Seat ID (Player 0 - Player 4)
SEAT_ID_START = 5
SEAT_ID_SIZE = 5

# Game Scores (5 Players)
SCORES_START = 10
SCORES_SIZE = 5

# Triggers (Final Turn Flag)
TRIGGER_START = 15
TRIGGER_SIZE = 1

# Market (Top Discard 14 bits, Top Draw 1 bit)
DISCARD_START = 16
DISCARD_SIZE = 14
DRAW_START = 30
DRAW_SIZE = 1

# Initial Red Bags (5 Players) - Count of Red BACKED cards
BAG_START = 31
BAG_SIZE = 5

# Memory (Graveyard 26) - 13 Ranks * 2 Back Colors
# Indices 0-12: Red Backs (Ace-King)
# Indices 13-25: Blue Backs (Ace-King)
GRAVEYARD_START = 36
GRAVEYARD_SIZE = 26

# The Table (675 bits total)
TABLE_START = 62 
SLOT_SIZE = 15
PLAYER_GRID_SIZE = 9 * SLOT_SIZE 

# --- SCORING LOOKUP ---
# Maps Face Value (1-13) to Points
# A=1, 2=2... 8=-2, 9=9, 10=10, J=10, Q=10, K=0
POINT_MAP = {
    1: 1,   # Ace
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: -2,  # Special Rule
    9: 9,
    10: 10,
    11: 10, # Jack
    12: 10, # Queen
    13: 0   # King
}

def get_card_color(is_blue_back):
    """
    Returns 1.0 if Blue Back, 0.0 if Red Back.
    Helper for consistency.
    """
    return 1.0 if is_blue_back else 0.0