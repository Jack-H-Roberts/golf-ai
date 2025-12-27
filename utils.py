# Feature indices (Stage1, Stage2, Stage3, Stage4, Stage5)
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

# Market (Top Discard 14, Top Draw 1)
DISCARD_START = 16
DISCARD_SIZE = 14
DRAW_START = 30
DRAW_SIZE = 1

# Memory (Graveyard 26)
GRAVEYARD_START = 31
GRAVEYARD_SIZE = 26

# The Table (675 bits total)
# 5 Players * 9 Slots * 15 Bits
TABLE_START = 57 
SLOT_SIZE = 15
PLAYER_GRID_SIZE = 9 * SLOT_SIZE # 135

def get_card_color(face_value):
    """
    Standard Golf: 
    Red: 2, 3, 4, 5, 6, 7, 8 (point values vary)
    Blue: A, 9, 10, J, Q, K
    Note: You should adjust this based on your specific deck's rules.
    """
    # Example logic: 2-8 are Red (0), others Blue (1)
    if 2 <= face_value <= 8:
        return 0 # Red
    else:
        return 1 # Blue