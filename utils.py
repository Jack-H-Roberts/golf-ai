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
# Note: Market cards are usually known, so Discard likely uses 14 bits (Color + Face)
DISCARD_START = 16
DISCARD_SIZE = 14
DRAW_START = 30
DRAW_SIZE = 1

# Initial Red Bags (5 Players)
BAG_START = 31
BAG_SIZE = 5

# Memory (Graveyard 26)
GRAVEYARD_START = 36
GRAVEYARD_SIZE = 26

# The Table (675 bits total)
# 5 Players * 9 Slots * 15 Bits = 675
# Encoding:
# Bit 0: isRed
# Bit 1: isBlue
# Bit 2-14: isAce through isKing
# Logic: If Sum(Bits 2-14) == 0, the face is Unknown.
TABLE_START = 62 
SLOT_SIZE = 15
PLAYER_GRID_SIZE = 9 * SLOT_SIZE # 135

def get_card_color(face_value):
    """
    Standard Golf Color Mapping:
    Red: 2, 3, 4, 5, 6, 7, 8
    Blue: A, 9, 10, J, Q, K
    Returns: 0 for Red, 1 for Blue.
    """
    if 2 <= face_value <= 8:
        return 0 # Red
    else:
        return 1 # Blue