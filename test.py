from environment import GolfEnv

env = GolfEnv(num_envs=1024)
env.reset()
env.deal_initial_hands()
env.debug_print_hand(player_idx=0)