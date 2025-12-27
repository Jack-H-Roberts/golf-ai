import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
import os

from environment import GolfEnv
from model import GolfModel

# --- SELF-PLAY SETTINGS ---
TOTAL_UPDATES = 2000      
NUM_ENVS = 4096           
NUM_STEPS = 128           # Shorter rollouts for faster updates
BATCH_SIZE = NUM_ENVS * NUM_STEPS # This is the theoretical max, actual buffer will be smaller due to filtering
MINIBATCH_SIZE = 16384    
LEARNING_RATE = 2.5e-4
SAVE_FILENAME = "latest_model.pt"
OPPONENT_UPDATE_FREQ = 100 # Every 50 updates, the opponent becomes the current agent

def train_selfplay():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- STARTING SELF-PLAY ON {device} ---")
    
    # 1. Init Envs
    env = GolfEnv(num_envs=NUM_ENVS, device=device)
    
    # 2. Init Agent (Learner - Seat 0)
    agent = GolfModel().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    # Load previous weights if available
    if os.path.exists(SAVE_FILENAME):
        print(f"Loading {SAVE_FILENAME} for Agent...")
        ckpt = torch.load(SAVE_FILENAME, map_location=device)
        agent.load_state_dict(ckpt['model_state_dict'])

    # 3. Init Opponent (Seats 1-4)
    opponent = copy.deepcopy(agent)
    opponent.eval() # Opponent never learns
    print("Opponent initialized (Clone of Agent).")

    # 4. Buffers (We use lists because rollout length varies per env due to filtering)
    # Actually, to keep it fast, we will allocate fixed buffers and use a pointer.
    # We estimate we get roughly 1/5th of the turns (since 5 players). 
    # So we allocate buffer for full NUM_STEPS but expect to fill only a fraction.
    
    # Wait, 'NUM_STEPS' usually implies 'Steps per Env'. 
    # If we run 128 steps, each env steps 128 times.
    # Approx 25 of those will be P0.
    # We will simply collect ALL P0 transitions.
    
    start_time = time.time()
    next_obs = env.reset()
    next_done = torch.zeros(NUM_ENVS, device=device)

    for update in range(1, TOTAL_UPDATES + 1):
        # Update Opponent Logic
        if update > 1 and update % OPPONENT_UPDATE_FREQ == 0:
            print(f"[{update}] Upgrading Opponent to current Agent version.")
            opponent.load_state_dict(agent.state_dict())

        # Storage for this rollout
        b_obs, b_actions, b_logprobs, b_rewards, b_dones, b_values, b_masks = [], [], [], [], [], [], []
        
        agent.eval()
        
        # --- ROLLOUT PHASE ---
        for step in range(NUM_STEPS):
            # We need to pick actions for ALL envs.
            # Identify who is playing in which env
            curr_players = env.current_player # [N]
            is_hero = (curr_players == 0)     # [N] Bool
            
            # 1. Get Agent Actions (For Hero Envs)
            # We run the agent on ALL obs for simplicity (batching is faster than masking sometimes),
            # OR we mask. Masking is better for specific stats.
            # Let's run full batch for both to keep tensor shapes consistent, then mask results.
            
            # Hero Forward
            with torch.no_grad():
                mask = env.get_action_mask()
                
                # Agent Preds
                a_logits, a_val = agent(next_obs, mask=mask)
                a_dist = torch.distributions.Categorical(logits=a_logits)
                a_action = a_dist.sample()
                a_logprob = a_dist.log_prob(a_action)
                
                # Opponent Preds
                o_logits, _ = opponent(next_obs, mask=mask)
                o_dist = torch.distributions.Categorical(logits=o_logits)
                o_action = o_dist.sample()
            
            # 2. Combine Actions
            # If is_hero, use a_action. Else use o_action.
            final_action = torch.where(is_hero, a_action, o_action)
            
            # 3. Step Env
            # Convert to One-Hot
            action_oh = torch.zeros((NUM_ENVS, 10), device=device)
            action_oh.scatter_(1, final_action.unsqueeze(1), 100.0)
            
            next_obs_new, rewards, dones_new, _ = env.step(action_oh)
            
            # 4. Store ONLY Hero Data
            # We take the transitions where the actor WAS the hero.
            if is_hero.any():
                b_obs.append(next_obs[is_hero])
                b_actions.append(a_action[is_hero])
                b_logprobs.append(a_logprob[is_hero])
                b_values.append(a_val[is_hero].flatten())
                b_masks.append(mask[is_hero])
                b_rewards.append(rewards[is_hero]) # The reward resulting from the action
                b_dones.append(dones_new[is_hero])
            
            next_obs = next_obs_new
            next_done = dones_new
            
            if next_done.any():
                env.reset_indices(next_done.bool())

        # --- PREPARE DATA ---
        if len(b_obs) == 0:
            print("No hero turns this rollout? Odd.")
            continue
            
        # Cat lists into tensors
        t_obs = torch.cat(b_obs)
        t_actions = torch.cat(b_actions)
        t_logprobs = torch.cat(b_logprobs)
        t_values = torch.cat(b_values)
        t_masks = torch.cat(b_masks)
        t_rewards = torch.cat(b_rewards)
        t_dones = torch.cat(b_dones)
        
        # Calculate Advantages (GAE) on the disjoint buffer
        # Note: Standard GAE relies on t+1. Since our buffer is "Hero Turns Only", 
        # t+1 in this buffer might be 5 steps later in the game.
        # However, for PPO in multi-agent, treating the "Wait Time" as a state transition 
        # where reward is 0 is a common simplification.
        # Alternatively, we just use raw Returns (Reward-to-Go).
        
        # Let's use Reward-to-Go for simplicity in Self-Play sparse buffers
        # Or simple GAE assuming the next line in buffer is the next decision.
        with torch.no_grad():
            advantages = torch.zeros_like(t_rewards)
            lastgaelam = 0
            # Bootstrapping with 0 (assuming end of buffer is approx end of value info)
            next_val = 0 
            
            for t in reversed(range(len(t_rewards))):
                non_terminal = 1.0 - t_dones[t].float()
                # If non-terminal, we use the next value in the buffer as approx
                # This is "skipping" the opponent moves, essentially treating opponent moves as part of the environment dynamics.
                delta = t_rewards[t] + 0.99 * next_val * non_terminal - t_values[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * non_terminal * lastgaelam
                next_val = t_values[t]
                
            returns = advantages + t_values

        # --- PPO UPDATE ---
        agent.train()
        n_samples = t_obs.size(0)
        inds = np.arange(n_samples)
        
        # Adjust Minibatch size if samples < Minibatch
        curr_mb_size = min(MINIBATCH_SIZE, n_samples)
        
        for epoch in range(4):
            np.random.shuffle(inds)
            for start in range(0, n_samples, curr_mb_size):
                end = start + curr_mb_size
                mb = inds[start:end]
                
                _, new_val = agent(t_obs[mb], mask=t_masks[mb])
                new_logits, _ = agent(t_obs[mb], mask=t_masks[mb])
                probs = torch.distributions.Categorical(logits=new_logits)
                new_logprob = probs.log_prob(t_actions[mb])
                entropy = probs.entropy()
                
                logratio = new_logprob - t_logprobs[mb]
                ratio = logratio.exp()
                
                mb_adv = advantages[mb]
                # Normalize
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                pg_loss = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg_loss, pg_loss2).mean()
                
                new_val = new_val.view(-1)
                v_loss = 0.5 * ((new_val - returns[mb]) ** 2).mean()
                
                loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                
        # --- LOG ---
        fps = int(n_samples / (time.time() - start_time))
        print(f"Update {update:04d} | Samples: {n_samples} | FPS: {fps} | Reward: {t_rewards.mean():.4f} | Loss: {loss.item():.4f}")
        start_time = time.time()
        
        # Checkpoint
        if update % 50 == 0:
            torch.save({
                'update': update,
                'model_state_dict': agent.state_dict()
            }, SAVE_FILENAME)

if __name__ == "__main__":
    train_selfplay()