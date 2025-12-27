import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import copy
import os

from environment import GolfEnv
from model import GolfModel

# --- SELF-PLAY "PRECISION" SETTINGS ---
TOTAL_UPDATES = 5000      
NUM_ENVS = 4096           
NUM_STEPS = 128           
BATCH_SIZE = NUM_ENVS * NUM_STEPS 
MINIBATCH_SIZE = 16384    

# --- CRITICAL HYPERPARAMETERS ---
LEARNING_RATE = 2.5e-5    
OPPONENT_UPDATE_FREQ = 50 
ENTROPY_COEF = 0.005
CLIP_RANGE = 0.1

SAVE_FILENAME = "latest_model.pt"

def train_selfplay():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- STARTING SELF-PLAY (PRECISION MODE) ON {device} ---")
    
    # 1. Init Envs
    env = GolfEnv(num_envs=NUM_ENVS, device=device)
    
    # 2. Init Agent (Learner - Seat 0)
    agent = GolfModel().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    
    # Load Bootstrap Weights
    if os.path.exists(SAVE_FILENAME):
        print(f"Loading {SAVE_FILENAME}...")
        try:
            ckpt = torch.load(SAVE_FILENAME, map_location=device)
            # Handle both dictionary and direct state_dict
            if 'model_state_dict' in ckpt:
                agent.load_state_dict(ckpt['model_state_dict'])
            else:
                agent.load_state_dict(ckpt)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}. Starting from scratch (NOT RECOMMENDED).")
    else:
        print("WARNING: No model found! You should run train_local.py first!")

    # 3. Init Opponent (Seats 1-4)
    opponent = copy.deepcopy(agent)
    opponent.eval() 
    print("Opponent initialized.")

    start_time = time.time()
    next_obs = env.reset()
    next_done = torch.zeros(NUM_ENVS, device=device)

    try:
        for update in range(1, TOTAL_UPDATES + 1):
            # Update Opponent Logic
            if update > 1 and update % OPPONENT_UPDATE_FREQ == 0:
                print(f"[{update}] UPGRADE: Opponent updated to latest version.")
                opponent.load_state_dict(agent.state_dict())

            # Storage
            b_obs, b_actions, b_logprobs, b_rewards, b_dones, b_values, b_masks = [], [], [], [], [], [], []
            
            agent.eval()
            
            # --- ROLLOUT PHASE ---
            for step in range(NUM_STEPS):
                curr_players = env.current_player
                is_hero = (curr_players == 0)    
                
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
                
                # Combine Actions
                final_action = torch.where(is_hero, a_action, o_action)
                
                # Step Env
                action_oh = torch.zeros((NUM_ENVS, 10), device=device)
                action_oh.scatter_(1, final_action.unsqueeze(1), 100.0)
                
                next_obs_new, rewards, dones_new, _ = env.step(action_oh)
                
                # Store ONLY Hero Data
                if is_hero.any():
                    b_obs.append(next_obs[is_hero])
                    b_actions.append(a_action[is_hero])
                    b_logprobs.append(a_logprob[is_hero])
                    b_values.append(a_val[is_hero].flatten())
                    b_masks.append(mask[is_hero])
                    b_rewards.append(rewards[is_hero]) 
                    b_dones.append(dones_new[is_hero])
                
                next_obs = next_obs_new
                next_done = dones_new
                
                if next_done.any():
                    env.reset_indices(next_done.bool())

            # --- PREPARE DATA ---
            if len(b_obs) == 0: continue
                
            t_obs = torch.cat(b_obs)
            t_actions = torch.cat(b_actions)
            t_logprobs = torch.cat(b_logprobs)
            t_values = torch.cat(b_values)
            t_masks = torch.cat(b_masks)
            t_rewards = torch.cat(b_rewards)
            t_dones = torch.cat(b_dones)
            
            # GAE / Returns
            with torch.no_grad():
                advantages = torch.zeros_like(t_rewards)
                lastgaelam = 0
                next_val = 0 
                
                for t in reversed(range(len(t_rewards))):
                    non_terminal = 1.0 - t_dones[t].float()
                    delta = t_rewards[t] + 0.99 * next_val * non_terminal - t_values[t]
                    advantages[t] = lastgaelam = delta + 0.99 * 0.95 * non_terminal * lastgaelam
                    next_val = t_values[t]
                    
                returns = advantages + t_values

            # --- PPO UPDATE ---
            agent.train()
            n_samples = t_obs.size(0)
            inds = np.arange(n_samples)
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
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    
                    # TIGHTER CLIPPING (0.1 instead of 0.2)
                    pg_loss = -mb_adv * ratio
                    pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE)
                    pg_loss = torch.max(pg_loss, pg_loss2).mean()
                    
                    new_val = new_val.view(-1)
                    v_loss = 0.5 * ((new_val - returns[mb]) ** 2).mean()
                    
                    loss = pg_loss - ENTROPY_COEF * entropy.mean() + 0.5 * v_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()
                    
            # --- LOG ---
            fps = int(n_samples / (time.time() - start_time))
            # Format nicely
            print(f"Up {update:04d} | Rew: {t_rewards.mean():.4f} | Loss: {loss.item():.2f} | FPS: {fps}")
            start_time = time.time()
            
            # Checkpoint
            if update % 50 == 0:
                torch.save({'model_state_dict': agent.state_dict()}, SAVE_FILENAME)

    except KeyboardInterrupt:
        print("Saving...")
        torch.save({'model_state_dict': agent.state_dict()}, SAVE_FILENAME)

if __name__ == "__main__":
    train_selfplay()