import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os

from environment import GolfEnv
from model import GolfModel

# --- LOCAL SPEED SETTINGS ---
TOTAL_UPDATES = 200      # Run a bit longer to get a smart AI
NUM_ENVS = 4096          # MAXIMUM PARALLELISM for 2060 Super (Try 2048 if crash)
NUM_STEPS = 128           # Shorter rollouts to save VRAM with high env count
BATCH_SIZE = NUM_ENVS * NUM_STEPS
MINIBATCH_SIZE = 32768    
LEARNING_RATE = 2.5e-4
SAVE_FILENAME = "latest_model.pt"

def train_local():
    # Speed Optimization: Enable CuDNN Benchmark
    torch.backends.cudnn.benchmark = True
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- STARTING LOCAL TRAIN ON {device} ---")
    print(f"Envs: {NUM_ENVS} | Steps: {NUM_STEPS} | Batch: {BATCH_SIZE}")
    
    # 1. Init
    env = GolfEnv(num_envs=NUM_ENVS, device=device)
    agent = GolfModel().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # 2. Buffers
    obs = torch.zeros((NUM_STEPS, NUM_ENVS, 737), device=device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    masks = torch.zeros((NUM_STEPS, NUM_ENVS, 10), device=device)

    # Start Game
    next_obs = env.reset()
    next_done = torch.zeros(NUM_ENVS, device=device)
    
    start_time = time.time()

    for update in range(1, TOTAL_UPDATES + 1):
        # A. Collect Data
        agent.eval()
        for step in range(NUM_STEPS):
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action_mask = env.get_action_mask()
                masks[step] = action_mask
                
                logits, value = agent(next_obs, mask=action_mask)
                values[step] = value.flatten()
                
                probs = torch.distributions.Categorical(logits=logits)
                action = probs.sample()
                logprobs[step] = probs.log_prob(action)
                actions[step] = action

            action_one_hot = torch.zeros((NUM_ENVS, 10), device=device)
            action_one_hot.scatter_(1, action.unsqueeze(1), 100.0) 
            
            next_obs, reward, next_done, info = env.step(action_one_hot)
            rewards[step] = reward
            
            if next_done.any():
                env.reset_indices(next_done.bool())

        # B. Calculate Advantages
        with torch.no_grad():
            next_mask = env.get_action_mask()
            _, next_value = agent(next_obs, mask=next_mask)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value.flatten()
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + 0.99 * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + values

        # C. Train
        agent.train()
        b_obs = obs.reshape((-1, 737))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_masks = masks.reshape((-1, 10))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(BATCH_SIZE)
        
        for epoch in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newvalue = agent(b_obs[mb_inds], mask=b_masks[mb_inds])
                newlogits, _ = agent(b_obs[mb_inds], mask=b_masks[mb_inds])
                probs = torch.distributions.Categorical(logits=newlogits)
                newlogprob = probs.log_prob(b_actions[mb_inds])
                entropy = probs.entropy()

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg_loss, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        # D. Log
        fps = int(BATCH_SIZE / (time.time() - start_time))
        print(f"Update {update:03d} | FPS: {fps} | Reward: {rewards.mean():.4f} | Loss: {loss.item():.4f}")
        start_time = time.time()

    # --- SAVE MODEL ---
    print(f"--- Saving to {SAVE_FILENAME} ---")
    checkpoint = {
        'update': TOTAL_UPDATES,
        'model_state_dict': agent.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, SAVE_FILENAME)
    print("Done!")

if __name__ == "__main__":
    train_local()