import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from environment import GolfEnv
from model import GolfModel

# --- Quick Test Hyperparameters ---
TOTAL_UPDATES = 50       # Run for just 50 loops
NUM_ENVS = 1024          # Full parallelism
NUM_STEPS = 64           # Shorter rollouts for quick feedback
BATCH_SIZE = NUM_ENVS * NUM_STEPS
MINIBATCH_SIZE = 2048    # Large chunks for GPU speed
LEARNING_RATE = 2.5e-4

def train_local():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- STARTING LOCAL GPU TEST ON {device} ---")
    print(f"Batch Size: {BATCH_SIZE} | Total Interactions: {TOTAL_UPDATES * BATCH_SIZE}")

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

            # Convert action to one-hot for env
            action_one_hot = torch.zeros((NUM_ENVS, 10), device=device)
            action_one_hot.scatter_(1, action.unsqueeze(1), 100.0) 
            
            next_obs, reward, next_done, info = env.step(action_one_hot)
            rewards[step] = reward
            
            if next_done.any():
                env.reset_indices(next_done.bool())

        # B. Calculate Advantages (GAE)
        with torch.no_grad():
            next_mask = env.get_action_mask()
            _, next_value = agent(next_obs, mask=next_mask)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    # FIX: Explicitly cast bool tensor to float before subtraction
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
        clip_fracs = []
        
        # 4 Epochs per update
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

        # D. Log to Console
        fps = int(BATCH_SIZE / (time.time() - start_time))
        print(f"Update {update:02d} | FPS: {fps} | Reward: {rewards.mean():.4f} | Loss: {loss.item():.4f}")
        start_time = time.time()

if __name__ == "__main__":
    train_local()