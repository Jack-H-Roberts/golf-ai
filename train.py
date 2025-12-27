import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import time

from environment import GolfEnv
from model import GolfModel

# --- Hyperparameters ---
TOTAL_TIMESTEPS = 10_000_000
LEARNING_RATE = 2.5e-4
NUM_ENVS = 1024
NUM_STEPS = 128  # Steps per update loop
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
BATCH_SIZE = NUM_ENVS * NUM_STEPS
MINIBATCH_SIZE = 4096 # Split batch into smaller chunks for GPU memory
NUM_EPOCHS = 4

def train():
    run_name = f"golf_ppo_{int(time.time())}"
    wandb.init(project="golf-ai", name=run_name, monitor_gym=False, save_code=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 1. Initialize Env & Agent
    env = GolfEnv(num_envs=NUM_ENVS, device=device)
    agent = GolfModel().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # 2. storage buffers
    obs = torch.zeros((NUM_STEPS, NUM_ENVS, 737), device=device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
    # Store masks so we know what was legal at that time
    masks = torch.zeros((NUM_STEPS, NUM_ENVS, 10), device=device)

    # Global step counter
    global_step = 0
    start_time = time.time()
    
    # Start the game
    next_obs = env.reset()
    next_done = torch.zeros(NUM_ENVS, device=device)
    
    num_updates = TOTAL_TIMESTEPS // BATCH_SIZE

    for update in range(1, num_updates + 1):
        # --- A. Rollout Phase (Collect Data) ---
        agent.eval()
        for step in range(0, NUM_STEPS):
            global_step += 1 * NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                # Get mask for current state
                action_mask = env.get_action_mask()
                masks[step] = action_mask
                
                # Get action from agent
                logits, value = agent(next_obs, mask=action_mask)
                values[step] = value.flatten()
                
                # Sample action (Categorical distribution)
                # We apply mask to logits inside model, so probs are valid
                probs = torch.distributions.Categorical(logits=logits)
                action = probs.sample()
                logprob = probs.log_prob(action)

            actions[step] = action
            logprobs[step] = logprob

            # Execute Step
            # Note: We need to convert action to one-hot or just indices?
            # Our step() expects indices [1024, 10] or [1024]?
            # Our environment.step() expects [1024, 10] logits usually?
            # Wait, our current step() code expects "actions" to be logits OR indices?
            # Let's check environment.py: 
            # "actions is a tensor of shape [Batch, 10] containing move preferences."
            # ADJUSTMENT: We need to pass One-Hot or Logits to env.step()
            # Since PPO selects a specific integer action, we should construct a "fake" logit vector
            # where the chosen action is 100.0 and others are 0.0, 
            # OR update step() to accept integer actions.
            # FAST FIX: Pass one-hot.
            
            action_one_hot = torch.zeros((NUM_ENVS, 10), device=device)
            action_one_hot.scatter_(1, action.unsqueeze(1), 100.0) 
            
            next_obs, reward, next_done, info = env.step(action_one_hot)
            rewards[step] = reward
            
            # Auto-Reset Logic
            if next_done.any():
                # For PPO, we technically should distinguish "terminal observation".
                # But for this simple version, we just reset in place.
                env.reset_indices(next_done.bool())

        # --- B. Bootstrap Value (GAE) ---
        with torch.no_grad():
            next_mask = env.get_action_mask() # for the state after the last step
            _, next_value = agent(next_obs, mask=next_mask)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value.flatten()
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
            
            returns = advantages + values

        # --- C. Optimization Phase (Train) ---
        agent.train()
        # Flatten the batch
        b_obs = obs.reshape((-1, 737))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_masks = masks.reshape((-1, 10))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimization Epochs
        b_inds = np.arange(BATCH_SIZE)
        for epoch in range(NUM_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newvalue = agent(b_obs[mb_inds], mask=b_masks[mb_inds])
                
                # Get new logprobs (re-run network)
                newlogits, _ = agent(b_obs[mb_inds], mask=b_masks[mb_inds])
                probs = torch.distributions.Categorical(logits=newlogits)
                newlogprob = probs.log_prob(b_actions[mb_inds])
                entropy = probs.entropy()

                # Ratio
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Surrogate Loss
                mb_advantages = b_advantages[mb_inds]
                # Normalize advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Total Loss
                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # --- D. Logging ---
        avg_reward = rewards.sum() / NUM_ENVS # Approx reward per game block
        print(f"Update {update}/{num_updates} | Step {global_step} | Reward: {avg_reward:.3f} | Loss: {loss.item():.3f}")
        
        wandb.log({
            "global_step": global_step,
            "charts/avg_reward": avg_reward,
            "losses/policy_loss": pg_loss.item(),
            "losses/value_loss": v_loss.item(),
            "losses/entropy": entropy_loss.item()
        })
    
    print("Training Complete. Saving Model...")
    torch.save(agent.state_dict(), "golf_ai_final.pth")
    wandb.finish()

if __name__ == "__main__":
    train()