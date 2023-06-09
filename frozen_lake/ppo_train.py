import os
import shutil
import time
import argparse
import random
from distutils.util import strtobool

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.tensorboard import SummaryWriter
from ppo_args import parse_args
from ppo_agent import AtariAgent, ReplayBuffer, Framestack
    
if __name__ == "__main__":
    args = parse_args()
    run_name = "{}_ppo_{}".format(args.gym_env[:-3], time.strftime("%H%M_%d-%m-%Y"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.debug:

        # Make folders
        checkpoint_folder = os.path.join("checkpoints", run_name)
        if os.path.exists(checkpoint_folder):
            shutil.rmtree(checkpoint_folder)
        os.mkdir(checkpoint_folder)

        # Tensorboard logging
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    # Env setup
    env = gym.make("FrozenLake-v1",  render_mode = "rgb_array", map_name="8x8", is_slippery=False)
    obs_shape = (4,84,84)

    # Setup Agent
    agent = AtariAgent(env).to(device)
    replay_buffer = ReplayBuffer(device, args.num_steps, obs_shape)
    optimizer = optim.Adam(agent.parameters(), lr =args.learning_rate, eps=1e-5) 

    # Start Game
    global_step = 0
    start_time = time.time()
    env.reset()
    framestack = Framestack(env)
    next_state = framestack.get(env).to(device)
    next_done = torch.zeros(1).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # Training Loop
    for update in range(1, num_updates + 1):
        #Learning rate Annealing
        frac = 1.0 - (update  - 1.0) / num_updates
        lr_now = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lr_now

        
        episodic_reward = 0 
        # Policy rollout loop
        for step in range(0, args.num_steps):
            
            global_step += 1
            replay_buffer.states[step] = next_state
            replay_buffer.dones[step] = next_done
        
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_state)
                replay_buffer.values[step] = value.flatten()
            replay_buffer.actions[step] = action
            replay_buffer.logprobs[step] = logprob

            # Take a step in the environment and store data in replay buffer
            _, reward, terminated, truncated, infos = env.step(action.cpu().item())
            next_state = framestack.get(env).to(device)
            done = int(np.any([terminated, truncated], axis=0))
            replay_buffer.rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = torch.Tensor([done]).to(device)
            
            # Log episodic return in Tensorboard
            episodic_reward += reward
        
        exe_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print("Update: %d / %d | Global step: %d | Episodic return: %1.5f" % (update, num_updates, global_step, episodic_reward), "| Execution time: %s " % (exe_time))
        writer.add_scalar("charts/episodic_return", episodic_reward, global_step)
        
        # GAE Estimation
        with torch.no_grad():
            next_value = agent.get_value(next_state).reshape(1, -1).to(device)

            advantages = torch.zeros_like(replay_buffer.rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - replay_buffer.dones[t + 1]
                    nextvalues = replay_buffer.values[t + 1]
                
                delta = replay_buffer.rewards[t] + args.gamma * nextvalues * nextnonterminal - replay_buffer.values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + replay_buffer.values  

        # flatten the batch
        b_states = replay_buffer.states.reshape((-1,) + obs_shape)
        b_logprobs = replay_buffer.logprobs.reshape(-1)
        b_actions = replay_buffer.actions.reshape((-1,) + env.action_space.shape)
        b_values = replay_buffer.values.reshape(-1)          
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Optimizing the policy and value network
        clipfracs = []
        b_inds = np.arange(args.batch_size) #Acquire all indices of the batch
       
        for epoch in range(args.update_epochs): 
            np.random.shuffle(b_inds) #shuffle indices

            # loop through entire batch, one mini-batch at a time
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Forward pass a mini-batch of states
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_states[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Debug variables
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean() # Approximate KL divergence
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]

                # Advantage Normalization
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                #torch.clamp(input, min, max)
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                # Entropy Loss
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Backpropagate and optimize with global gradient clipping
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Save Checkpoint
        if (update % 1000 == 0):
            checkpoint_file = os.path.join("checkpoints/{}".format(run_name), str(update) + ".pth")
            torch.save(agent.state_dict(), checkpoint_file)
    env.close()
    writer.close()

    # Save final model
    save_file = os.path.join("models", run_name + ".pth")
    torch.save(agent.state_dict(), save_file)       













    

       














