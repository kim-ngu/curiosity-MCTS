import os
import shutil
import time
import argparse

import gymnasium as gym
import torch

from torch.utils.tensorboard import SummaryWriter

from utils.memory import Memory
from args_4 import parse_args
from vae import VAE
from utils.get_pixel_state import get_pixel_state

if __name__ == "__main__":

    # ----------- SETUP ------------- #
    args = parse_args()
    run_name = "{}_vae-{}_{}".format(args.gym_env[:-3], args.img_size, time.strftime("%H%M_%d-%m-%Y"))

    # Folder setup
    if not args.debug:

        # Make checkpoint folder
        checkpoint_folder = os.path.join("checkpoints/", run_name)
        if os.path.exists(checkpoint_folder):
            shutil.rmtree(checkpoint_folder)
        os.mkdir(checkpoint_folder)

        # Tensorboard logging
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )   

    # CUDA setup 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    env = gym.make("FrozenLake-v1",  render_mode = "rgb_array", map_name="8x8", is_slippery=False)

    # Memory setup
    memory = Memory(env, args)

    # VAE setup
    vae = VAE(args)

    # ---------- Training Loop ---------- #

    start_episode = 1
    start_time = time.time()
    
    for episode in range(start_episode, args.vae_n_episodes + 1):
        memory.clear()
        env.reset()
        state = get_pixel_state(env)
        terminated = False

        # Gather data
        for step in range (0, args.vae_n_steps):

            if terminated:
                memory.append_state(state)
                env.reset()
                state = get_pixel_state(env)
                terminated = False
            
            else:
                # Append to memory
                memory.append_state(state)
                action = env.action_space.sample()

                # Step in environment
                _, _, terminated, _, _ = env.step(action)
                state = get_pixel_state(env)

        # Train VAE
        states = memory.get_states()
        vae_avg_episode_loss = vae.train(states)

        # Logging
        exe_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"VAE Training | Episode: {episode} / {args.vae_n_episodes} | Avg. episode loss: {vae_avg_episode_loss} | Execution Time: {exe_time}")
        writer.add_scalar("charts/vae_loss", vae_avg_episode_loss, episode)
    
        # Saving
        if (episode % args.save_interval == 0):
            vae.save(run_name, episode)


