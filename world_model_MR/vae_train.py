import os
import shutil
import time
import argparse

import gymnasium as gym
import torch

from torch.utils.tensorboard import SummaryWriter

from utils.memory import Memory
from vae_args import parse_args
from vae import VAE

if __name__ == "__main__":

    # ----------- SETUP ------------- #
    args = parse_args()
    run_name = "{}_vae-{}_{}".format(args.gym_env[4:], args.img_size, time.strftime("%H%M_%d-%m-%Y"))

    # Folder setup
    if not args.debug:

        # Make checkpoint folder
        checkpoint_folder = os.path.join("checkpoints/", run_name)
        if os.path.exists(checkpoint_folder):
            shutil.rmtree(checkpoint_folder)
        os.mkdir(checkpoint_folder)

        # Make subfolders
        os.mkdir(os.path.join(checkpoint_folder, "vae"))

        # Tensorboard logging
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )   

    # CUDA setup 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # Env Setup
    env = gym.make(args.gym_env, render_mode = "rgb_array", frameskip = 8)
    env = gym.wrappers.ResizeObservation(env, args.img_size)

    # Memory setup
    memory = Memory(env, args)

    # VAE setup
    vae = VAE(args)

    # ---------- Training Loop ---------- #

    start_episode = 1
    start_time = time.time()
    
    for episode in range(start_episode, args.n_episodes + 1):
        memory.clear()
        state, info = env.reset()
        lives = info['lives']
        terminated = False

        # Gather data
        for step in range (0, args.n_steps):

            if terminated:
                memory.append_state(state)

                state, info = env.reset()
                lives = info['lives']
                terminated = False
            
            memory.append_state(state)
            action = env.action_space.sample()

            # Step in environment
            state, _, terminated, _, info = env.step(action)

            # frameskip upon death
            if lives > info['lives']:
                for _ in range (5): env.step(0) # It takes 5 frames to respawn

            lives = info['lives']

        # Train VAE
        states = memory.get_states()
        vae_avg_episode_loss = vae.train(states)

        # Logging
        exe_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"VAE Training | Episode: {episode} / {args.n_episodes} | Avg. episode loss: {vae_avg_episode_loss} | Execution Time: {exe_time}")
        writer.add_scalar("charts/vae_loss", vae_avg_episode_loss, episode)
    
        # Saving
        if (episode % args.save_interval == 0):
            vae.save(run_name, episode)


