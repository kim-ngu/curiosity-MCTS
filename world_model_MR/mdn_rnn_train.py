import os
import shutil
import time
import random

import numpy as np
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter

# Own classes
from mdn_rnn_args import parse_args
from utils.memory import Memory
from mdn_rnn_net import RNN
from vae import VAE

if __name__ == "__main__":

    # ------------------------------- #
    # ----------- SETUP ------------- #
    # ------------------------------- #

    args = parse_args()
    file_name = os.path.basename(__file__).rstrip(".py")

    if args.rnn_deterministic:
        run_name = "{}_{}_vae-{}_rnn-{}-{}_{}".format(file_name ,args.gym_env[4:], args.img_size, args.rnn_n_layers, args.rnn_h_size, time.strftime("%H%M_%d-%m-%Y"))
    else:
        run_name = "{}_{}_vae-{}_mdn-rnn-{}-{}_{}".format(file_name, args.gym_env[4:], args.img_size, args.rnn_n_layers, args.rnn_h_size, time.strftime("%H%M_%d-%m-%Y"))
    # -----------  Folder setup ----------- #
    if not args.debug:

        # Make checkpoint folder
        checkpoint_folder = os.path.join("checkpoints/", run_name)
        if os.path.exists(checkpoint_folder):
            shutil.rmtree(checkpoint_folder)
        os.mkdir(checkpoint_folder)

        # Make subfolders
        os.mkdir(os.path.join(checkpoint_folder, "vae"))
        os.mkdir(os.path.join(checkpoint_folder, "rnn"))
        os.mkdir(os.path.join(checkpoint_folder, "mcts"))

        # Tensorboard logging
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )   
    
    # ----------- CUDA setup ----------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # ----------- Env setup -----------  #
    env = gym.make(args.gym_env, render_mode = "rgb_array", frameskip = 8)
    env = gym.wrappers.ResizeObservation(env, args.img_size)
    env = gym.wrappers.TransformReward(env, lambda r: r*10e-5)
    action_space = [0,1,2,3,4,5,11,12]

    # ----------- Memory setup -----------  #
    memory = Memory(env, args)

    # ----------- VAE setup ----------- #
    vae = VAE(args)
    vae_load_path = f"models/MR_vae_{args.img_size}.pth"
    vae.load(vae_load_path, mode = "train")

    # ----------- RNN setup ----------- #
    rnn = RNN(args)
  
    if args.load_rnn:
        rnn_load_path = ""
        rnn.load(rnn_load_path, mode = "train")
    
    # ------------------------------------- #
    # ----------  TRAINING LOOP ----------- #
    # ------------------------------------- #

    start_episode = 1
    start_time = time.time()

    for episode in range(start_episode, args.n_episodes + 1):
        memory.clear()
        state, info = env.reset()
        lives = info['lives']
        reward = 0
        death = 0
        terminated = False

        # --------- Gather data ------------ #
        for step in range(0, args.n_steps):

            # Reset if terminated
            if terminated:
                memory.append_state(state)
                memory.append_action(0)
                memory.append_reward(reward)
                memory.append_death(death)

                state, info = env.reset()
                lives = info['lives']
                reward = 0
                death = 0
                terminated = False
            
            else:
                # Append to memory
                memory.append_state(state)
                action = random.choice(action_space)
                memory.append_action(action)
                memory.append_reward(reward)
                memory.append_death(death)

                # Step in environment
                state, reward, terminated, truncated, info = env.step(action)
                
                # Determine death
                if lives > info['lives']:
                    death = 1
                    for _ in range (5): env.step(0) # It takes 5 frames to respawn
                else:
                    death = 0
                lives = info['lives']
        
        # ---------- VAE Training ---------- #

        if args.train_vae_off:
            print("\nVAE Training initiated")

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

        # ---------- RNN Training ---------- #
        print("\nRNN Training initiated")
        
        if args.train_rnn_off:
            # VAE forward pass
            states = memory.get_states()
            z_states = vae.encode(states)

            # Train RNN
            za_sequences, targets = memory.make_rnn_training_data(z_states)
            rnn_avg_episode_loss = rnn.train(za_sequences, targets)

            # Logging
            exe_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            print(f"RNN Training | Episode: {episode} / {args.n_episodes} | Avg. episode loss: {rnn_avg_episode_loss} | Execution Time: {exe_time}")
            writer.add_scalar("charts/rnn_loss", rnn_avg_episode_loss, episode)

            # Saving
            if (episode % args.save_interval == 0):
                rnn.save(run_name, episode)
        
    env.close()
    writer.close()





        

            


        

        





