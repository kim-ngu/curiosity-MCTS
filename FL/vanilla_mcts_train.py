import os
import shutil
from copy import deepcopy
import random
import time
import csv

from math import *
import numpy as np
import torch


from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

from vanilla_mcts_args import parse_args
from utils.write_to_csv import write_to_csv
from utils.get_pixel_state import get_pixel_state
from utils.vanilla_memory import Memory
from vanilla_mcts import Node, MCTS


def exe_time(start_time):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

if __name__ == "__main__":
    args = parse_args()
    run_name = "{}_mcts_{}".format(args.gym_env[:-3], time.strftime("%H%M_%d-%m-%Y"))
    csv_file = '{}_mcts_terminal_output_{}.csv'.format(args.gym_env[:-3],time.strftime("%H%M_%d-%m-%Y"))
    # Create folders
    if not args.debug:
        # Create checkpoint folder
        checkpoint_folder = os.path.join("checkpoints", run_name)
        if os.path.exists(checkpoint_folder):
            shutil.rmtree(checkpoint_folder)
        os.mkdir(checkpoint_folder)

        # Make subfolders
        os.mkdir(os.path.join(checkpoint_folder, "mcts"))
        os.mkdir(os.path.join(checkpoint_folder, "vae"))

        # Tensorboard logging
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
            )
        
        # CSV file
        with open(csv_file, 'w', encoding='UTF8', newline='') as file:
            csv_writer = csv.writer(file,delimiter=';', dialect = 'excel')
            csv_writer.writerow(['Step', 'Action', 'Reward_ext', 'Reward_int'])
    
    # CUDA setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # Env Setup
    env = gym.make("FrozenLake-v1",  render_mode = "rgb_array", map_name="8x8", is_slippery=False)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=32)
    memory = Memory(env, args)

    # Network setup
    mcts = MCTS(args)
    mcts.vae.load("models/FL_vae_8x8.pth")
    #mcts.temperature = 0
    moves = ['left ', 'down ', 'right', 'up   ']

    start_time = time.time()
    total_steps = 1
    for update in range(1, args.rnn_n_updates + 1):
        
        # Init
        reward_ext = 0

        env.reset()
        sim_env = deepcopy(env)
        state = get_pixel_state(env)
        root = Node(sim_env, state)
        memory.clear()

        done = False

        # Training Data
        total_states = []
        total_targets = []
        # Play using current rnn network n times
        for i in range(args.rnn_n_steps):

            root, action = mcts.search(root)
            #action = env.action_space.sample() # Pseudo action for debugging

            _, reward_ext, terminated, truncated, _ = env.step(action)
            next_state = get_pixel_state(env)
            done = np.any([terminated, truncated], axis=0)

            # Append current step to memory
            memory.append(state, action, reward_ext)
            
            # Print
            print(f" Step: {total_steps} | Action: {moves[action]} | Reward_ext: {reward_ext}")

            # Write to CSV
            write_to_csv(csv_file, [total_steps, moves[action], reward_ext, 0])

            total_steps += 1

            if done:
                print("\n -------- Environment reset -----------\n")
                write_to_csv(csv_file, ['reset', 'reset', 'reset', 'reset'])

                # Make episodic training data
                states, targets = memory.make_mcts_training_data()
                total_states += states
                total_targets += targets

                # Reset
                reward_ext = 0

                env.reset()
                state = get_pixel_state(env)
                sim_env = deepcopy(env)
                root = Node(sim_env, state)

                memory.clear()
                done = False

            else:
                state = next_state

        # Add final steps
        if len(memory.actions_one_hot) > 1:
            states, targets = memory.make_mcts_training_data()
            total_states += states
            total_targets += targets

        # MCTS net training
        total_states = torch.stack(total_states).to(device)
        total_targets = torch.stack(total_targets).to(device)

        policy_loss, value_loss = mcts.train(total_states, total_targets, args)
        print(f"\nMCTS net training | Update: {update} / {args.rnn_n_updates} | Policy Loss: {policy_loss:.4f} | Value Net Loss: {value_loss:.4f} | Execution time: {exe_time(start_time)}\n")
        write_to_csv(csv_file, ['update', 'update', 'update', 'update'])
        writer.add_scalar("charts/policy_loss", policy_loss, update)
        writer.add_scalar("charts/value_loss", value_loss, update)

        """
        # Vae training
        vae_loss = mcts.vae.train(total_states)
        print(f"VAE training      | Update: {update} / {args.rnn_n_updates} | Loss: {vae_loss:.4f} | Execution time: {exe_time(start_time)}")
        writer.add_scalar("charts/vae_loss", vae_loss, update)
        """

        # Save mcts & vae
        if (update % args.save_interval == 0):
            mcts.save(run_name, update)
           # mcts.vae.save(run_name, update)

        # Action selection becomes more greedy as training progresses
        if update == int(args.rnn_n_updates * 0.5):
            mcts.explore_const = 0.5
        
        if update == int(args.rnn_n_updates * 0.75):
            mcts.explore_const = 0.25        
    
    env.close()

    




    





    



