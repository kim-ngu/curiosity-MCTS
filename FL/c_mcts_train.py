import os
import shutil
from copy import deepcopy
import random
import time

from math import *
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

from c_mcts_args import parse_args
from utils.get_pixel_state import get_pixel_state
from utils.memory import Memory
from c_mcts import Node, MCTS
from rnd import RND


def exe_time(start_time):
    return time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))

if __name__ == "__main__":
    args = parse_args()
    run_name = "{}_mcts_{}".format(args.gym_env[:-3], time.strftime("%H%M_%d-%m-%Y"))
    # Create folders
    if not args.debug:
        # Create checkpoint folder
        checkpoint_folder = os.path.join("checkpoints", run_name)
        if os.path.exists(checkpoint_folder):
            shutil.rmtree(checkpoint_folder)
        os.mkdir(checkpoint_folder)

        # Make subfolders
        os.mkdir(os.path.join(checkpoint_folder, "mcts"))
        os.mkdir(os.path.join(checkpoint_folder, "rnd"))
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
        reward_int = 0

        # Init RND
        mcts.rnd = RND(args)

        env.reset()
        sim_env = deepcopy(env)
        state = get_pixel_state(env)
        root = Node(sim_env, state)

        memory.clear()
        for i in range (args.rnn_sequence_length):
            memory.states.append(state)
            memory.append_action(0)

        done = False

        # Training Data
        total_states = []
        total_state_sequences = []
        total_targets = []
        # Play using current rnn network n times
        for i in range(args.rnn_n_steps):

            # Train RND on current state
            z_state = mcts.vae.encode(state.unsqueeze(0).to(device))
            mcts.rnd.train(z_state)

            # Get action from mcts and step in environment
            history = memory.get_history()
            root, action = mcts.search(root, history)
            #action = env.action_space.sample() # Pseudo action for debugging

            _, reward_ext, terminated, truncated, _ = env.step(action)
            next_state = get_pixel_state(env)
            done = np.any([terminated, truncated], axis=0)

            with torch.no_grad():
                next_z_state = mcts.vae.encode(next_state.unsqueeze(0).to(device))
                reward_int = mcts.rnd.reward(next_z_state)

            # Append current step to memory
            memory.append(state, action, reward_ext, reward_int)
            # Print
            print(f" Step: {total_steps} | Action: {moves[action]} | Reward_ext: {reward_ext} | Reward_int: {reward_int}")
            total_steps += 1

            if done:
                print("\n -------- Environment reset -----------\n")
                # Make episodic training data
                states_t, state_sequences, targets = memory.make_mcts_training_data()
                total_states += states_t
                total_state_sequences += state_sequences
                total_targets += targets

                # Reset
                
                # RND
                mcts.rnd = RND(args)
            
                reward_ext = 0
                reward_int = 0

                env.reset()
                state = get_pixel_state(env)
                sim_env = deepcopy(env)
                root = Node(sim_env, state)

                memory.clear()
                for i in range (args.rnn_sequence_length):
                    memory.states.append(state)
                    memory.append_action(0)

                done = False
            else:
                state = next_state

        # Add final steps
        if len(memory.actions_one_hot) > 1:
            states_t, state_sequences, targets = memory.make_mcts_training_data()
            total_states += states_t
            total_state_sequences += state_sequences
            total_targets += targets

        # MCTS net training
        total_states = torch.stack(total_states).to(device)
        total_state_sequences = torch.stack(total_state_sequences).to(device)
        total_targets = torch.stack(total_targets).to(device)

        policy_loss, value_loss = mcts.train(total_states, total_state_sequences, total_targets, args)
        mcts.rnd.save(run_name, update)
        print(f"\nMCTS net training | Update: {update} / {args.rnn_n_updates} | Policy Loss: {policy_loss:.4f} | Value Net Loss: {value_loss:.4f} | Execution time: {exe_time(start_time)}\n")
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

        """
        # RND training
        rnd_loss = mcts.rnd.train(total_states)
        print(f"RND training      | Update: {update} / {args.rnn_n_updates} | Loss: {rnd_loss:.4f} | Execution time: {exe_time(start_time)}")
        writer.add_scalar("charts/rnd_loss", rnd_loss, update)
        
        # Save rnd
        if (update % args.save_interval == 0):
            mcts.rnd.save(run_name, update)
        """
        # Action selection becomes more greedy as training progresses
        if update == int(args.rnn_n_updates * 0.5):
            mcts.explore_const = 0.5
        
        if update == int(args.rnn_n_updates * 0.75):
            mcts.explore_const = 0.25        
    
    env.close()

    




    





    



