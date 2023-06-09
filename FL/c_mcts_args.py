import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()

    # ---------- General settings --------- #
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--gym-env", type=str, default="FrozenLake-v1")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--save-interval", type=int, default=1)

    # ---------- VAE settings ---------- #

    # Training
    parser.add_argument("--vae-n-episodes", type=int, default=500)
    parser.add_argument("--vae-n-steps", type=int, default=512)
    parser.add_argument("--vae-n-epochs", type=int, default=20)
    parser.add_argument("--vae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--vae-batch-size", type=int, default=32)

    # ---------- RND settings ----------- #

    # Training
    parser.add_argument("--rnd-learning-rate", type=float, default=1e-4)
    parser.add_argument("--rnd-batch-size", type=int, default=1)
    parser.add_argument("--rnd-n-epochs", type=int, default=8)

    # Parameters
    parser.add_argument("--rnd-gamma-ext", type=float, default=0.999)
    parser.add_argument("--rnd-gamma-int", type=float, default=0.99)

    parser.add_argument("--rnd-coef-ext", type=float, default=4.0)
    parser.add_argument("--rnd-coef-int", type=float, default=1.0)

    # ---------- MCTS settings --------- #

    # rnn Network Training
    parser.add_argument("--rnn-clip-value", type=float, default=0.5)
    parser.add_argument("--rnn-n-updates", type=int, default=8)
    parser.add_argument("--rnn-n-steps", type=int, default=128)
    parser.add_argument("--rnn-n-epochs", type=int, default=10)
    parser.add_argument("--rnn-learning-rate", type=float, default=1e-5)
    parser.add_argument("--rnn-batch-size", type=int, default=16)
    #parser.add_argument("--rnn-n-replays", type=int, default=10)

    # rnn network architecture
    parser.add_argument("--rnn-input-size", type=int, default=32) # z-state + actions
    parser.add_argument("--rnn-h-size", type=int, default=256)
    parser.add_argument("--rnn-n-layers", type=int, default=1)
    parser.add_argument("--rnn-sequence-length", type=int, default=8)
    
    # MLP architecture
    parser.add_argument("--mlp-h-size", type=int, default=512)
    parser.add_argument("--mlp-n-layers", type=int, default=4)

    # Output settings
    parser.add_argument("--rnn-n-actions", type=int, default=4)

    # MCTS hyperparameters
    parser.add_argument("--mcts-n-simulations", type=int, default=600)
    parser.add_argument("--mcts-temperature", type=float, default=1.0)
    parser.add_argument("--mcts-dirichlet-eps", type=float, default=0.25)
    parser.add_argument("--mcts-dirichlet-alpha", type=float, default=0.03)

    args = parser.parse_args()
    return args