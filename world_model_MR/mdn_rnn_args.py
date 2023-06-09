import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()

    # ---------- General settings --------- #
    
    parser.add_argument("--file-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--gym-env", type=str, default="ALE/MontezumaRevenge-v5")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug", action='store_true')

    parser.add_argument("--n-episodes", type=int, default=300)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--save_interval", type=int, default=20)

    # ---------- Data Settings  ---------- #
    parser.add_argument("--death-augment", action="store_true")
    parser.add_argument("--death-reps", type=int, default=8)

    # ---------- Vae settings ---------- #

    # Training
    parser.add_argument("--train-vae-off", action="store_false")
    parser.add_argument("--vae-n-epochs", type=int, default=20)
    parser.add_argument("--vae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--vae-batch-size", type=int, default=32)

    # Network setup
    parser.add_argument("--img-size", type=int, default=96)

    # ---------- RNN settings ----------- #

    # Training
    parser.add_argument("--rnn-deterministic", action="store_true")
    parser.add_argument("--train-rnn-off", action="store_false")
    parser.add_argument("--rnn-n-epochs", type=int, default=20)
    parser.add_argument("--rnn-clip-value", type=float, default=0.5)
    parser.add_argument("--rnn-learning-rate", type=float, default=1e-3)

    # Network setup
    parser.add_argument("--load-rnn", action='store_true')
    parser.add_argument("--rnn-input-size", type=int, default=33)
    parser.add_argument("--rnn-sequence_length", type=int, default=32)
    parser.add_argument("--rnn-batch-size", type=int, default=32)
    parser.add_argument("--rnn-h-size", type=int, default=512)
    parser.add_argument("--rnn-n-layers", type=int, default=1)
    parser.add_argument("--rnn-output-size", type=int, default=33) # z-state + reward
    parser.add_argument("--rnn-n-mixtures", type=int, default=5)

    args = parser.parse_args()
    return args