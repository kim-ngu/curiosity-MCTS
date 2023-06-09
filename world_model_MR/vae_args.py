import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()

    # ---------- General settings --------- #
    
    parser.add_argument("--file-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--gym-env", type=str, default="ALE/MontezumaRevenge-v5")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug", action='store_true')

    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--save_interval", type=int, default=20)

    # ---------- Vae settings ---------- #

    # Training
    parser.add_argument("--vae-n-epochs", type=int, default=10)
    parser.add_argument("--vae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--vae-batch-size", type=int, default=32)

    # Network setup
    parser.add_argument("--img-size", type=int, default=96)

    args = parser.parse_args()
    return args