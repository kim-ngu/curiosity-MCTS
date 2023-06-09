import argparse
import os
from distutils.util import strtobool

def parse_args():
    parser = argparse.ArgumentParser()

    # Basics
    parser.add_argument("--gym-env", type=str, default="FrozenLake-v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--debug", action='store_true')

    # Hyperparameters
    parser.add_argument("--total-timesteps", type=int, default=10**6,
        help="total timesteps of the experiments")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout. Total rollout data: num-steps x num-envs")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=3,
        help="the K epochs to update the policy")    
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    
    # PPO Hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="lambda for the general advantage estimation")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="clipping coefficient (epsilon)")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="value function coefficient (c1)")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="entropy bonus coefficient (c2)")

    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps) # Batch size = num-steps x num-envs (Total rollout data)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args    