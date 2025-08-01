import torch
import numpy as np
import argparse


import madrona_basketball as mba


from pathlib import Path
from env import EnvWrapper
from agent import Agent
from action import DiscreteActionDistributions
from controllers import SimpleControllerManager
from helpers import infer
import warnings
warnings.filterwarnings("error")







if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    # For simulator constructor:
    arg_parser.add_argument("--discrete-x", type=int, default=20)
    arg_parser.add_argument("--discrete-y", type=int, default=15)
    arg_parser.add_argument("--start-x", type=int, default=10.0)
    arg_parser.add_argument("--start-y", type=int, default=7.5)
    arg_parser.add_argument("--num-envs", type=int, default=10)
    arg_parser.add_argument("--gpu-id", type=int, default=0)
    arg_parser.add_argument("--gpu-sim", action='store_true')

    # For inference specifically
    arg_parser.add_argument("--viewer", action='store_true', default=True)
    arg_parser.add_argument("--checkpoint-0", required=True)
    arg_parser.add_argument("--checkpoint-1", required=False)
    arg_parser.add_argument("--log-path", type=str, default="logs/trajectories.npz")
    arg_parser.add_argument("--max-steps", type=int, default=10000)
    arg_parser.add_argument("--num-episodes", type=int, default=5)
    arg_parser.add_argument("--stochastic", action='store_true', default=True) # Determines if the policy only uses its best action or samples from its distribution
    
    args = arg_parser.parse_args()
    device = torch.device('cuda' if args.gpu_sim and torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for inference")


    # Create environment
    environment = EnvWrapper(args.num_envs, args.gpu_sim, frozen_path=args.checkpoint_1, gpu_id=args.gpu_id, viewer=args.viewer)
    input_dimensions = environment.get_input_dim()
    action_buckets = environment.get_action_buckets()

    # Load policy
    policy = Agent(input_dimensions, num_channels=64, num_layers=3, action_buckets=action_buckets).to(device)
    policy.load(args.checkpoint_0)

    # Print interactive inference instructions if viewer is enabled
    if args.viewer:
        print("🎮 Interactive mode: Press 'H' to toggle human control for selected agent in World 0")

    infer(device, environment, policy, "logs/inference_trajectories.npz", stochastic=args.stochastic)


    
