import torch
import numpy as np
import argparse


import madrona_basketball as mba


from pathlib import Path
import warnings
warnings.filterwarnings("error")

torch.manual_seed(0)


arg_parser = argparse.ArgumentParser()

# For simulator constructor:
arg_parser.add_argument("--discrete-x", type=int, default=20)
arg_parser.add_argument("--discrete-y", type=int, default=15)
arg_parser.add_argument("--start-x", type=int, default=10.0)
arg_parser.add_argument("--start-y", type=int, default=7.5)
arg_parser.add_argument("--max_episode-length", default=10000)
arg_parser.add_argument("--num-worlds", default=1)
arg_parser.add_argument("--num-steps", default=10000)
arg_parser.add_argument("--gpu-id", type=int, default=0)
arg_parser.add_argument("--gpu-sim", action='store_true')


# For inference specifically
arg_parser.add_argument("--checkpoint-path", required=True)
arg_parser.add_argument("--action-log-path", type=str)

args = arg_parser.parse_args()



# Initializing the simulator
sim = mba.SimpleGridworldSimulator(\
    exec_mode = mba.madrona.ExecMode.CUDA if args.gpu_sim else mba.madrona.ExecMode.CPU

    discrete_x = args.discrete_x,
    discrete_y = args.discrete_y,
    start_x = args.start_x, 
    start_y = arsg.start_y,  
    max_episode_length = args.max_episode_length,
    num_worlds = args.num_worlds,
    gpu_id = args.gpu_id
)



# Policy and Tensor setup
# I have no idea how to do this, the other infer.py in escape room uses some custom functions that create the neural network
# Should it be an RNN? Why??