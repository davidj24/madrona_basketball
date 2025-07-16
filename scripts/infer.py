import torch
import numpy as np
import argparse


import madrona_basketball as mba


from pathlib import Path
from env import EnvWrapper
from agent import Agent
from action import DiscreteActionDistributions
import warnings
warnings.filterwarnings("error")

torch.manual_seed(0)



def infer(args):

    device = torch.device('cuda' if args.gpu_sim and torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for inference")


    # We don't need to combine multiple observation tensors since our current observation tensor already covers everything, so the setup_obs function from escape_room can just be replaced by the exported observation tensor... right???
    environment = EnvWrapper(1, args.gpu_sim, args.gpu_id, args.viewer)
    input_dimensions = environment.get_input_dim()
    action_buckets=environment.get_action_buckets()

    policy = Agent(input_dimensions, num_channels=64, num_layers=3, action_buckets=action_buckets).to(device)
    policy.load(args.checkpoint_path)
    policy.eval()
    print(f"Successfully loaded policies from {args.checkpoint_path}.")

    observations_tensor = environment.observations.to(device)
    actions_tensor = environment.actions

    trajectory_log = []
    static_log = {} # This is just for the hoops
    if args.log_path:
        Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"logging trajectory to {args.log_path}")

    
    obs, reward, done = environment.reset()
    obs = obs.to(device)

    if args.log_path:
        static_log['hoop_pos'] = environment.worlds.hoop_pos_tensor().to_torch().cpu().numpy().copy()

    print("Starting inference")
    if (args.stochastic):
        print("Running policy with sampling from action space")
    else:
        print("Running policy deterministically with best actions only")
    
    for step in range(args.max_episode_length):
        with torch.no_grad():
            if args.stochastic:
                actions, _, _ = policy.forward(obs)
                actions_tensor[:, environment.agent_idx] = actions
            else:
                backbone_features = policy.backbone(obs)
                logits = policy.actor(backbone_features)
                action_dists = DiscreteActionDistributions(action_buckets, logits=logits)
                action_dists.best(actions_tensor[:, environment.agent_idx])

        if (args.log_path):
            log_entry = {
                "agent_pos" : environment.worlds.agent_pos_tensor().to_torch().cpu().numpy().copy(),
                "ball_pos" : environment.worlds.basketball_pos_tensor().to_torch().cpu().numpy().copy()
            }
            trajectory_log.append(log_entry)
        
        obs, _, done = environment.step(actions_tensor[:, environment.agent_idx])
        obs = obs.to(device)


        if done[0]:
            print(f"episode finished at step {step+1}.")
            break

    if args.log_path and trajectory_log:
        episode_log = {}
        for key in trajectory_log[0].keys():
            episode_log[key] = np.array([step[key] for step in trajectory_log])

        np.savez_compressed(args.log_path, **static_log, **episode_log)
        print("Finished logging. Trajectory saaved to {args.log_path}")
    print("Inference Complete")

    


if __name__ == "__main__":
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
    arg_parser.add_argument("--viewer", action='store_true')
    arg_parser.add_argument("--stochastic", action='store_true') # Determines if the policy only uses its best action or samples from its distribution


    # For inference specifically
    arg_parser.add_argument("--checkpoint-path", required=True)
    arg_parser.add_argument("--log-path", type=str)

    args = arg_parser.parse_args()
    infer(args)