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

    # Create environment
    environment = EnvWrapper(args.num_envs, args.gpu_sim, args.gpu_id, args.viewer)
    input_dimensions = environment.get_input_dim()
    action_buckets = environment.get_action_buckets()

    # Load policy
    policy = Agent(input_dimensions, num_channels=64, num_layers=3, action_buckets=action_buckets).to(device)
    policy.load(args.checkpoint_path)
    policy.eval()
    print(f"Successfully loaded policies from {args.checkpoint_path}.")

    # Get tensor references - these are the full tensors for all agents
    observations_tensor = environment.observations  # Shape: [num_worlds, num_agents, obs_dim]
    actions_tensor = environment.actions  # Shape: [num_worlds, num_agents, action_dim]
    
    print(f"Observations tensor shape: {observations_tensor.shape}")
    print(f"Actions tensor shape: {actions_tensor.shape}")
    
    # Initialize action tensor to zero actions for all agents
    actions_tensor.fill_(0)

    trajectory_log = []
    static_log = {} # This is just for the hoops
    if args.log_path:
        Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"logging trajectory to {args.log_path}")

    # Reset environment
    obs, reward, done = environment.reset()
    obs = obs.to(device)
    
    print(f"Reset obs shape: {obs.shape}")

    if args.log_path:
        static_log['hoop_pos'] = environment.worlds.hoop_pos_tensor().to_torch().cpu().numpy().copy()

    print("Starting inference")
    if (args.stochastic):
        print("Running policy with sampling from action space")
    else:
        print("Running policy deterministically with best actions only")

    # Initialize episode counters for each environment
    episode_counts = torch.zeros(args.num_envs, dtype=torch.long)
    step = 0

    # Main inference loop
    while step < args.max_steps:
        with torch.no_grad():
            if args.stochastic:
                actions, _, _ = policy.forward(obs)
                obs, reward, done = environment.step(actions)
                obs = obs.to(device)
            else:
                backbone_features = policy.backbone(obs)
                logits = policy.actor(backbone_features)
                action_dists = DiscreteActionDistributions(action_buckets, logits=logits)
                best_actions = torch.zeros(args.num_envs, len(action_buckets), dtype=torch.long, device=device)
                action_dists.best(best_actions)
                
                if step % 50 == 0:
                    print(f"Step {step}: Agent actions = {best_actions[0].cpu().numpy()}")
                
                obs, reward, done = environment.step(best_actions)
                obs = obs.to(device)

        if (args.log_path):
            log_entry = {
                "agent_pos" : environment.worlds.agent_pos_tensor().to_torch().cpu().numpy().copy(),
                "ball_pos" : environment.worlds.basketball_pos_tensor().to_torch().cpu().numpy().copy(),
                "orientation": environment.worlds.orientation_tensor().to_torch().cpu().numpy().copy(),
                "done": done.cpu().numpy().copy()
            }
            trajectory_log.append(log_entry)

        # Update episode counts when an episode is done
        if args.num_episodes > 0:
            episode_counts += done.cpu().long()
            # Check if all environments have completed the required number of episodes
            if torch.all(episode_counts >= args.num_episodes):
                print(f"All environments have completed {args.num_episodes} episodes.")
                break
        
        step += 1
        

    if args.log_path and trajectory_log:
        episode_log = {}
        for key in trajectory_log[0].keys():
            episode_log[key] = np.array([step[key] for step in trajectory_log])

        static_log['num_episodes'] = args.num_episodes

        np.savez_compressed(args.log_path, **static_log, **episode_log)
        print("Finished logging. Trajectory saved to {args.log_path}")
    print("Inference Complete")

    


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
    arg_parser.add_argument("--checkpoint-path", required=True)
    arg_parser.add_argument("--log-path", type=str, default="logs/trajectories.npz")
    arg_parser.add_argument("--max-steps", type=int, default=10000)
    arg_parser.add_argument("--num-episodes", type=int, default=5)
    arg_parser.add_argument("--stochastic", action='store_true', default=True) # Determines if the policy only uses its best action or samples from its distribution


    args = arg_parser.parse_args()
    infer(args)