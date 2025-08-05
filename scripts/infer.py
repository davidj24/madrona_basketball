import torch
import numpy as np
import argparse


import madrona_basketball as mba


from pathlib import Path
from env import EnvWrapper
from agent import Agent
from action import DiscreteActionDistributions
from controllers import SimpleControllerManager
import warnings
warnings.filterwarnings("error")

# torch.manual_seed(0)



def infer(args):

    device = torch.device('cuda' if args.gpu_sim and torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for inference")


    # Create environment
    environment = EnvWrapper(args.num_envs, args.gpu_sim, frozen_path=args.checkpoint_1, gpu_id=args.gpu_id, viewer=args.viewer)
    input_dimensions = environment.get_input_dim()
    action_buckets = environment.get_action_buckets()

    # Load policy
    policy = Agent(input_dimensions, num_channels=256, num_layers=2, action_buckets=action_buckets).to(device)
    policy.load(args.checkpoint_0)
    policy.eval()
    print(f"Successfully loaded policies from {args.checkpoint_0}.")

    # Initialize SimpleControllerManager for interactive inference
    controller_manager = SimpleControllerManager(policy, device)
    
    # Connect controller manager to environment for interactive inference
    environment.set_controller_manager(controller_manager)
    
    # Print interactive inference instructions if viewer is enabled
    if args.viewer:
        print("🎮 Interactive mode: Press 'H' to toggle human control for selected agent in World 0")

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
            actions, _, _ = policy(obs, stochastic=args.stochastic)

            # Check if viewer exists and human control is active
            if (environment.viewer is not None and 
                controller_manager.is_human_control_active()):
                
                # Get human input for world 0
                human_action = environment.viewer.get_human_action()
                human_agent_idx = environment.viewer.get_selected_agent_index()
                
                # Convert to correct device/format
                if isinstance(human_action, list):
                    human_action = torch.tensor(human_action, dtype=torch.long, device=device)
                else:
                    human_action = human_action.to(device)
                
                # Use step_with_world_actions to override world 0
                obs, reward, done = environment.step_with_world_actions(
                    actions, 
                    human_action_world_0=human_action, 
                    human_agent_idx=human_agent_idx
                )
            else:
                # Use regular step when no human control
                obs, reward, done = environment.step(actions)
            
            obs = obs.to(device)

        if (args.log_path):
            log_entry = {
                "agent_pos" : environment.worlds.agent_pos_tensor().to_torch().cpu().numpy().copy(),
                "ball_pos" : environment.worlds.basketball_pos_tensor().to_torch().cpu().numpy().copy(),
                "ball_vel" : environment.worlds.ball_velocity_tensor().to_torch().cpu().numpy().copy(),
                "orientation" : environment.worlds.orientation_tensor().to_torch().cpu().numpy().copy(),
                "ball_physics" : environment.worlds.ball_physics_tensor().to_torch().cpu().numpy().copy(),
                "agent_possession" : environment.worlds.agent_possession_tensor().to_torch().cpu().numpy().copy(),
                "actions" : actions.cpu().numpy().copy(), # Local because we are tracking the policy actions not the action component from the environment
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
        print(f"Finished logging. Trajectory saved to {args.log_path}")
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
    arg_parser.add_argument("--checkpoint-0", required=True)
    arg_parser.add_argument("--checkpoint-1", required=False)
    arg_parser.add_argument("--log-path", type=str, default="logs/trajectories.npz")
    arg_parser.add_argument("--max-steps", type=int, default=10000)
    arg_parser.add_argument("--num-episodes", type=int, default=5)
    arg_parser.add_argument("--stochastic", action='store_true', default=True) # Determines if the policy only uses its best action or samples from its distribution
    

    args = arg_parser.parse_args()
    infer(args)
