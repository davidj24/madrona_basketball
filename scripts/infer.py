import torch
import numpy as np
import tyro
import os
import random


import madrona_basketball as mba


from pathlib import Path
from env import EnvWrapper
from agent import Agent
from action import DiscreteActionDistributions
from controllers import SimpleControllerManager
from dataclasses import dataclass
from typing import Optional
import warnings
warnings.filterwarnings("error")


@dataclass
class Args:
    model_name: Optional[str]=None # This is for inferring all versions of a model
    trainee_idx: int=1
    trainee_checkpoint: Optional[str]=None # The model you want to evaluate
    frozen_checkpoint: Optional[str]=None
    log_path: Optional[str]="logs/trajectories.npz"
    max_steps: int=10000
    num_episodes: int=5
    stochastic: bool=True
    viewer: bool=False
    test_seed: int=0

    discrete_x: int=20
    discrete_y: int=15
    start_x: float=10.0
    start_y: float=7.5
    num_envs: int=10
    gpu_id: int=0
    gpu_sim: bool=False

def infer(device, environment, policy, log_path: str = "logs/trajectories.npz", num_episodes: int = 5, max_steps: int = 10000, stochastic: bool = True):
    policy.eval()
    print(f"Successfully loaded {policy}.")

    # Initialize SimpleControllerManager for interactive inference
    controller_manager = SimpleControllerManager(policy, device)
    
    # Connect controller manager to environment for interactive inference
    environment.set_controller_manager(controller_manager)

    # Get tensor references - these are the full tensors for all agents
    observations_tensor = environment.observations  # Shape: [num_worlds, num_agents, obs_dim]
    actions_tensor = environment.actions  # Shape: [num_worlds, num_agents, action_dim]
    
    print(f"Observations tensor shape: {observations_tensor.shape}")
    print(f"Actions tensor shape: {actions_tensor.shape}")
    
    # Initialize action tensor to zero actions for all agents
    actions_tensor.fill_(0)

    trajectory_log = []
    static_log = {} # This is just for the hoops
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        print(f"logging trajectory to {log_path}")

    # Reset environment
    obs, reward, done = environment.reset()
    obs = obs.to(device)
    
    print(f"Reset obs shape: {obs.shape}")

    if log_path:
        static_log['hoop_pos'] = environment.worlds.hoop_pos_tensor().to_torch().cpu().numpy().copy()

    print("Starting inference")
    if (stochastic):
        print("Running policy with sampling from action space")
    else:
        print("Running policy deterministically with best actions only")

    # Initialize episode counters for each environment
    episode_counts = torch.zeros(environment.num_worlds, dtype=torch.long, device=device)
    step = 0

    # Main inference loop
    while step < max_steps:
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

        if (log_path):
            log_entry = {
                "agent_pos" : environment.worlds.agent_pos_tensor().to_torch().cpu().numpy().copy(),
                "ball_pos" : environment.worlds.basketball_pos_tensor().to_torch().cpu().numpy().copy(),
                "ball_vel" : environment.worlds.ball_velocity_tensor().to_torch().cpu().numpy().copy(),
                "orientation" : environment.worlds.orientation_tensor().to_torch().cpu().numpy().copy(),
                "ball_physics" : environment.worlds.ball_physics_tensor().to_torch().cpu().numpy().copy(),
                "agent_possession" : environment.worlds.agent_possession_tensor().to_torch().cpu().numpy().copy(),
                "game_state" : environment.worlds.game_state_tensor().to_torch().cpu().numpy().copy(),
                "rewards" : environment.worlds.reward_tensor().to_torch().cpu().numpy().copy(), # Local because we are tracking the policy actions not the action component from the environment
                "actions" : environment.worlds.action_tensor().to_torch().cpu().numpy().copy(), # Local because we are tracking the policy actions not the action component from the environment
                "done": done.cpu().numpy().copy()
            }
            trajectory_log.append(log_entry)

        # Update episode counts when an episode is done
        if num_episodes > 0:
            episode_counts += done.cpu().long()
            # Check if all environments have completed the required number of episodes
            if torch.all(episode_counts >= num_episodes):
                print(f"All environments have completed {num_episodes} episodes.")
                break
        
        step += 1
        

    if log_path and trajectory_log:
        episode_log = {}
        for key in trajectory_log[0].keys():
            episode_log[key] = np.array([step[key] for step in trajectory_log])

        static_log['num_episodes'] = num_episodes

        np.savez_compressed(log_path, **static_log, **episode_log)
        print(f"Finished logging. Trajectory saved to {log_path}")
    print("Inference Complete")


def multi_gen_infer(device):
    """Saves inferences of all versions of a given model underneath logs/model_name"""
    checkpoint_dir = "checkpoints"
    all_files = os.listdir(checkpoint_dir)
    print(f"All files in {checkpoint_dir}: {all_files}")
    
    checkpoints_to_test = sorted([f for f in all_files if f.startswith(f"{args.model_name}_") and f.endswith(".pth")])
    print(f"Found {len(checkpoints_to_test)} different models to test: {checkpoints_to_test}")
    
    if len(checkpoints_to_test) == 0:
        print(f"No checkpoints found matching pattern '{args.model_name}_*.pth'")
        return



    # Testing loop
    for checkpoint_name in checkpoints_to_test:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        log_path = f"logs/mgi/{args.model_name}_/{checkpoint_name.replace('.pth', '.npz')}"
        
        print(f"Testing checkpoint: {checkpoint_path}")
        print(f"Saving results to: {log_path}")

        torch.manual_seed(args.test_seed)
        np.random.seed(args.test_seed)
        random.seed(args.test_seed)

        environment = EnvWrapper(num_worlds=args.num_envs, frozen_path=args.frozen_checkpoint, viewer=False, trainee_agent_idx=args.trainee_idx)
        evaluated_agent = Agent(environment.get_input_dim(), num_channels=256, num_layers=2, action_buckets=environment.get_action_buckets()).to(device)
        evaluated_agent.load(checkpoint_path)

        infer(device, environment, evaluated_agent, log_path, args.num_episodes, args.max_steps, args.stochastic)




if __name__ == "__main__":
    args = tyro.cli(Args)
    device = torch.device('cuda' if args.gpu_sim and torch.cuda.is_available() else 'cpu')
    print(f"Using {device} for inference")


    if args.model_name is None:
        # Single model inference
        environment = EnvWrapper(args.num_envs, args.gpu_sim, frozen_path=args.frozen_checkpoint, gpu_id=args.gpu_id, viewer=args.viewer)
        input_dimensions = environment.get_input_dim()
        action_buckets = environment.get_action_buckets()

        # Load policy
        policy = Agent(input_dimensions, num_channels=256, num_layers=2, action_buckets=action_buckets).to(device)
        policy.load(args.trainee_checkpoint)

        # Print interactive inference instructions if viewer is enabled
        if args.viewer:
            print("🎮 Interactive mode: Press 'H' to toggle human control for selected agent in World 0")

        infer(device, environment, policy, "logs/inference_trajectories.npz", stochastic=args.stochastic)
    else:
        # Multi-generation inference
        multi_gen_infer(device)


    