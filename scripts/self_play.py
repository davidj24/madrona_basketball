import tyro
import subprocess
import os
import torch

from dataclasses import dataclass
from typing import Optional
from agent import Agent



@dataclass
class Args:
    wandb_track: bool=True
    use_gpu: bool = True
    num_envs: int = 8192
    model_name_0: Optional[str] = 'model_0'
    model_name_1: Optional[str] = 'model_1'


    # Self Play Stuff
    num_training_cycles: int = 5
    iter_per_agent: int = 500
    first_trainee_idx: int = 0
    checkpoint_0: Optional[str] = None
    checkpoint_1: Optional[str] = None





def run_ppo(trainee_idx: int, args: Args, trainee_checkpoint: str, frozen_checkpoint: str, model_name: str):

    command = [
        "python", "scripts/ppo.py",
        "--model-name", model_name,
        "--trainee-idx", str(trainee_idx),
        "--trainee-checkpoint-path", trainee_checkpoint,
        "--frozen-checkpoint-path", frozen_checkpoint,
        "--num-iterations", str(args.iter_per_agent),
        "--num-envs", str(args.num_envs)
    ]

    if not args.use_gpu:
        command.append("--no-use-gpu")
    if not args.wandb_track:
        command.append("--no-wandb-track")

    print(f"Executing command {''.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training session for {model_name} failed with exit code {e.returncode}")
        exit()
    print(f"Finished training for {model_name}")




if __name__ == '__main__':
    args = tyro.cli(Args)
    CHECKPOINT_DIR = 'checkpoints'


    if args.checkpoint_0 is None:
        model_0_path = os.path.join(CHECKPOINT_DIR, 'model_0_initial')
        if not os.path.exists(model_0_path):
            print(f"Creating model0 policy at {model_0_path}")
            obs_size = 128
            action_buckets = [2, 8, 3, 2, 2, 2]
            model_0 = Agent(obs_size,num_channels=64, num_layers=3, action_buckets=action_buckets)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(model_0.state_dict(), model_0_path)
    else: 
        model_0_path = args.checkpoint_0

    if args.checkpoint_1 is None:
        model_1_path = os.path.join(CHECKPOINT_DIR, 'model_1_initial')
        if not os.path.exists(model_1_path):
            print(f"Creating model0 policy at {model_1_path}")
            obs_size = 128
            action_buckets = [2, 8, 3, 2, 2, 2]
            model_1 = Agent(obs_size,num_channels=64, num_layers=3, action_buckets=action_buckets)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(model_1.state_dict(), model_1_path)
    else: 
        model_1_path = args.checkpoint_1



    for generation in range(args.num_training_cycles):
        first_model_name = f"{args.model_name_0 if args.first_trainee_idx == 0 else args.model_name_1}_gen_{generation}"
        second_model_name = f"{args.model_name_1 if args.first_trainee_idx == 0 else args.model_name_0}_gen_{generation}"
        first_model_path = model_0_path if args.first_trainee_idx == 0 else model_1_path
        second_model_path = model_1_path if args.first_trainee_idx == 0 else model_0_path


        run_ppo(args.first_trainee_idx, args, first_model_path, second_model_path, first_model_name)
        if args.first_trainee_idx == 0:
            model_0_path = f"{CHECKPOINT_DIR}/{first_model_name}_{args.iter_per_agent}.pth"
        else:
            model_1_path = f"{CHECKPOINT_DIR}/{first_model_name}_{args.iter_per_agent}.pth"


        run_ppo(1-args.first_trainee_idx, args, second_model_path, first_model_path, second_model_name)
        if args.first_trainee_idx == 0:
            model_1_path = f"{CHECKPOINT_DIR}/{second_model_name}_{args.iter_per_agent}.pth"
        else:
            model_0_path = f"{CHECKPOINT_DIR}/{second_model_name}_{args.iter_per_agent}.pth"


        print(f"Cycle {generation}/{args.num_training_cycles} complete.")





