import tyro
import subprocess
import os
import torch
import random

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
    viewer: bool = True


    # Self Play Stuff
    num_training_cycles: int = 5
    iter_per_agent: int = 5000
    first_trainee_idx: int = 1
    checkpoint_0: Optional[str] = None
    checkpoint_1: Optional[str] = None





def run_ppo(trainee_idx: int, args: Args, trainee_checkpoint: str, frozen_checkpoint: str, model_name: str):

    command = [
        "python3", "scripts/ppo.py",
        "--model-name", model_name,
        "--trainee-idx", str(trainee_idx),
        "--trainee-checkpoint-path", trainee_checkpoint,
        "--frozen-checkpoint-path", frozen_checkpoint,
        "--num-iterations", str(args.iter_per_agent),
        "--num-envs", str(args.num_envs),
        "--save-model-every-n-iterations", str(args.iter_per_agent//10)
    ]

    if not args.use_gpu:
        command.append("--no-use-gpu")
    if not args.wandb_track:
        command.append("--no-wandb-track")
    if args.viewer:
        command.append("--viewer")

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
        model_0_path = os.path.join(CHECKPOINT_DIR, f'{args.model_name_0}_initial' if args.model_name_0 is not None else 'model_0_initial')
        if not os.path.exists(model_0_path):
            print(f"Creating model0 policy at {model_0_path}")
            obs_size = 128
            action_buckets = [2, 8, 3, 2, 2, 2]
            model_0 = Agent(obs_size,num_channels=256, num_layers=2, action_buckets=action_buckets)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(model_0.state_dict(), model_0_path)
    else: 
        model_0_path = args.checkpoint_0

    if args.checkpoint_1 is None:
        model_1_path = os.path.join(CHECKPOINT_DIR, f'{args.model_name_1}_initial' if args.model_name_1 is not None else 'model_1_initial')
        if not os.path.exists(model_1_path):
            print(f"Creating model0 policy at {model_1_path}")
            obs_size = 128
            action_buckets = [2, 8, 3, 2, 2, 2]
            model_1 = Agent(obs_size,num_channels=256, num_layers=2, action_buckets=action_buckets)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            torch.save(model_1.state_dict(), model_1_path)
    else: 
        model_1_path = args.checkpoint_1


    # Vector to hold old models/generations for trainee to go against randomly. Models are stored in chronological order
    model_list = []
    max_models_in_list = 3 # The maximum number of models per agent we want to keep in the list at a time. Ex: if max_models = 3, we will have a max of 6 total models in the list, 3 for each agent
    probability_train_against_old_model = 0

    second_model_path = model_1_path if args.first_trainee_idx == 0 else model_0_path
    for generation in range(args.num_training_cycles):
        first_model_name = f"{args.model_name_0 if args.first_trainee_idx == 0 else args.model_name_1}_gen_{generation}"
        second_model_name = f"{args.model_name_1 if args.first_trainee_idx == 0 else args.model_name_0}_gen_{generation}"
    

        # ====================== FIRST TRAINING SESSION ========================
        first_model_path = model_0_path if args.first_trainee_idx == 0 else model_1_path
        model_list.append(first_model_path)
        print(f"\n🔄 GENERATION {generation} - FIRST TRAINING SESSION")
        print(f"   Training: {first_model_name} (Agent {args.first_trainee_idx} - {'Offensive' if args.first_trainee_idx == 0 else 'Defensive'}) with path: {first_model_path}")
        print(f"   Against:  Agent {1-args.first_trainee_idx} ({'Offensive' if 1-args.first_trainee_idx == 0 else 'Defensive'}) - {second_model_path}")
        run_ppo(args.first_trainee_idx, args, first_model_path, second_model_path, first_model_name)
        
        # Update the model path after first training
        if args.first_trainee_idx == 0:
            model_0_path = f"{CHECKPOINT_DIR}/{first_model_name}_{args.iter_per_agent}.pth"
            first_model_path = model_0_path
        else:
            model_1_path = f"{CHECKPOINT_DIR}/{first_model_name}_{args.iter_per_agent}.pth"
            first_model_path = model_1_path


        rand_num = random.randint(1, 100)
        if rand_num <= probability_train_against_old_model:
            model_num = random.randrange(0, len(model_list), 2)
            first_model_path = model_list[model_num] # Train second agent against older model with some probability
            print(f"Random Older Opponent chosen. In the next training session, the opponent will be: {first_model_path}")


        # ====================== SECOND TRAINING SESSION ========================
        second_model_path = model_1_path if args.first_trainee_idx == 0 else model_0_path
        model_list.append(second_model_path)

        if len(model_list) > 2*max_models_in_list: # If the model list has gotten too long, remove the oldest 2
            print(f"Retiring models {model_list[0]} and {model_list[1]}. They're too old now.")
            del model_list[0]
            del model_list[1]

        print(f"\n🔄 GENERATION {generation} - SECOND TRAINING SESSION")
        print(f"   Training: {second_model_name} (Agent {1-args.first_trainee_idx} - {'Offensive' if 1-args.first_trainee_idx == 0 else 'Defensive'}) with path: {second_model_path}")
        print(f"   Against:  Agent {args.first_trainee_idx} ({'Offensive' if args.first_trainee_idx == 0 else 'Defensive'}) - {first_model_path}")

        run_ppo(1-args.first_trainee_idx, args, second_model_path, first_model_path, second_model_name)
        if args.first_trainee_idx == 0:
            model_1_path = f"{CHECKPOINT_DIR}/{second_model_name}_{args.iter_per_agent}.pth"
            second_model_path = model_1_path
        else:
            model_0_path = f"{CHECKPOINT_DIR}/{second_model_name}_{args.iter_per_agent}.pth"
            second_model_path = model_0_path

        rand_num = random.randint(1, 100)
        if rand_num <= probability_train_against_old_model:
            model_num = random.randrange(1, len(model_list), 2)
            second_model_path = model_list[model_num]
            print(f"Random Older Opponent chosen. In the next training session, the opponent will be: {second_model_path}")

        print(f"\n✅ Cycle {generation}/{args.num_training_cycles-1} complete.")
        print(f"   Current Agent 0 (Offensive): {model_0_path}")
        print(f"   Current Agent 1 (Defensive): {model_1_path}")




