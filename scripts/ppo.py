import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from agent import Agent
from env import EnvWrapper
from ppo_stats import PPOStats, PPOTimer
from controllers import SimpleControllerManager


@dataclass
class Args:
    seed: int = 1
    torch_deterministic: bool = True
    use_gpu: bool = True
    viewer: bool = False

    # Wandb
    env_id: str = "MadronaBasketball"
    wandb_track: bool = True
    wandb_project_name: str = "MadronaBasketballPPO"
    wandb_entity: str = None
    model_name: Optional[str] = None

    # Algorithm specific arguments
    num_iterations: int = 100_000
    num_envs: int = 8192
    num_rollout_steps: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.001
    vf_coef: float = 4.0
    max_grad_norm: float = 1.0
    target_kl: float = None

    # Self Play
    trainee_idx: Optional[int] = 0
    trainee_checkpoint_path: Optional[str] = None
    frozen_checkpoint_path: Optional[str] = None

    # to be filled in runtime
    rollout_batch_size: int = 0
    minibatch_size: int = 0


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.rollout_batch_size = int(args.num_envs * args.num_rollout_steps)
    args.minibatch_size = int(args.rollout_batch_size // args.num_minibatches)
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = f"{args.env_id}__{args.seed}__{int(time.time())}"
    if args.wandb_track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=model_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{model_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join(
            [f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    # Environment setup
    envs = EnvWrapper(args.num_envs, use_gpu=args.use_gpu, frozen_path=args.frozen_checkpoint_path, gpu_id=0, viewer=args.viewer, trainee_agent_idx=args.trainee_idx)
    obs_size = envs.get_input_dim()
    act_size = envs.get_action_space_size()
    action_buckets = envs.get_action_buckets()

    print(f"\n🎯 TRAINING CONFIGURATION:")
    print(f"   Trainee Agent Index: {args.trainee_idx} ({'Offensive Player (with ball)' if args.trainee_idx == 0 else 'Defensive Player (without ball)'})")
    if args.frozen_checkpoint_path:
        print(f"   Frozen Agent: {1-args.trainee_idx} ({'Offensive Player (with ball)' if 1-args.trainee_idx == 0 else 'Defensive Player (without ball)'})")
        print(f"   Frozen Checkpoint: {args.frozen_checkpoint_path}")
    else:
        print(f"   No frozen agent (single agent training)")
    print(f"   Model Name: {args.model_name}")
    print(f"   Iterations: {args.num_iterations}")
    print(f"   Environments: {args.num_envs}")
    print("="*60)

    agent = Agent(obs_size, num_channels=256, num_layers=2,
                  action_buckets=action_buckets).to(device)
    if (args.trainee_checkpoint_path is not None):
        agent.load(args.trainee_checkpoint_path)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    # Initialize SimpleControllerManager for interactive training
    controller_manager = SimpleControllerManager(agent, device)
    
    # Connect controller manager to environment for interactive training
    envs.set_controller_manager(controller_manager)
    
    # Print interactive training instructions if viewer is enabled
    if args.viewer:
        print("\n" + "="*60)
        print("🎮 INTERACTIVE TRAINING MODE ENABLED")
        print("="*60)
        print("Controls:")
        print("  H                - Toggle human control for selected agent in WORLD 0")
        print("  Ctrl+P           - Pause/resume training") 
        print("  R                - Reset simulation")
        print("  Left Click       - Select agent to control")
        print("")
        print("Human Control (when active on World 0):")
        print("  WASD             - Move selected agent")
        print("  Q/E or ,/.       - Rotate selected agent")
        print("  Left Shift       - Grab ball")
        print("  B                - Pass ball")
        print("  Enter/Right Shift - Shoot ball")
        print("")
        print("="*60)
        print("Training will proceed normally. Use H to take control when needed.\n")

    # Storage setup
    obs = torch.zeros((args.num_rollout_steps, args.num_envs, obs_size)).to(device)
    actions = torch.zeros((args.num_rollout_steps, args.num_envs, act_size)).to(device)
    log_probs = torch.zeros((args.num_rollout_steps, args.num_envs, act_size)).to(device)
    rewards = torch.zeros((args.num_rollout_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_rollout_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_rollout_steps, args.num_envs)).to(device)

    # Train start
    stats = PPOStats()
    ppo_timer = PPOTimer()

    next_obs, _, _ = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)
    for iteration in range(1, args.num_iterations + 1):
        ppo_timer.start_iter()

        # Begin rollouts
        ppo_timer.start_rollout()
        for step in range(0, args.num_rollout_steps):
            ppo_timer.update_step(args.num_envs)

            # policy inference
            with torch.no_grad():
                ppo_timer.start_inference()
                rl_action, log_prob, value = agent(next_obs)
                values[step] = value.flatten()
                ppo_timer.end_inference()

            # sim step
            ppo_timer.start_sim()
            if (hasattr(envs, 'viewer') and envs.viewer is not None and 
                controller_manager.is_human_control_active()):
                # Get human action for world 0 and the selected agent
                try:
                    selected_agent_idx = envs.viewer.get_selected_agent_index()
                    human_action_world_0 = controller_manager.get_action(next_obs[0], envs.viewer)
                    next_obs, reward, next_done = envs.step_with_world_actions(rl_action, human_action_world_0, selected_agent_idx)
                except Exception as e:
                    print(f"Warning: Human control error in step: {e}")
                    next_obs, reward, next_done = envs.step(rl_action)
            else:
                next_obs, reward, next_done = envs.step(rl_action)

            ppo_timer.end_sim()

            # store
            obs[step] = next_obs
            dones[step] = next_done
            actions[step] = rl_action
            log_probs[step] = log_prob
            rewards[step] = reward.view(-1)

        # Advantages and bootstrap
        with torch.no_grad():
            # update observation normalizer
            agent.update_obs_normalizer(obs)

            # invert value normalizer
            next_value = agent.get_value(next_obs).reshape(1, -1)
            # values = agent.unnorm_value(values)
            # next_value = agent.unnorm_value(next_value)

            # bootstrap value if not done
            advantages = torch.zeros_like(rewards).to(device)
            last_gae_lam = 0
            for t in reversed(range(args.num_rollout_steps)):
                if t == args.num_rollout_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]
                delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
                advantages[t] = last_gae_lam = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae_lam
            returns = advantages + values

            # normalize returns and update value normalizer
            # agent.update_value_normalizer(returns.view(-1, 1))
            # returns = agent.normalize_value(returns)

        ppo_timer.end_rollout()

        # Flatten the batch
        b_obs = obs.reshape((-1, obs_size))
        b_logprobs = log_probs.reshape((-1, act_size))
        b_actions = actions.reshape((-1, act_size))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimization steps
        ppo_timer.start_update()
        b_inds = np.arange(args.rollout_batch_size)
        for epoch in range(args.update_epochs):
            # Sample minibatches
            np.random.shuffle(b_inds)
            for start in range(0, args.rollout_batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                new_log_prob, entropy, new_value = agent.get_stats(
                    b_obs[mb_inds], b_actions[mb_inds])
                ratio = torch.exp(new_log_prob - b_logprobs[mb_inds])

                mb_advantages = b_advantages[mb_inds]
                sigma, mu = torch.std_mean(mb_advantages, dim=0, unbiased=True)
                mb_advantages = (mb_advantages - mu) / (sigma + 1e-8)

                # Policy loss
                mb_advantages = mb_advantages.reshape(-1, 1)
                pg_loss1 = -ratio * mb_advantages
                pg_loss2 = -torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * mb_advantages
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_value = new_value.view(-1)
                v_loss = ((new_value - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                loss = (pg_loss
                        - args.ent_coef * entropy_loss
                        + 0.5 * v_loss * args.vf_coef)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # Update the stats
                returns_mean, returns_stdev = torch.var_mean(b_returns[mb_inds])
                rewards_mean, rewards_min, rewards_max = rewards.mean(), rewards.min(), rewards.max()
                stats.update(loss.item(), pg_loss.item(), v_loss.item(), entropy_loss.item(),
                              returns_mean.item(), returns_stdev.item(), rewards_mean.item(),
                              rewards_min.item(), rewards_max.item())

        ppo_timer.end_update()
        ppo_timer.end_iter()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        fps = ppo_timer.get_iter_fps()
        global_step = ppo_timer.global_step
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/SPS", fps, global_step)

        # Print every 100 update iterations
        if iteration % 100 == 0:
            p_advantages = b_advantages.reshape(-1)
            p_values = b_values.reshape(-1)
            update_timer_end = time.perf_counter()

            print(f"\nUpdate: {iteration} took {ppo_timer.t_iter:.4f} seconds. FPS: {fps}")
            print(f"    Sim only: {ppo_timer.t_sim:.4f}s, Inference: {ppo_timer.t_inference:.4f}s, Update: {ppo_timer.t_update:.4f}s")
            print(f"    Loss: {stats.loss: .3e}, A: {stats.action_loss: .3e}, V: {stats.value_loss: .3e}, E: {stats.entropy_loss: .3e}")
            print()
            print(f"    Rewards          => Avg: {stats.rewards_mean: .3f}, Min: {stats.rewards_min: .3f}, Max: {stats.rewards_max: .3f}")
            print(f"    Values           => Avg: {p_values.mean(): .3f}, Min: {p_values.min(): .3f}, Max: {p_values.max(): .3f}")
            print(f"    Advantages       => Avg: {p_advantages.mean(): .3f}, Min: {p_advantages.min(): .3f}, Max: {p_advantages.max(): .3f}")
            print(f"    Returns          => Avg: {stats.returns_mean}")
            stats.reset()
            ppo_timer.reset()

            update_timer_start = time.perf_counter()


        # Every 100 iterations, save the model
        if iteration % 100 == 0:
            folder = "checkpoints"
            if not os.path.exists(folder):
                os.makedirs(folder)

            if (args.model_name):
                torch.save(agent.state_dict(), os.path.join(folder, f"{args.model_name}_{iteration}.pth"))
                print(f"Model {args.model_name} saved at iteration {iteration}")
                
            else:
                torch.save(agent.state_dict(), os.path.join(folder, f"{iteration}.pth"))
                print(f"Model saved at iteration {iteration}")

    writer.close()
