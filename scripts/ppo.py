import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

from agent import Agent
from env import EnvWrapper
from ppo_stats import PPOStats
from controllers import SimpleControllerManager


@dataclass
class Args:
    seed: int = 1
    torch_deterministic: bool = True
    use_gpu: bool = True
    viewer: bool = True

    # Wandb
    env_id: str = "MadronaBasketball"
    wandb_track: bool = True
    wandb_project_name: str = "MadronaBasketballPPO"
    wandb_entity: str = None
    model_name: Optional[str] = None

    # Algorithm specific arguments
    num_iterations: int = 100_000
    num_envs: int = 8192
    num_rollout_steps: int = 100
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.999
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
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
    envs = EnvWrapper(args.num_envs, use_gpu=args.use_gpu, frozen_path=args.frozen_checkpoint_path, gpu_id=0, viewer=args.viewer)
    obs_size = envs.get_input_dim()
    act_size = envs.get_action_space_size()
    action_buckets = envs.get_action_buckets()

    agent = Agent(obs_size, num_channels=64, num_layers=3,
                  action_buckets=action_buckets).to(device)
    if (args.trainee_checkpoint_path is not None):
        agent.load(args.trainee_checkpoint_path)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

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
    global_step = 0
    start_time = time.time()
    next_obs, _, _ = envs.reset()
    next_done = torch.zeros(args.num_envs).to(device)
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        # Begin rollouts
        for step in range(0, args.num_rollout_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                rl_action, log_prob, value = agent(next_obs)
                values[step] = value.flatten()
                
            actions[step] = rl_action
            log_probs[step] = log_prob

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
            rewards[step] = reward.view(-1)
            # time.sleep(1)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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

        # Flatten the batch
        b_obs = obs.reshape((-1, obs_size))
        b_logprobs = log_probs.reshape((-1, act_size))
        b_actions = actions.reshape((-1, act_size))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimization steps
        b_inds = np.arange(args.rollout_batch_size)
        clip_fracs = []
        for epoch in range(args.update_epochs):
            # Sample minibatches
            np.random.shuffle(b_inds)
            for start in range(0, args.rollout_batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                new_log_prob, entropy, new_value = agent.get_stats(
                    b_obs[mb_inds], b_actions[mb_inds])
                log_ratio = new_log_prob - b_logprobs[mb_inds]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                                mb_advantages.std() + 1e-8)

                # Policy loss
                mb_advantages = mb_advantages.reshape(-1, 1)
                pg_loss1 = -ratio * mb_advantages
                pg_loss2 = -torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef) * mb_advantages
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_value = new_value.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        new_value - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

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


            # Break if the approximate KL divergence exceeds the target
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clip_fracs), global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Print every 100 update iterations
        if iteration % 100 == 0:
            p_advantages = b_advantages.reshape(-1)
            p_values = b_values.reshape(-1)

            print(f"\nUpdate: {iteration}")
            print(f"    Loss: {stats.loss: .3e}, A: {stats.action_loss: .3e}, V: {stats.value_loss: .3e}, E: {stats.entropy_loss: .3e}")
            print()
            print(f"    Rewards          => Avg: {stats.rewards_mean: .3f}, Min: {stats.rewards_min: .3f}, Max: {stats.rewards_max: .3f}")
            print(f"    Values           => Avg: {p_values.mean(): .3f}, Min: {p_values.min(): .3f}, Max: {p_values.max(): .3f}")
            print(f"    Advantages       => Avg: {p_advantages.mean(): .3f}, Min: {p_advantages.min(): .3f}, Max: {p_advantages.max(): .3f}")
            print(f"    Returns          => Avg: {stats.returns_mean}")
            stats.reset()

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
