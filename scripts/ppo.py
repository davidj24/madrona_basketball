import madrona_basketball as mba

import os
import random
import time
import subprocess

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
from buffers import RolloutBuffer


@dataclass
class Args:
    seed: int = 321
    torch_deterministic: bool = True
    use_gpu: bool = True
    full_viewer: bool = False
    viewer: bool = True
    log_every_n_iterations: int = 100
    save_model_every_n_iterations: int = 100

    trainee_idx: Optional[int] = 1
    trainee_checkpoint: Optional[str] = None
    frozen_checkpoint: Optional[str] = None

    env_id: str = "MadronaBasketball"
    model_name: Optional[str] = "Model"

    num_iterations: int = 100_000
    num_envs: int = 8192
    num_rollout_steps: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.998
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 1.0
    max_grad_norm: float = 1.0

    # to be filled in runtime
    rollout_batch_size: int = 0
    minibatch_size: int = 0


@torch.no_grad()
def rollout(agent, env, buffer: RolloutBuffer, stats, timer, controller_manager, 
           is_recording: bool, is_waiting_for_new_episode: bool, recorded_trajectory: list, 
           static_log: dict, iteration: int, log_folder: str):
    obs, _, _ = env.reset()
    for step in range(0, args.num_rollout_steps):
        timer.start_inference()
        actions, log_probs, values = agent(obs)
        timer.end_inference()

        # sim step
        timer.start_sim()

        if (hasattr(env, 'viewer') and env.viewer is not None and
                controller_manager.is_human_control_active()):
            try:
                selected_agent_idx = env.viewer.get_selected_agent_index()
                human_action_world_0 = controller_manager.get_action(
                    obs[0], env.viewer)
                obs_, rews, dones = env.step_with_world_actions(
                    actions, human_action_world_0, selected_agent_idx)

            except Exception as e:
                print(f"Warning: Human control error in step: {e}")
                obs_, rews, dones = env.step(actions)
        else:
            obs_, rews, dones = env.step(actions)

        stats.update(rews, dones)
        not_dones = 1.0 - dones
        timer.end_sim()

        # ======================================== LOGGING ========================================
        if is_recording and recorded_trajectory is not None:
            log_entry = {
                "agent_pos": env.worlds.agent_pos_tensor().to_torch()[:1].cpu().numpy().copy(),
                "ball_pos": env.worlds.basketball_pos_tensor().to_torch()[:1].cpu().numpy().copy(),
                "ball_vel": env.worlds.ball_velocity_tensor().to_torch()[:1].cpu().numpy().copy(),
                "orientation": env.worlds.orientation_tensor().to_torch()[:1].cpu().numpy().copy(),
                "ball_physics": env.worlds.ball_physics_tensor().to_torch()[:1].cpu().numpy().copy(),
                "agent_possession": env.worlds.agent_possession_tensor().to_torch()[:1].cpu().numpy().copy(),
                "game_state": env.worlds.game_state_tensor().to_torch()[:1].cpu().numpy().copy(),
                "rewards": env.worlds.reward_tensor().to_torch()[:1].cpu().numpy().copy(),
                "actions": env.worlds.action_tensor().to_torch()[:1].cpu().numpy().copy(),
                "done": dones[:1].cpu().numpy().copy()
            }
            recorded_trajectory.append(log_entry)

        if is_recording and dones[0].item() > 0 and recorded_trajectory is not None and log_folder is not None:
            print(f"Recorded episode has finished.")
            log_path = os.path.join(log_folder, f"iter_{iteration}_episode.npz")
            episode_log = {}
            for key in recorded_trajectory[0].keys():
                episode_log[key] = np.array([step[key] for step in recorded_trajectory])
            
            # Save static data with corrected hoop positions for proper rendering
            corrected_static_log = static_log.copy()
            corrected_static_log['hoop_pos'] = np.array([[[3.25, 8.5, 0.0], [28.75, 8.5, 0.0]]])
            episode_log['hoop_pos'] = corrected_static_log['hoop_pos']
            np.savez_compressed(log_path, **episode_log)
            print(f"Episode trajectory saved to {log_path}")
            recorded_trajectory.clear()
            return False, False

        if is_waiting_for_new_episode and dones[0].item() > 0:
            print(f"Episode of world 0 has just ended. Starting recording next step.")
            return True, False

        # Store
        buffer.obs[step] = obs
        buffer.actions[step] = actions
        buffer.values[step] = values
        buffer.log_probs[step] = log_probs
        buffer.not_dones[step] = not_dones
        buffer.rewards[step] = rews

        if step == buffer.horizon - 1:
            buffer.next_value[:] = agent.evaluate(obs_)

        obs = obs_
    
    return is_recording, is_waiting_for_new_episode


@torch.no_grad()
def compute_advantages(buffer, agent, gamma, gae_lambda):
    advantages = buffer.advantages
    rewards = buffer.rewards
    returns = buffer.returns
    values = agent.unnorm_value(buffer.values)
    next_value = agent.unnorm_value(buffer.next_value)

    # Bootstrap
    advantages[:] = 0.0
    last_gae_lam = 0.0
    for t in reversed(range(buffer.horizon)):
        if t == buffer.horizon - 1:
            next_non_terminal = buffer.not_dones[t]
            next_values = next_value
        else:
            next_non_terminal = buffer.not_dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        advantages[
            t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

    returns[:] = advantages + values
    # Update normalizers
    agent.obs_norm.update(buffer.obs.view(-1, buffer.obs.shape[-1]))
    agent.value_norm.update(values.view(-1, 1))
    agent.value_norm.update(returns.view(-1, 1))

    # Normalize advantages, values, returns
    mu, sigma = advantages.mean(), advantages.std()
    advantages[:] = (advantages - mu) / (sigma + 1e-8)
    values[:] = agent.value_norm(values)
    returns[:] = agent.value_norm(returns)
    return


def update_policy(buffer, agent, optimizer, device):
    total_steps = buffer.get_total_steps()
    minibatch_size = total_steps // args.num_minibatches

    # Mini epochs
    for epoch in range(args.update_epochs):
        b_inds = torch.randperm(total_steps, device=device)
        # Sample minibatches
        for start in range(0, total_steps, minibatch_size):
            mb_inds = b_inds[start:start + minibatch_size]
            o, a, lp, v, adv, ret = buffer.get_minibatch(mb_inds)
            # stats of batch
            lp_, e, v_ = agent.get_stats(o, a)

            # actor loss
            ratio = torch.exp(lp_ - lp)
            surr1 = -adv * ratio
            surr2 = -adv * torch.clamp(ratio, 1 - args.clip_coef,
                                       1 + args.clip_coef)
            pg_loss = torch.max(surr1, surr2).mean()
            # critic loss
            vf_loss = (v_ - ret) ** 2
            v_clip = v + (v_ - v).clamp(-args.clip_coef, args.clip_coef)
            vf_loss_clip = (v_clip - ret) ** 2
            c_loss = 0.5 * torch.max(vf_loss, vf_loss_clip).mean()
            # entropy loss
            entropy_loss = -e.mean()

            c_loss *= args.vf_coef
            entropy_loss *= args.ent_coef
            loss = pg_loss + c_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
    return



def main():
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")

    envs = EnvWrapper(args.num_envs, use_gpu=args.use_gpu,
                      frozen_path=args.frozen_checkpoint, gpu_id=0,
                      viewer=args.full_viewer,
                      trainee_agent_idx=args.trainee_idx)
    obs_size = envs.get_input_dim()
    act_size = envs.get_action_space_size()
    action_buckets = envs.get_action_buckets()

    print(f"\n🎯 TRAINING CONFIGURATION:")
    print(
        f"   Trainee Agent Index: {args.trainee_idx} ({'Offensive Player (with ball)' if args.trainee_idx == 0 else 'Defensive Player (without ball)'})")
    if args.frozen_checkpoint:
        print(
            f"   Frozen Agent: {1 - args.trainee_idx} ({'Offensive Player (with ball)' if 1 - args.trainee_idx == 0 else 'Defensive Player (without ball)'})")
        print(f"   Frozen Checkpoint: {args.frozen_checkpoint}")
    else:
        print(f"   No frozen agent (single agent training)")
    print(f"   Model Name: {args.model_name}")
    print(f"   Iterations: {args.num_iterations}")
    print(f"   Environments: {args.num_envs}")
    print("=" * 60)

    agent = Agent(obs_size, num_channels=32, num_layers=2,
                  action_buckets=action_buckets).to(device)
    if (args.trainee_checkpoint is not None):
        agent.load(args.trainee_checkpoint)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

    controller_manager = SimpleControllerManager(agent, device)
    envs.set_controller_manager(controller_manager)

    viewer_process = None
    log_folder = f"logs/{model_name}"
    if args.viewer:
        print(f"Setting up viewer process...")
        os.makedirs(log_folder, exist_ok=True)

        command = [
            "python3", "-m", "scripts.viewer",
            "--live-log-folder", log_folder
        ]
        try:
            viewer_process = subprocess.Popen(command)
            print(f"Viewer process started with PID: {viewer_process.pid}")
            print(f"Viewer is now watching: {log_folder}")
        except Exception as e:
            print(f"Failed to start viewer process: {e}")
            viewer_process = None

    # ======================================== Start Training ========================================
    stats = PPOStats(n_envs=args.num_envs, device=device)
    timer = PPOTimer()
    writer = SummaryWriter(f"runs/{model_name}")

    # Allocate storage
    buffer = RolloutBuffer(
        n_steps=args.num_rollout_steps,
        n_envs=args.num_envs,
        obs_dim=obs_size,
        act_dim=act_size,
        device=device
    )

    static_log = {}
    static_log['hoop_pos'] = envs.worlds.hoop_pos_tensor().to_torch().cpu().numpy().copy()

    # Initialize logging variables
    is_recording = False
    is_waiting_for_new_episode = False
    recorded_trajectory = []

    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-8)
    for iteration in range(1, args.num_iterations + 1):
        timer.start_iter()
        timer.add_steps(args.num_envs * args.num_rollout_steps)

        # Collect rollouts
        timer.start_rollout()
        agent.eval()
        is_recording, is_waiting_for_new_episode = rollout(agent, envs, buffer, stats, timer, controller_manager, is_recording, is_waiting_for_new_episode, recorded_trajectory, static_log, iteration, log_folder)
        timer.end_rollout()

        # Advantages
        compute_advantages(buffer, agent, args.gamma, args.gae_lambda)
        

        # Optimization steps
        timer.start_update()
        agent.train()
        update_policy(buffer, agent, optimizer, device)
        timer.end_update()

        timer.end_iter()

        # Logging
        if iteration % args.log_every_n_iterations == 0:
            print(f"\nUpdate: {iteration}", end=' ')
            timer.print()
            stats.print()

        timer.reset()
        stats.reset()

        if args.viewer and iteration % args.log_every_n_iterations == 0:
            print(f"multiple of {args.log_every_n_iterations} reached. Waiting for next episode of world 0")
            is_waiting_for_new_episode = True

        if iteration % args.save_model_every_n_iterations == 0:
            folder = "checkpoints"
            if not os.path.exists(folder):
                os.makedirs(folder)

            if (args.model_name):
                torch.save(agent.state_dict(), os.path.join(folder,
                                                            f"{args.model_name}_{iteration}.pth"))
                print(f"Model {args.model_name} saved at iteration {iteration}")

            else:
                torch.save(agent.state_dict(),
                           os.path.join(folder, f"{iteration}.pth"))
                print(f"Model saved at iteration {iteration}")

    # Ensure viewer process is terminated
    if viewer_process is not None:
        print(f"Terminating viewer process (PID: {viewer_process.pid})...")

        if viewer_process.poll() is None:
            viewer_process.terminate()
            try:
                viewer_process.wait(timeout=5)
                print("Viewer process terminated successfully")
            except subprocess.TimeoutExpired:
                print("Viewer process didn't terminate gracefully, forcing kill...")
                viewer_process.kill()
                viewer_process.wait()
                print("Viewer process killed")
        else:
            print(
                f"Viewer process already exited with code: {viewer_process.returncode}")

    writer.close()
    print("Cleanup completed")


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.rollout_batch_size = int(args.num_envs * args.num_rollout_steps)
    args.minibatch_size = int(args.rollout_batch_size // args.num_minibatches)
    model_name = args.model_name if args.model_name else f"{args.env_id}__{args.seed}__{int(time.time())}"

    main()
