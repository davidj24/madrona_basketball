import wandb
from torch.utils.tensorboard import SummaryWriter

from ppo_stats import PPOStats, PPOTimer


class WandbLogger:
    def __init__(self,
                 project: str,
                 entity: str,
                 run_name: str):
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            sync_tensorboard=True,
        )
        self.writer = SummaryWriter(f"runs/{run_name}")

    def log(self, stats: PPOStats, timer: PPOTimer):
        step = timer.global_step
        self.writer.add_scalar("charts/mean_rewards", stats.mean_reward.get_mean(), step)
        self.writer.add_scalar("charts/episode_lengths", stats.mean_episode_length.get_mean(), step)
        # Performance
        self.writer.add_scalar("performance/FPS", timer.get_iter_fps(), step)
        # Losses
        self.writer.add_scalar("losses/actor_loss", stats.a_loss, step)
        self.writer.add_scalar("losses/critic_loss", stats.c_loss, step)
        self.writer.add_scalar("losses/entropy_loss", stats.e_loss, step)
        self.writer.add_scalar("losses/bounds_loss", stats.b_loss, step)

    def close(self):
        self.writer.close()
        wandb.finish()
