import torch


class RolloutBuffer:
    def __init__(self, n_steps, n_envs, obs_dim, act_dim, device):
        self.obs = torch.zeros((n_steps, n_envs, obs_dim), device=device)
        self.actions = torch.zeros((n_steps, n_envs, act_dim), device=device)
        self.values = torch.zeros((n_steps, n_envs), device=device)
        self.log_probs = torch.zeros((n_steps, n_envs), device=device)
        self.rewards = torch.zeros((n_steps, n_envs), device=device)
        self.dones = torch.zeros((n_steps, n_envs), device=device)
        self.terminated = torch.zeros((n_steps, n_envs), device=device)

        # Computed after rollout
        self.next_value = torch.zeros((n_envs,), device=device)
        self.next_dones = torch.zeros((n_envs,), device=device)
        self.advantages = torch.zeros((n_steps, n_envs), device=device)
        self.returns = torch.zeros((n_steps, n_envs), device=device)

        self.horizon = n_steps
        self.n_envs = n_envs

    def get_total_steps(self):
        return self.horizon * self.n_envs

    def get_minibatch(self, indices):
        o = self.obs.view(-1, *self.obs.shape[2:])[indices]
        a = self.actions.view(-1, *self.actions.shape[2:])[indices]
        lp = self.log_probs.view(-1)[indices]
        v = self.values.view(-1)[indices]
        adv = self.advantages.view(-1)[indices]
        ret = self.returns.view(-1)[indices]

        return o, a, lp, v, adv, ret