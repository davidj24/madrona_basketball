import torch
import torch.nn as nn

from action import DiscreteActionDistributions

class RunningMeanStd(torch.nn.Module):
    def __init__(self, dim: int, clamp: float=0):
        super().__init__()
        self.epsilon = 1e-5
        self.clamp = clamp
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def forward(self, x, unnorm=False):
        mean = self.mean.to(torch.float32)
        var = self.var.to(torch.float32)+self.epsilon
        if unnorm:
            if self.clamp:
                x = torch.clamp(x, min=-self.clamp, max=self.clamp)
            return mean + torch.sqrt(var) * x
        x = (x - mean) * torch.rsqrt(var)
        if self.clamp:
            return torch.clamp(x, min=-self.clamp, max=self.clamp)
        return x

    @torch.no_grad()
    def update(self, x):
        x = x.view(-1, x.size(-1))
        var, mean = torch.var_mean(x, dim=0, unbiased=True)
        count = x.size(0)
        count_ = count + self.count
        delta = mean - self.mean
        m = self.var * self.count + var * count + delta**2 * self.count * count / count_
        self.mean.copy_(self.mean+delta*count/count_)
        self.var.copy_(m / count_)
        self.count.copy_(count_)


class DiagonalPopArt(torch.nn.Module):
    def __init__(self, dim: int, weight: torch.Tensor, bias: torch.Tensor, momentum:float=0.1):
        super().__init__()
        self.epsilon = 1e-5

        self.momentum = momentum
        self.register_buffer("m", torch.zeros((dim,), dtype=torch.float64))
        self.register_buffer("v", torch.full((dim,), self.epsilon, dtype=torch.float64))
        self.register_buffer("debias", torch.zeros(1, dtype=torch.float64))

        self.weight = weight
        self.bias = bias

    def forward(self, x, unnorm=False):
        debias = self.debias.clip(min=self.epsilon)
        mean = self.m/debias
        var = (self.v - self.m.square()).div_(debias)
        if unnorm:
            std = torch.sqrt(var)
            return (mean + std * x).to(x.dtype)
        x = ((x - mean) * torch.rsqrt(var)).to(x.dtype)
        return x

    @torch.no_grad()
    def update(self, x):
        x = x.view(-1, x.size(-1))
        running_m = torch.mean(x, dim=0)
        running_v = torch.mean(x.square(), dim=0)
        new_m = self.m.mul(1-self.momentum).add_(running_m, alpha=self.momentum)
        new_v = self.v.mul(1-self.momentum).add_(running_v, alpha=self.momentum)
        std = (self.v - self.m.square()).sqrt_()
        new_std_inv = (new_v - new_m.square()).rsqrt_()

        scale = std.mul_(new_std_inv)
        shift = (self.m - new_m).mul_(new_std_inv)

        self.bias.data.mul_(scale).add_(shift)
        self.weight.data.mul_(scale.unsqueeze_(-1))

        self.debias.data.mul_(1-self.momentum).add_(1.0*self.momentum)
        self.m.data.copy_(new_m)
        self.v.data.copy_(new_v)


def backbone_layer_init(layer):
    torch.nn.init.kaiming_normal_(layer.weight, nn.init.calculate_gain("relu"))
    torch.nn.init.constant_(layer.bias, val=0)
    return layer

def head_layer_init(layer):
    torch.nn.init.orthogonal_(layer.weight, gain=0.01)
    torch.nn.init.constant_(layer.bias, 0)
    return layer


class Agent(nn.Module):
    def __init__(self, input_dim, num_channels, num_layers, action_buckets):
        super().__init__()
        self.action_dim = sum(action_buckets)
        self.action_buckets = action_buckets

        # Create a backbone MLP
        layers = [
            backbone_layer_init(nn.Linear(input_dim, num_channels)),
            nn.LayerNorm(num_channels),
            nn.ReLU(),
        ]
        for i in range(num_layers - 1):
            layers.append(backbone_layer_init(nn.Linear(num_channels, num_channels)))
            layers.append(nn.LayerNorm(num_channels))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)

        # Create actor and critic heads
        self.actor = head_layer_init(nn.Linear(num_channels, self.action_dim))
        self.critic = head_layer_init(nn.Linear(num_channels, 1))

        # Observation and value normalization
        self.ob_normalizer = RunningMeanStd(input_dim, clamp=5.0)
        self.value_normalizer = DiagonalPopArt(1, self.critic.weight, self.critic.bias)

    def norm_obs_backbone(self, obs):
        # normalize observations and run through backbone MLP
        obs_norm = self.ob_normalizer(obs)
        x = self.backbone(obs_norm)
        return x

    def forward(self, obs):
        x = self.norm_obs_backbone(obs)

        # Create action distributions
        logits = self.actor(x)
        action_dists = DiscreteActionDistributions(self.action_buckets, logits=logits)
        actions, log_probs = action_dists.sample()
        # Value function
        value = self.critic(x)
        return actions, log_probs, value

    def unnorm_obs(self, obs):
        return self.ob_normalizer(obs, True)

    def update_obs_normalizer(self, obs_raw):
        return self.ob_normalizer.update(obs_raw)

    def update_value_normalizer(self, returns):
        return self.value_normalizer.update(returns)

    def normalize_value(self, returns):
        return self.value_normalizer(returns)

    def unnorm_value(self, values):
        return self.value_normalizer(values, unnorm=True)

    def get_value(self, obs):
        x = self.norm_obs_backbone(obs)
        return self.critic(x)

    def get_stats(self, obs, actions):
        x = self.norm_obs_backbone(obs)

        action_dists = DiscreteActionDistributions(self.action_buckets, logits=self.actor(x))
        log_probs, entropies = action_dists.action_stats(actions)
        value = self.critic(x)
        return log_probs, entropies, value

    def load(self, path):
        state_dict = torch.load(path, weights_only=True, map_location="cpu")
        self.load_state_dict(state_dict)
