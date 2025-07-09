import torch
import torch.nn as nn

from action import DiscreteActionDistributions


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

    def forward(self, obs):
        x = self.backbone(obs)
        # Create action distributions
        logits = self.actor(x)
        action_dists = DiscreteActionDistributions(self.action_buckets, logits=logits)
        actions, log_probs = action_dists.sample()

        value = self.critic(x)
        return actions, log_probs, value

    def get_value(self, obs):
        x = self.backbone(obs)
        return self.critic(x)

    def get_stats(self, obs, actions):
        x = self.backbone(obs)
        action_dists = DiscreteActionDistributions(self.action_buckets, logits=self.actor(x))
        log_probs, entropies = action_dists.action_stats(actions)
        value = self.critic(x)
        return log_probs, entropies, value

    def load(self, path):
        state_dict = torch.load(path, weights_only=True, map_location="cpu")
        print(state_dict)
        self.load_state_dict(state_dict)