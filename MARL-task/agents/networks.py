"""
Neural network architectures for multi-agent reinforcement learning.

Implements:
- Actor: Policy network (local observation -> action)
- Critic: Q-value network (global state + all actions -> Q-value)
- AttentionCritic: Attention-based critic for MAAC baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class Actor(nn.Module):
    """
    Actor network for continuous action space.

    Maps local observation to 3D action: [position, wait_time, shore_power_prob]
    Uses separate output heads for each action dimension.
    """

    def __init__(self, obs_dim: int, action_dim: int = 3,
                 hidden_dims: Tuple[int, ...] = (256, 256)):
        """
        Initialize Actor network.

        Args:
            obs_dim: Observation dimension (17 for local features)
            action_dim: Action dimension (3: position, time, probability)
            hidden_dims: Tuple of hidden layer sizes
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared backbone
        layers = []
        in_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Separate output heads for each action dimension
        # Position head: outputs value in [0, 1] (will be scaled by berth_length)
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )

        # Time head: outputs non-negative value (will be scaled by max_wait_time)
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()  # Changed from Softplus for better stability
        )

        # Shore power probability head: outputs in [0, 1]
        self.shore_power_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Sigmoid()
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            obs: Observation tensor (batch, obs_dim)

        Returns:
            action: Action tensor (batch, 3) in [0, 1] for all dimensions
        """
        # Extract features
        features = self.backbone(obs)

        # Compute action components
        position = self.position_head(features)      # [0, 1]
        wait_time = self.time_head(features)         # [0, 1]
        shore_power_prob = self.shore_power_head(features)  # [0, 1]

        # Concatenate
        action = torch.cat([position, wait_time, shore_power_prob], dim=-1)

        return action


class Critic(nn.Module):
    """
    Critic network for centralized value estimation (CTDE).

    Takes global state and all agents' actions, outputs Q-value.
    """

    def __init__(self, global_state_dim: int, num_agents: int, action_dim: int = 3,
                 hidden_dims: Tuple[int, ...] = (512, 512, 256)):
        """
        Initialize Critic network.

        Args:
            global_state_dim: Global state dimension (num_agents * obs_dim)
            num_agents: Number of agents
            action_dim: Action dimension per agent
            hidden_dims: Tuple of hidden layer sizes
        """
        super().__init__()

        self.global_state_dim = global_state_dim
        self.num_agents = num_agents
        self.action_dim = action_dim

        # Input dimension: global state + all actions
        input_dim = global_state_dim + num_agents * action_dim

        # Build network
        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, global_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            global_state: Global state tensor (batch, global_state_dim)
            actions: All agents' actions (batch, num_agents * action_dim)

        Returns:
            q_value: Q-value tensor (batch, 1)
        """
        # Concatenate state and actions
        x = torch.cat([global_state, actions], dim=-1)

        # Compute Q-value
        q_value = self.network(x)

        return q_value


class AttentionCritic(nn.Module):
    """
    Attention-based Critic for MAAC baseline.

    Uses multi-head attention to aggregate information from other agents.
    """

    def __init__(self, obs_dim: int, action_dim: int, num_agents: int,
                 hidden_dim: int = 256, num_heads: int = 4):
        """
        Initialize Attention Critic.

        Args:
            obs_dim: Observation dimension per agent
            action_dim: Action dimension per agent
            num_agents: Number of agents
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Encoders
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            batch_first=True
        )

        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor,
                agent_id: int) -> torch.Tensor:
        """
        Forward pass with attention.

        Args:
            observations: All agents' observations (batch, num_agents, obs_dim)
            actions: All agents' actions (batch, num_agents, action_dim)
            agent_id: Current agent ID (for query)

        Returns:
            q_value: Q-value tensor (batch, 1)
        """
        batch_size = observations.shape[0]

        # Encode observations and actions
        obs_encoded = self.obs_encoder(observations)  # (batch, num_agents, hidden)
        action_encoded = self.action_encoder(actions)  # (batch, num_agents, hidden)

        # Concatenate encodings
        features = torch.cat([obs_encoded, action_encoded], dim=-1)
        # (batch, num_agents, hidden*2)

        # Use current agent as query, all agents as key/value
        query = features[:, agent_id:agent_id+1, :]  # (batch, 1, hidden*2)

        # Multi-head attention
        attended, _ = self.attention(query, features, features)
        # (batch, 1, hidden*2)

        # Compute Q-value
        q_value = self.q_head(attended.squeeze(1))  # (batch, 1)

        return q_value


def test_networks():
    """Test network architectures."""
    print("Testing network architectures...")

    # Parameters
    obs_dim = 17
    action_dim = 3
    num_agents = 5
    global_state_dim = num_agents * obs_dim
    batch_size = 32

    # Test Actor
    print("\n1. Testing Actor...")
    actor = Actor(obs_dim, action_dim)
    obs = torch.randn(batch_size, obs_dim)
    action = actor(obs)
    assert action.shape == (batch_size, action_dim)
    assert torch.all((action >= 0) & (action <= 1))
    print(f"   Actor output shape: {action.shape}")
    print(f"   Action range: [{action.min():.3f}, {action.max():.3f}]")

    # Test Critic
    print("\n2. Testing Critic...")
    critic = Critic(global_state_dim, num_agents, action_dim)
    global_state = torch.randn(batch_size, global_state_dim)
    all_actions = torch.randn(batch_size, num_agents * action_dim)
    q_value = critic(global_state, all_actions)
    assert q_value.shape == (batch_size, 1)
    print(f"   Critic output shape: {q_value.shape}")
    print(f"   Q-value range: [{q_value.min():.3f}, {q_value.max():.3f}]")

    # Test AttentionCritic
    print("\n3. Testing AttentionCritic...")
    attn_critic = AttentionCritic(obs_dim, action_dim, num_agents)
    observations = torch.randn(batch_size, num_agents, obs_dim)
    actions = torch.randn(batch_size, num_agents, action_dim)
    agent_id = 0
    q_value = attn_critic(observations, actions, agent_id)
    assert q_value.shape == (batch_size, 1)
    print(f"   AttentionCritic output shape: {q_value.shape}")
    print(f"   Q-value range: [{q_value.min():.3f}, {q_value.max():.3f}]")

    print("\n✓ All network tests passed!")


if __name__ == '__main__':
    test_networks()
