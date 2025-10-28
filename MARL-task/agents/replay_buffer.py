"""
Experience replay buffer for multi-agent reinforcement learning.
"""

import numpy as np
import torch
from collections import deque
from typing import Dict, List, Tuple, Optional
import random


class ReplayBuffer:
    """
    Experience replay buffer for MARL.

    Stores transitions: (observations, actions, rewards, next_observations, dones)
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int,
                 num_agents: int, device: str = 'cpu'):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
            obs_dim: Observation dimension per agent
            action_dim: Action dimension per agent
            num_agents: Number of agents
            device: Device for tensors
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = device

        # Storage
        self.observations = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, num_agents), dtype=np.float32)
        self.next_observations = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        # Global states (for centralized critic)
        self.global_states = np.zeros((capacity, num_agents * obs_dim), dtype=np.float32)
        self.next_global_states = np.zeros((capacity, num_agents * obs_dim), dtype=np.float32)

        # Pointers
        self.position = 0
        self.size = 0

    def add(self, observations: Dict[int, np.ndarray],
            actions: Dict[int, np.ndarray],
            rewards: Dict[int, float],
            next_observations: Dict[int, np.ndarray],
            global_state: np.ndarray,
            next_global_state: np.ndarray,
            done: bool):
        """
        Add a transition to the buffer.

        Args:
            observations: Dict of observations {agent_id: obs}
            actions: Dict of actions {agent_id: action}
            rewards: Dict of rewards {agent_id: reward}
            next_observations: Dict of next observations
            global_state: Global state vector
            next_global_state: Next global state vector
            done: Episode done flag
        """
        # Convert dicts to arrays (assumes agent IDs are 0, 1, 2, ...)
        obs_array = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        action_array = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
        reward_array = np.zeros(self.num_agents, dtype=np.float32)
        next_obs_array = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)

        for agent_id in observations.keys():
            obs_array[agent_id] = observations[agent_id]
            action_array[agent_id] = actions[agent_id]
            reward_array[agent_id] = rewards[agent_id]
            next_obs_array[agent_id] = next_observations[agent_id]

        # Store
        self.observations[self.position] = obs_array
        self.actions[self.position] = action_array
        self.rewards[self.position] = reward_array
        self.next_observations[self.position] = next_obs_array
        self.global_states[self.position] = global_state
        self.next_global_states[self.position] = next_global_state
        self.dones[self.position] = float(done)

        # Update pointers
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Batch size

        Returns:
            Dictionary of tensors
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        batch = {
            'observations': torch.FloatTensor(self.observations[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_observations': torch.FloatTensor(self.next_observations[indices]).to(self.device),
            'global_state': torch.FloatTensor(self.global_states[indices]).to(self.device),
            'next_global_state': torch.FloatTensor(self.next_global_states[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).unsqueeze(-1).to(self.device)
        }

        return batch

    def __len__(self):
        """Return current buffer size."""
        return self.size

    def clear(self):
        """Clear the buffer."""
        self.position = 0
        self.size = 0


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized experience replay buffer.

    Uses TD-error for prioritization (optional enhancement).
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int,
                 num_agents: int, alpha: float = 0.6, device: str = 'cpu'):
        """
        Initialize prioritized replay buffer.

        Args:
            capacity: Maximum buffer size
            obs_dim: Observation dimension
            action_dim: Action dimension
            num_agents: Number of agents
            alpha: Prioritization exponent
            device: Device
        """
        super().__init__(capacity, obs_dim, action_dim, num_agents, device)

        self.alpha = alpha
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0

    def add(self, observations: Dict, actions: Dict, rewards: Dict,
            next_observations: Dict, global_state: np.ndarray,
            next_global_state: np.ndarray, done: bool):
        """Add with maximum priority."""
        super().add(observations, actions, rewards, next_observations,
                   global_state, next_global_state, done)

        # Assign max priority to new experience
        self.priorities[self.position - 1] = self.max_priority

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Sample with priorities.

        Args:
            batch_size: Batch size
            beta: Importance sampling exponent

        Returns:
            batch: Dictionary of tensors
            indices: Sampled indices
            weights: Importance sampling weights
        """
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / priorities.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)

        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        # Get batch
        batch = {
            'observations': torch.FloatTensor(self.observations[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_observations': torch.FloatTensor(self.next_observations[indices]).to(self.device),
            'global_state': torch.FloatTensor(self.global_states[indices]).to(self.device),
            'next_global_state': torch.FloatTensor(self.next_global_states[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).unsqueeze(-1).to(self.device)
        }

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled experiences.

        Args:
            indices: Indices of experiences
            priorities: New priorities (TD-errors)
        """
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())


def test_replay_buffer():
    """Test replay buffer."""
    print("Testing ReplayBuffer...")

    # Parameters
    capacity = 1000
    obs_dim = 17
    action_dim = 3
    num_agents = 5
    batch_size = 32

    # Create buffer
    buffer = ReplayBuffer(capacity, obs_dim, action_dim, num_agents)

    # Add transitions
    for i in range(100):
        observations = {j: np.random.randn(obs_dim).astype(np.float32)
                       for j in range(num_agents)}
        actions = {j: np.random.randn(action_dim).astype(np.float32)
                  for j in range(num_agents)}
        rewards = {j: np.random.randn() for j in range(num_agents)}
        next_observations = {j: np.random.randn(obs_dim).astype(np.float32)
                            for j in range(num_agents)}
        global_state = np.random.randn(num_agents * obs_dim).astype(np.float32)
        next_global_state = np.random.randn(num_agents * obs_dim).astype(np.float32)
        done = i == 99

        buffer.add(observations, actions, rewards, next_observations,
                  global_state, next_global_state, done)

    print(f"Buffer size: {len(buffer)}")

    # Sample batch
    batch = buffer.sample(batch_size)

    print(f"Batch observations shape: {batch['observations'].shape}")
    print(f"Batch actions shape: {batch['actions'].shape}")
    print(f"Batch rewards shape: {batch['rewards'].shape}")
    print(f"Batch global_state shape: {batch['global_state'].shape}")

    assert batch['observations'].shape == (batch_size, num_agents, obs_dim)
    assert batch['actions'].shape == (batch_size, num_agents, action_dim)
    assert batch['rewards'].shape == (batch_size, num_agents)

    print("✓ ReplayBuffer test passed!")

    # Test PrioritizedReplayBuffer
    print("\nTesting PrioritizedReplayBuffer...")
    per_buffer = PrioritizedReplayBuffer(capacity, obs_dim, action_dim, num_agents)

    for i in range(100):
        observations = {j: np.random.randn(obs_dim).astype(np.float32)
                       for j in range(num_agents)}
        actions = {j: np.random.randn(action_dim).astype(np.float32)
                  for j in range(num_agents)}
        rewards = {j: np.random.randn() for j in range(num_agents)}
        next_observations = {j: np.random.randn(obs_dim).astype(np.float32)
                            for j in range(num_agents)}
        global_state = np.random.randn(num_agents * obs_dim).astype(np.float32)
        next_global_state = np.random.randn(num_agents * obs_dim).astype(np.float32)
        done = i == 99

        per_buffer.add(observations, actions, rewards, next_observations,
                      global_state, next_global_state, done)

    batch, indices, weights = per_buffer.sample(batch_size)
    print(f"Sampled indices: {indices[:5]}")
    print(f"IS weights: {weights[:5]}")

    print("✓ PrioritizedReplayBuffer test passed!")


if __name__ == '__main__':
    test_replay_buffer()
