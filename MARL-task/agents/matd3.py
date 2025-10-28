"""
MATD3: Multi-Agent Twin Delayed Deep Deterministic Policy Gradient

Implementation of MATD3 algorithm with:
- Centralized Training, Decentralized Execution (CTDE)
- Double Q-learning with twin critics
- Delayed policy updates
- Target policy smoothing
- Dimension-aware exploration noise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy

from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer


class MATD3Agent:
    """
    MATD3 agent for multi-agent berth allocation.

    Key features:
    - Each agent has its own Actor (policy)
    - Shared twin Critics (centralized value estimation)
    - Delayed policy updates (policy_delay parameter)
    - Target policy smoothing with noise
    """

    def __init__(self, config: dict):
        """
        Initialize MATD3 agent.

        Args:
            config: Configuration dictionary with:
                - obs_dim: Observation dimension (17)
                - action_dim: Action dimension (3)
                - num_agents: Number of agents
                - actor_lr: Actor learning rate
                - critic_lr: Critic learning rate
                - gamma: Discount factor
                - tau: Soft update coefficient
                - policy_delay: Policy update delay
                - policy_noise: Target policy smoothing noise
                - noise_clip: Noise clipping value
                - exploration_noise: Exploration noise parameters
                - berth_length: Berth length (for action scaling)
                - max_wait_time: Max waiting time (for action scaling)
                - device: Device (cpu/cuda)
        """
        self.config = config
        self.obs_dim = config['obs_dim']
        self.action_dim = config.get('action_dim', 3)
        self.num_agents = config['num_agents']
        self.device = config.get('device', 'cpu')

        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.tau = config.get('tau', 0.005)
        self.policy_delay = config.get('policy_delay', 2)
        self.policy_noise = config.get('policy_noise', 0.2)
        self.noise_clip = config.get('noise_clip', 0.5)
        self.grad_clip = config.get('grad_clip', 1.0)

        # Action scaling
        self.berth_length = config.get('berth_length', 2000)
        self.max_wait_time = config.get('max_wait_time', 48.0)

        # Exploration noise
        self.exploration_noise = config.get('exploration_noise', {
            'position': 0.1,      # Std for position (normalized)
            'time': 0.1,          # Std for time (normalized)
            'probability': 0.1    # Std for probability
        })

        # Build networks
        self._build_networks(config)

        # Training step counter
        self.train_step = 0

    def _build_networks(self, config):
        """Build actor and critic networks."""
        # Create Actors for each agent
        self.actors = []
        self.actors_target = []
        self.actor_optimizers = []

        actor_hidden = config.get('actor_hidden', (256, 256))

        for i in range(self.num_agents):
            actor = Actor(self.obs_dim, self.action_dim, hidden_dims=actor_hidden).to(self.device)
            actor_target = Actor(self.obs_dim, self.action_dim, hidden_dims=actor_hidden).to(self.device)
            actor_target.load_state_dict(actor.state_dict())

            self.actors.append(actor)
            self.actors_target.append(actor_target)
            self.actor_optimizers.append(
                optim.Adam(actor.parameters(), lr=config.get('actor_lr', 1e-4))
            )

        # Twin Critics (shared across agents)
        global_state_dim = self.num_agents * self.obs_dim
        critic_hidden = config.get('critic_hidden', (512, 512, 256))

        self.critic_1 = Critic(
            global_state_dim, self.num_agents, self.action_dim,
            hidden_dims=critic_hidden
        ).to(self.device)

        self.critic_2 = Critic(
            global_state_dim, self.num_agents, self.action_dim,
            hidden_dims=critic_hidden
        ).to(self.device)

        self.critic_1_target = Critic(
            global_state_dim, self.num_agents, self.action_dim,
            hidden_dims=critic_hidden
        ).to(self.device)

        self.critic_2_target = Critic(
            global_state_dim, self.num_agents, self.action_dim,
            hidden_dims=critic_hidden
        ).to(self.device)

        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Critic optimizer
        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=config.get('critic_lr', 1e-3)
        )

    def select_action(self, observations: Dict[int, np.ndarray],
                     explore: bool = True) -> Dict[int, np.ndarray]:
        """
        Select actions for all agents.

        Args:
            observations: Dict mapping agent_id to observation
            explore: Whether to add exploration noise

        Returns:
            Dict mapping agent_id to action
        """
        actions = {}

        for agent_id, obs in observations.items():
            # Convert to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            # Get action from actor
            with torch.no_grad():
                action = self.actors[agent_id](obs_tensor).squeeze(0).cpu().numpy()

            # Add exploration noise
            if explore:
                action = self._add_exploration_noise(action)

            actions[agent_id] = action

        return actions

    def _add_exploration_noise(self, action: np.ndarray) -> np.ndarray:
        """
        Add dimension-aware exploration noise.

        Based on paper design:
        - Position: Gaussian noise
        - Time: Gaussian noise (changed from exponential for stability)
        - Probability: Uniform noise

        Args:
            action: Action array [position, time, probability] (all in [0, 1])

        Returns:
            Noisy action
        """
        noisy_action = action.copy()

        # Position noise (Gaussian)
        position_noise = np.random.normal(0, self.exploration_noise['position'])
        noisy_action[0] = np.clip(noisy_action[0] + position_noise, 0, 1)

        # Time noise (Gaussian)
        time_noise = np.random.normal(0, self.exploration_noise['time'])
        noisy_action[1] = np.clip(noisy_action[1] + time_noise, 0, 1)

        # Probability noise (Uniform)
        prob_noise = np.random.uniform(
            -self.exploration_noise['probability'],
            self.exploration_noise['probability']
        )
        noisy_action[2] = np.clip(noisy_action[2] + prob_noise, 0, 1)

        return noisy_action

    def update(self, replay_buffer: ReplayBuffer, batch_size: int) -> Dict[str, float]:
        """
        Update networks using a batch from replay buffer.

        Args:
            replay_buffer: Replay buffer
            batch_size: Batch size

        Returns:
            Dictionary of training statistics
        """
        # Sample batch
        batch = replay_buffer.sample(batch_size)

        observations = batch['observations']           # (batch, num_agents, obs_dim)
        actions = batch['actions']                     # (batch, num_agents, action_dim)
        rewards = batch['rewards']                     # (batch, num_agents)
        next_observations = batch['next_observations'] # (batch, num_agents, obs_dim)
        global_state = batch['global_state']           # (batch, global_state_dim)
        next_global_state = batch['next_global_state'] # (batch, global_state_dim)
        dones = batch['dones']                         # (batch, 1)

        # Update critics
        critic_loss, target_q_mean = self._update_critics(
            observations, actions, rewards, next_observations,
            global_state, next_global_state, dones
        )

        # Delayed policy update
        actor_loss = None
        if self.train_step % self.policy_delay == 0:
            actor_loss = self._update_actors(observations, global_state)
            self._soft_update_targets()

        self.train_step += 1

        # Return stats
        stats = {
            'critic_loss': critic_loss,
            'target_q_mean': target_q_mean,
        }
        if actor_loss is not None:
            stats['actor_loss'] = actor_loss

        return stats

    def _update_critics(self, observations, actions, rewards, next_observations,
                       global_state, next_global_state, dones):
        """Update twin critics."""
        batch_size = observations.shape[0]

        with torch.no_grad():
            # Compute target actions from target actors
            next_actions_list = []
            for i in range(self.num_agents):
                next_action = self.actors_target[i](next_observations[:, i, :])

                # Target policy smoothing: add clipped noise
                noise = torch.randn_like(next_action) * self.policy_noise
                noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
                next_action = torch.clamp(next_action + noise, 0, 1)

                next_actions_list.append(next_action)

            # Concatenate all next actions
            next_actions = torch.cat(next_actions_list, dim=-1)
            # (batch, num_agents * action_dim)

            # Compute target Q-values (min of twin critics)
            target_q1 = self.critic_1_target(next_global_state, next_actions)
            target_q2 = self.critic_2_target(next_global_state, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # Compute target values
            # Use mean reward across agents (can be modified)
            mean_reward = rewards.mean(dim=1, keepdim=True)
            target_value = mean_reward + (1 - dones) * self.gamma * target_q

        # Current Q-values
        current_actions = actions.reshape(batch_size, -1)  # Flatten actions
        current_q1 = self.critic_1(global_state, current_actions)
        current_q2 = self.critic_2(global_state, current_actions)

        # Critic loss (MSE)
        critic_loss = F.mse_loss(current_q1, target_value) + \
                     F.mse_loss(current_q2, target_value)

        # Optimize critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            self.grad_clip
        )
        self.critic_optimizer.step()

        return critic_loss.item(), target_q.mean().item()

    def _update_actors(self, observations, global_state):
        """Update actor policies."""
        total_actor_loss = 0.0

        # Update each actor
        for i in range(self.num_agents):
            # Compute actions for all agents
            actions_list = []
            for j in range(self.num_agents):
                if j == i:
                    # Use current actor for agent i
                    action = self.actors[j](observations[:, j, :])
                else:
                    # Use frozen actors for other agents
                    with torch.no_grad():
                        action = self.actors[j](observations[:, j, :])
                actions_list.append(action)

            # Concatenate all actions
            all_actions = torch.cat(actions_list, dim=-1)

            # Actor loss = -Q(s, a)
            actor_loss = -self.critic_1(global_state, all_actions).mean()

            # Optimize actor
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actors[i].parameters(),
                self.grad_clip
            )
            self.actor_optimizers[i].step()

            total_actor_loss += actor_loss.item()

        return total_actor_loss / self.num_agents

    def _soft_update_targets(self):
        """Soft update of target networks."""
        # Update critics
        for param, target_param in zip(self.critic_1.parameters(),
                                      self.critic_1_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(self.critic_2.parameters(),
                                      self.critic_2_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        # Update actors
        for i in range(self.num_agents):
            for param, target_param in zip(self.actors[i].parameters(),
                                          self.actors_target[i].parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

    def save(self, filepath: str):
        """
        Save agent to file.

        Args:
            filepath: Save path
        """
        checkpoint = {
            'config': self.config,
            'train_step': self.train_step,
            'actors': [actor.state_dict() for actor in self.actors],
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizer': self.critic_optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str):
        """
        Load agent from file.

        Args:
            filepath: Load path
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.train_step = checkpoint['train_step']

        # Load actors
        for i, actor_state in enumerate(checkpoint['actors']):
            self.actors[i].load_state_dict(actor_state)
            self.actors_target[i].load_state_dict(actor_state)

        # Load critics
        self.critic_1.load_state_dict(checkpoint['critic_1'])
        self.critic_2.load_state_dict(checkpoint['critic_2'])
        self.critic_1_target.load_state_dict(checkpoint['critic_1'])
        self.critic_2_target.load_state_dict(checkpoint['critic_2'])

        # Load optimizers
        for i, opt_state in enumerate(checkpoint['actor_optimizers']):
            self.actor_optimizers[i].load_state_dict(opt_state)
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    def set_eval_mode(self):
        """Set networks to evaluation mode."""
        for actor in self.actors:
            actor.eval()
        self.critic_1.eval()
        self.critic_2.eval()

    def set_train_mode(self):
        """Set networks to training mode."""
        for actor in self.actors:
            actor.train()
        self.critic_1.train()
        self.critic_2.train()


def test_matd3():
    """Test MATD3 agent."""
    print("Testing MATD3 agent...")

    # Configuration
    config = {
        'obs_dim': 17,
        'action_dim': 3,
        'num_agents': 5,
        'actor_lr': 1e-4,
        'critic_lr': 1e-3,
        'gamma': 0.99,
        'tau': 0.005,
        'policy_delay': 2,
        'policy_noise': 0.2,
        'noise_clip': 0.5,
        'berth_length': 2000,
        'max_wait_time': 48.0,
        'device': 'cpu'
    }

    # Create agent
    agent = MATD3Agent(config)

    # Test action selection
    observations = {i: np.random.randn(17).astype(np.float32)
                   for i in range(config['num_agents'])}

    actions = agent.select_action(observations, explore=True)
    print(f"Actions shape: {list(actions.values())[0].shape}")
    print(f"Action example: {actions[0]}")

    # Test replay buffer
    from .replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(
        capacity=1000,
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        num_agents=config['num_agents']
    )

    # Add some transitions
    for _ in range(100):
        obs = {i: np.random.randn(17).astype(np.float32)
               for i in range(config['num_agents'])}
        acts = {i: np.random.rand(3).astype(np.float32)
                for i in range(config['num_agents'])}
        rews = {i: np.random.randn() for i in range(config['num_agents'])}
        next_obs = {i: np.random.randn(17).astype(np.float32)
                    for i in range(config['num_agents'])}
        global_state = np.random.randn(85).astype(np.float32)
        next_global_state = np.random.randn(85).astype(np.float32)

        buffer.add(obs, acts, rews, next_obs, global_state, next_global_state, False)

    # Test update
    stats = agent.update(buffer, batch_size=32)
    print(f"Training stats: {stats}")

    print("✓ MATD3 agent test passed!")


if __name__ == '__main__':
    test_matd3()
