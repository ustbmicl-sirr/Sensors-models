"""
Trainer for MATD3 agent on berth allocation task.
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import time
from typing import Dict, Any

from .evaluator import Evaluator
from .logger import TrainingLogger


class Trainer:
    """
    Trainer for MATD3 multi-agent reinforcement learning.
    """

    def __init__(self, env, agent, replay_buffer, config: dict):
        """
        Initialize trainer.

        Args:
            env: Environment
            agent: MATD3 agent
            replay_buffer: Replay buffer
            config: Configuration dictionary
        """
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.config = config

        # Training config
        self.num_episodes = config['training']['num_episodes']
        self.max_episode_steps = config['training']['max_episode_steps']
        self.batch_size = config['training']['batch_size']
        self.eval_interval = config['training']['eval_interval']
        self.save_interval = config['training']['save_interval']
        self.log_interval = config['training']['log_interval']

        # Paths
        self.save_dir = Path(config['training']['save_dir'])
        self.log_dir = Path(config['training']['log_dir'])

        # Create evaluator
        self.evaluator = Evaluator(env, agent, config)

        # Initialize logger
        experiment_name = config.get('experiment_name', None)
        self.logger = TrainingLogger(
            log_dir=str(self.log_dir),
            experiment_name=experiment_name,
            config=config
        )

        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_results = []

        # Best model tracking
        self.best_metric = -np.inf
        self.best_episode = 0

        # Global step counter for TensorBoard
        self.global_step = 0

    def train(self):
        """Main training loop."""
        self.logger.info("=" * 70)
        self.logger.info("Starting Training")
        self.logger.info("=" * 70)

        start_time = time.time()

        for episode in range(1, self.num_episodes + 1):
            # Run episode
            episode_stats = self._run_episode(episode)

            # Log episode results
            if episode % self.log_interval == 0:
                self._log_episode(episode, episode_stats)

            # Evaluate
            if episode % self.eval_interval == 0:
                eval_metrics = self.evaluator.evaluate()
                self.logger.log_evaluation(episode, eval_metrics)
                self.eval_results.append(eval_metrics)

                # Save best model
                metric = eval_metrics['berth_utilization']
                if metric > self.best_metric:
                    self.best_metric = metric
                    self.best_episode = episode
                    self._save_model(episode, best=True)

            # Save checkpoint
            if episode % self.save_interval == 0:
                self._save_model(episode)

        # Training complete
        elapsed_time = time.time() - start_time
        self.logger.info("=" * 70)
        self.logger.info("Training Complete!")
        self.logger.info("=" * 70)
        self.logger.info(f"Total time: {elapsed_time/3600:.2f} hours")
        self.logger.info(f"Best episode: {self.best_episode}")
        self.logger.info(f"Best metric (berth utilization): {self.best_metric:.4f}")

        # Close logger
        self.logger.close()

    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """
        Run a single training episode.

        Args:
            episode: Episode number

        Returns:
            Episode statistics
        """
        # Reset environment
        observations, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        training_stats = []

        # Episode loop
        for step in range(self.max_episode_steps):
            # Select actions
            actions = self.agent.select_action(observations, explore=True)

            # Execute actions
            next_observations, rewards, terminated, truncated, infos = self.env.step(actions)

            # Get global states
            global_state = self.env.get_global_state()

            # Store transitions
            # Note: We need to handle the case where not all agents are active
            if len(observations) > 0:
                # Pad observations for inactive agents
                full_observations = {}
                full_actions = {}
                full_rewards = {}
                full_next_observations = {}

                for agent_id in range(self.agent.num_agents):
                    if agent_id in observations:
                        full_observations[agent_id] = observations[agent_id]
                        full_actions[agent_id] = actions[agent_id]
                        full_rewards[agent_id] = rewards.get(agent_id, 0.0)
                        full_next_observations[agent_id] = next_observations.get(
                            agent_id, np.zeros(self.agent.obs_dim, dtype=np.float32)
                        )
                    else:
                        # Inactive agent - zero padding
                        full_observations[agent_id] = np.zeros(self.agent.obs_dim, dtype=np.float32)
                        full_actions[agent_id] = np.zeros(self.agent.action_dim, dtype=np.float32)
                        full_rewards[agent_id] = 0.0
                        full_next_observations[agent_id] = np.zeros(self.agent.obs_dim, dtype=np.float32)

                # Add to replay buffer
                self.replay_buffer.add(
                    observations=full_observations,
                    actions=full_actions,
                    rewards=full_rewards,
                    next_observations=full_next_observations,
                    global_state=global_state,
                    next_global_state=self.env.get_global_state(),
                    done=terminated or truncated
                )

            # Update agent
            if len(self.replay_buffer) >= self.batch_size:
                update_stats = self.agent.update(self.replay_buffer, self.batch_size)
                training_stats.append(update_stats)

                # Log training step metrics to TensorBoard
                self.global_step += 1
                if self.global_step % 10 == 0:  # Log every 10 steps
                    self.logger.log_step(self.global_step, {
                        'Loss/critic': update_stats.get('critic_loss', 0),
                        'Loss/actor': update_stats.get('actor_loss', 0),
                        'Q/target_mean': update_stats.get('target_q_mean', 0)
                    })

            # Update state
            observations = next_observations
            episode_reward += sum(rewards.values()) if rewards else 0
            episode_length += 1

            # Check termination
            if terminated or truncated:
                break

        # Aggregate training stats
        if training_stats:
            avg_stats = {
                key: np.mean([s[key] for s in training_stats if key in s])
                for key in training_stats[0].keys()
            }
        else:
            avg_stats = {}

        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'vessels_allocated': len(self.env.allocations),
            **avg_stats
        }

    def _log_episode(self, episode: int, stats: Dict[str, Any]):
        """Log episode statistics."""
        self.episode_rewards.append(stats['episode_reward'])
        self.episode_lengths.append(stats['episode_length'])

        # Log to both text and TensorBoard
        metrics = {
            'Reward/episode': stats['episode_reward'],
            'Episode/length': stats['episode_length'],
            'Episode/vessels_allocated': stats['vessels_allocated']
        }

        if 'critic_loss' in stats:
            metrics['Loss/critic_avg'] = stats['critic_loss']
        if 'actor_loss' in stats:
            metrics['Loss/actor_avg'] = stats['actor_loss']
        if 'target_q_mean' in stats:
            metrics['Q/target_mean_avg'] = stats['target_q_mean']

        self.logger.log_episode(episode, metrics)

    def _save_model(self, episode: int, best: bool = False):
        """
        Save model checkpoint.

        Args:
            episode: Episode number
            best: Whether this is the best model
        """
        if best:
            filepath = self.save_dir / 'best_model.pth'
            self.logger.info(f"💾 Saving best model to {filepath}")
        else:
            filepath = self.save_dir / f'model_episode_{episode}.pth'

        self.agent.save(str(filepath))
        self.logger.log_checkpoint(episode, str(filepath))

        # Save training statistics
        stats_path = self.save_dir / 'training_stats.npz'
        np.savez(
            stats_path,
            episode_rewards=self.episode_rewards,
            episode_lengths=self.episode_lengths,
            eval_results=self.eval_results
        )
