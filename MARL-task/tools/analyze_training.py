"""
Analyze Training Logs.

This script analyzes training logs and provides statistics.
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import sys


def parse_log_file(log_file: Path):
    """Parse log file and extract metrics."""
    episodes = []
    losses = defaultdict(list)
    rewards = []

    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return None

    with open(log_file, 'r') as f:
        for line in f:
            # Parse episode metrics
            if "Episode" in line and "Reward/episode" in line:
                try:
                    # Extract episode number
                    ep_match = re.search(r'Episode (\d+)', line)
                    if ep_match:
                        episode = int(ep_match.group(1))

                        # Extract reward
                        reward_match = re.search(r'Reward/episode[:\s]+([\d\.\-]+)', line)
                        if reward_match:
                            reward = float(reward_match.group(1))
                            episodes.append(episode)
                            rewards.append(reward)
                except Exception as e:
                    continue

            # Parse loss metrics
            if "Loss/critic" in line:
                try:
                    loss_match = re.search(r'Loss/critic[:\s]+([\d\.\-]+)', line)
                    if loss_match:
                        losses['critic'].append(float(loss_match.group(1)))
                except Exception:
                    continue

    return {
        'episodes': episodes,
        'rewards': rewards,
        'losses': dict(losses)
    }


def print_statistics(data: dict):
    """Print training statistics."""
    if not data or not data['rewards']:
        print("No training data found in log file")
        return

    rewards = data['rewards']
    episodes = data['episodes']

    print("=" * 70)
    print("Training Statistics")
    print("=" * 70)
    print(f"Total Episodes: {len(episodes)}")
    print(f"Episode Range: {min(episodes)} - {max(episodes)}")
    print()

    print("Rewards:")
    print(f"  Mean: {sum(rewards)/len(rewards):.2f}")
    print(f"  Min: {min(rewards):.2f}")
    print(f"  Max: {max(rewards):.2f}")
    print(f"  Latest: {rewards[-1]:.2f}")
    print()

    # Improvement trend
    if len(rewards) >= 10:
        early_avg = sum(rewards[:10]) / 10
        late_avg = sum(rewards[-10:]) / 10
        improvement = ((late_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0
        print(f"Improvement (first 10 vs last 10): {improvement:+.1f}%")
        print()

    # Loss statistics
    if data['losses']:
        print("Losses:")
        for loss_name, loss_values in data['losses'].items():
            if loss_values:
                print(f"  {loss_name}:")
                print(f"    Mean: {sum(loss_values)/len(loss_values):.4f}")
                print(f"    Latest: {loss_values[-1]:.4f}")


def export_to_csv(data: dict, output_file: Path):
    """Export data to CSV file."""
    import csv

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward'])

        for episode, reward in zip(data['episodes'], data['rewards']):
            writer.writerow([episode, reward])

    print(f"\n✓ Data exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze training logs')
    parser.add_argument(
        '--log-file',
        type=str,
        default='logs/training/text/training.log',
        help='Path to training log file'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export data to CSV file'
    )

    args = parser.parse_args()

    log_file = Path(args.log_file)

    print(f"Analyzing: {log_file}")
    print()

    data = parse_log_file(log_file)

    if data:
        print_statistics(data)

        if args.export:
            export_to_csv(data, Path(args.export))


if __name__ == '__main__':
    main()
