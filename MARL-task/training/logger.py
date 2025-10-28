"""
Training Logger with TensorBoard and Text Logging Support.

This module provides comprehensive logging for training and testing,
combining TensorBoard for metrics visualization and Python logging for text events.
"""

import logging
import logging.config
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


class TrainingLogger:
    """
    Comprehensive training logger with TensorBoard and text logging.

    Features:
    - TensorBoard metrics visualization
    - Structured text logging
    - Automatic log directory management
    - Episode and step-level logging
    """

    def __init__(
        self,
        log_dir: str = "logs/training",
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize training logger.

        Args:
            log_dir: Base directory for logs
            experiment_name: Name of experiment (default: timestamp)
            config: Training configuration to log
        """
        self.log_dir = Path(log_dir)

        # Generate experiment name if not provided
        if experiment_name is None:
            experiment_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        self.experiment_name = experiment_name

        # Create directory structure
        self.text_log_dir = self.log_dir / "text"
        self.tb_log_dir = self.log_dir / "tensorboard" / experiment_name
        self.text_log_dir.mkdir(parents=True, exist_ok=True)
        self.tb_log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Python logger
        self._setup_text_logger()

        # Initialize TensorBoard writer
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(self.tb_log_dir))
            self.logger.info(f"TensorBoard logging to: {self.tb_log_dir}")
        else:
            self.writer = None
            self.logger.warning("TensorBoard not available - metrics won't be visualized")

        # Log configuration
        if config:
            self.log_config(config)

        self.logger.info(f"Training logger initialized: {experiment_name}")

    def _setup_text_logger(self):
        """Setup Python text logger."""
        self.logger = logging.getLogger('training')
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        # File handler
        log_file = self.text_log_dir / "training.log"
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log_config(self, config: Dict[str, Any]):
        """Log training configuration."""
        self.logger.info("=" * 70)
        self.logger.info("Training Configuration:")
        self.logger.info("=" * 70)
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 70)

        # Log to TensorBoard as text
        if self.writer:
            config_text = "\n".join([f"{k}: {v}" for k, v in config.items()])
            self.writer.add_text("Config", config_text, 0)

    def log_episode(
        self,
        episode: int,
        metrics: Dict[str, float],
        log_text: bool = True,
        log_tensorboard: bool = True
    ):
        """
        Log episode-level metrics.

        Args:
            episode: Episode number
            metrics: Dictionary of metric names and values
            log_text: Whether to log to text file
            log_tensorboard: Whether to log to TensorBoard
        """
        # Text logging
        if log_text:
            metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"[Episode {episode:4d}] {metric_str}")

        # TensorBoard logging
        if log_tensorboard and self.writer:
            for metric_name, value in metrics.items():
                self.writer.add_scalar(metric_name, value, episode)

    def log_step(
        self,
        step: int,
        metrics: Dict[str, float],
        log_tensorboard: bool = True
    ):
        """
        Log step-level metrics (typically to TensorBoard only).

        Args:
            step: Global step number
            metrics: Dictionary of metric names and values
            log_tensorboard: Whether to log to TensorBoard
        """
        if log_tensorboard and self.writer:
            for metric_name, value in metrics.items():
                self.writer.add_scalar(metric_name, value, step)

    def log_checkpoint(self, episode: int, model_path: str):
        """Log checkpoint save event."""
        self.logger.info(f"[Checkpoint] Episode {episode} - Saved to: {model_path}")

    def log_evaluation(
        self,
        episode: int,
        eval_metrics: Dict[str, float]
    ):
        """
        Log evaluation results.

        Args:
            episode: Episode number
            eval_metrics: Evaluation metrics
        """
        self.logger.info("=" * 70)
        self.logger.info(f"Evaluation at Episode {episode}:")
        for metric_name, value in eval_metrics.items():
            self.logger.info(f"  {metric_name}: {value:.4f}")
        self.logger.info("=" * 70)

        # Log to TensorBoard with 'Eval/' prefix
        if self.writer:
            for metric_name, value in eval_metrics.items():
                self.writer.add_scalar(f"Eval/{metric_name}", value, episode)

    def log_comparison(
        self,
        episode: int,
        comparison_data: Dict[str, Dict[str, float]]
    ):
        """
        Log algorithm comparison results.

        Args:
            episode: Episode number
            comparison_data: Dict mapping algorithm names to their metrics
                Example: {
                    'MATD3': {'reward': 250, 'utilization': 0.85},
                    'Greedy': {'reward': 180, 'utilization': 0.75}
                }
        """
        self.logger.info("=" * 70)
        self.logger.info(f"Algorithm Comparison at Episode {episode}:")
        for algo_name, metrics in comparison_data.items():
            metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"  {algo_name}: {metric_str}")
        self.logger.info("=" * 70)

        # Log to TensorBoard
        if self.writer:
            for algo_name, metrics in comparison_data.items():
                for metric_name, value in metrics.items():
                    self.writer.add_scalar(
                        f"Comparison/{metric_name}/{algo_name}",
                        value,
                        episode
                    )

    def log_hyperparameters(
        self,
        hparams: Dict[str, Any],
        metrics: Dict[str, float]
    ):
        """
        Log hyperparameters and final metrics for comparison.

        Args:
            hparams: Hyperparameter dictionary
            metrics: Final performance metrics
        """
        if self.writer:
            self.writer.add_hparams(hparams, metrics)

    def log_histogram(
        self,
        tag: str,
        values,
        step: int
    ):
        """Log histogram data to TensorBoard."""
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def close(self):
        """Close all handlers."""
        if self.writer:
            self.writer.close()

        for handler in self.logger.handlers:
            handler.close()

        self.logger.info("Training logger closed")


class TestingLogger:
    """
    Logger for testing/evaluation phase.

    Similar to TrainingLogger but focused on test results.
    """

    def __init__(
        self,
        log_dir: str = "logs/testing",
        test_name: Optional[str] = None
    ):
        """
        Initialize testing logger.

        Args:
            log_dir: Base directory for test logs
            test_name: Name of test run
        """
        self.log_dir = Path(log_dir)

        if test_name is None:
            test_name = datetime.now().strftime("test_%Y%m%d_%H%M%S")

        self.test_name = test_name

        # Create directories
        self.text_log_dir = self.log_dir / "text"
        self.results_dir = self.log_dir / "results"
        self.text_log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self._setup_logger()

        self.logger.info(f"Testing logger initialized: {test_name}")

    def _setup_logger(self):
        """Setup Python logger for testing."""
        self.logger = logging.getLogger('testing')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        # File handler
        log_file = self.text_log_dir / "testing.log"
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log_test_start(self, config: Dict[str, Any]):
        """Log test configuration."""
        self.logger.info("=" * 70)
        self.logger.info("Test Configuration:")
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 70)

    def log_episode_result(
        self,
        episode: int,
        metrics: Dict[str, float]
    ):
        """Log single test episode result."""
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"[Test Episode {episode:3d}] {metric_str}")

    def log_summary(self, summary: Dict[str, Any]):
        """Log test summary statistics."""
        self.logger.info("=" * 70)
        self.logger.info("Test Summary:")
        self.logger.info("=" * 70)
        for key, value in summary.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 70)

    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """
        Save test results to JSON file.

        Args:
            results: Results dictionary
            filename: Output filename (default: test_{timestamp}.json)
        """
        import json

        if filename is None:
            filename = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_file = self.results_dir / filename

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Results saved to: {output_file}")

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def close(self):
        """Close logger handlers."""
        for handler in self.logger.handlers:
            handler.close()
