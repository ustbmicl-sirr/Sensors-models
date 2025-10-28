"""
Backend logging utilities for the web API.

Provides structured logging for API requests, algorithm execution,
and error tracking.
"""

import logging
import logging.handlers
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Any, Optional
import sys


class StructuredLogger:
    """
    Structured logger for backend services.

    Logs are formatted as JSON for easy parsing and analysis.
    """

    def __init__(
        self,
        name: str,
        log_dir: str = "logs/backend",
        log_level: int = logging.INFO
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name (e.g., 'api', 'algorithm', 'task')
            log_dir: Directory for log files
            log_level: Logging level
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(f"backend.{name}")
        self.logger.setLevel(log_level)
        self.logger.handlers.clear()

        # Console handler (human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)

        # File handler (JSON format)
        log_file = self.log_dir / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(JSONFormatter())

        # Error file handler
        error_file = self.log_dir / "error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s\n%(exc_info)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        **kwargs
    ):
        """
        Log API request.

        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            duration_ms: Request duration in milliseconds
            **kwargs: Additional context
        """
        self.logger.info(
            "API Request",
            extra={
                'event_type': 'api_request',
                'method': method,
                'path': path,
                'status_code': status_code,
                'duration_ms': duration_ms,
                **kwargs
            }
        )

    def log_task_generation(
        self,
        task_id: str,
        num_vessels: int,
        generation_mode: str,
        duration_ms: float
    ):
        """Log task generation event."""
        self.logger.info(
            f"Task generated: {task_id}",
            extra={
                'event_type': 'task_generated',
                'task_id': task_id,
                'num_vessels': num_vessels,
                'generation_mode': generation_mode,
                'duration_ms': duration_ms
            }
        )

    def log_algorithm_execution(
        self,
        task_id: str,
        algorithm: str,
        allocation_id: str,
        duration_ms: float,
        metrics: Dict[str, float]
    ):
        """Log algorithm execution."""
        self.logger.info(
            f"Algorithm executed: {algorithm}",
            extra={
                'event_type': 'algorithm_executed',
                'task_id': task_id,
                'algorithm': algorithm,
                'allocation_id': allocation_id,
                'duration_ms': duration_ms,
                **metrics
            }
        )

    def log_error(
        self,
        error_type: str,
        error_message: str,
        **context
    ):
        """Log error event."""
        self.logger.error(
            f"{error_type}: {error_message}",
            extra={
                'event_type': 'error',
                'error_type': error_type,
                'error_message': error_message,
                **context
            },
            exc_info=True
        )

    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, extra=kwargs, exc_info=True)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs log records as JSON objects.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'message', 'pathname', 'process', 'processName',
                'relativeCreated', 'thread', 'threadName', 'exc_info',
                'exc_text', 'stack_info'
            ]:
                log_data[key] = value

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)


# Global logger instances
api_logger = StructuredLogger('api')
task_logger = StructuredLogger('task')
algorithm_logger = StructuredLogger('algorithm')


def get_logger(name: str) -> StructuredLogger:
    """
    Get or create a logger instance.

    Args:
        name: Logger name

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)
