import logging

import xgboost


class EbmLogger:
    """
    A custom logger class.

    This class provides a way to log messages at different levels 
    (error, warning, info, debug) to a file.
    It also provides a way to get the logger instance.

    Attributes:
        logger (logging.Logger): The logger instance.
        log_path (str): The path to the log file.
        level (str): The log level (default: "info").
    """
    def __init__(self, log_path: str, level: str = "info") -> None:
        """
        Initializes the logger.

        Args:
            log_path (str): The path to the log file.
            level (str): The log level (default: "info").
        """
        self.logger = logging.getLogger(__name__)

        # Set the log level based on the provided level
        if level == "error":
            self.logger.setLevel(logging.ERROR)
        elif level == "warning":
            self.logger.setLevel(logging.WARNING)
        elif level == "info":
            self.logger.setLevel(logging.INFO)
        elif level == "debug":
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.NOTSET)

        # Create a file handler to log messages to a file
        log_file_handler = logging.FileHandler(f"{log_path}/ebm.log")
        log_file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s: %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )

        # Add the file handler to the logger
        self.logger.addHandler(log_file_handler)

    def get_logger(self) -> logging.Logger:
        """
        Returns the logger instance.

        Returns:
            logging.Logger: The logger instance.
        """
        return self.logger


class NullLogger:
    """
    A null logger class that does nothing.

    This class is used when no logging is needed.
    """
    def __init__(self) -> None:
        """
        Initializes the null logger.
        """
        pass

    def debug(self, *args, **kwargs):
        """
        Does nothing when debug message is logged.

        Args:
            *args: The message to log.
            **kwargs: Additional keyword arguments.
        """
        pass

    def info(self, *args, **kwargs):
        """
        Does nothing when info message is logged.

        Args:
            *args: The message to log.
            **kwargs: Additional keyword arguments.
        """
        pass

    def warn(self, *args, **kwargs):
        """
        Does nothing when warn message is logged.

        Args:
            *args: The message to log.
            **kwargs: Additional keyword arguments.
        """
        pass

    def warning(self, *args, **kwargs):
        """
        Does nothing when warning message is logged.

        Args:
            *args: The message to log.
            **kwargs: Additional keyword arguments.
        """
        pass

    def error(self, *args, **kwargs):
        """
        Does nothing when error message is logged.

        Args:
            *args: The message to log.
            **kwargs: Additional keyword arguments.
        """
        pass

    def critical(self, *args, **kwargs):
        """
        Does nothing when critical message is logged.

        Args:
            *args: The message to log.
            **kwargs: Additional keyword arguments.
        """
        pass


class XGBLogging(xgboost.callback.TrainingCallback):
    """
    Custom XGBoost training callback for logging model performance during training.
    
    Args:
        logger (logging.Logger): Logger instance to use for logging.
        epoch_log_interval (int, optional): Interval at which to log model performance
            (default: 100).
    
    Attributes:
        logger (logging.Logger): Logger instance used for logging.
        epoch_log_interval (int): Interval at which to log model performance.
    """
    def __init__(
        self,
        logger: logging.Logger,
        epoch_log_interval: int = 100,
    ) -> None:
        """
        Initializes the XGBLogger.
        
        Args:
            logger (logging.Logger): Logger instance to use for logging.
            epoch_log_interval (int, optional): Interval at which to log model 
                performance (default: to 100).
        """
        # Logger instance used for logging
        self.logger = logger
        # Interval at which to log model performance
        self.epoch_log_interval = epoch_log_interval

    def after_iteration(
        self,
        model: xgboost.Booster,
        epoch: int,
        evals_log: dict,
    ) -> bool:
        """
        Callback function called after each iteration of the XGBoost training process.
        
        Logs model performance at the specified interval.
        
        Args:
            model (xgboost.Booster): XGBoost model instance.
            epoch (int): Current epoch number.
            evals_log (dict): Dictionary containing evaluation metrics.
        
        Returns:
            bool: Always returns False, as specified by the XGBoost callback API.
        """
        # Log model performance at the specified interval
        if epoch % self.epoch_log_interval == 0:
            # Iterate over each data point and its corresponding metrics
            for data, metric in evals_log.items():
                # Get the list of metric keys
                metrics = list(metric.keys())

                # Construct a string containing the metric values
                metrics_str = ""
                for metric_key in metrics:
                    # Append the metric key and its value to the string
                    metrics_str += f"{metric_key}: {metric[metric_key][-1]}"

                # Log the model performance using the specified logger
                self.logger.info(f"Epoch: {epoch}, {data}: {metrics_str}")

        # Always return False, as specified by the XGBoost callback API
        return False
