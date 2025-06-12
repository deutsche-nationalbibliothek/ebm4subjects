import logging
from pathlib import Path

import xgboost


class EbmLogger:
    def __init__(self, log_path: Path, level: str = "info") -> None:
        self.logger = logging.getLogger(__name__)

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

        log_file_handler = logging.FileHandler(f"{log_path}/ebm.log")
        log_file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s: %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
        )

        self.logger.addHandler(log_file_handler)

    def get_logger(self) -> logging.Logger:
        return self.logger


class XGBLogging(xgboost.callback.TrainingCallback):
    def __init__(
        self,
        logger: logging.Logger,
        epoch_log_interval: int = 100,
    ) -> None:
        self.logger = logger
        self.epoch_log_interval = epoch_log_interval

    def after_iteration(
        self,
        model: xgboost.Booster,
        epoch: int,
        evals_log: dict,
    ) -> bool:
        if epoch % self.epoch_log_interval == 0:
            for data, metric in evals_log.items():
                metrics = list(metric.keys())
                metrics_str = ""
                for metric_key in metrics:
                    metrics_str += f"{metric_key}: {metric[metric_key][-1]}"
                self.logger.info(f"Epoch: {epoch}, {data}: {metrics_str}")

        return False
