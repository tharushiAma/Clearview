# src/data_layer/_common.py

import yaml
import logging
import sys
from pathlib import Path

def load_cfg(project_dir: str):
    """
    Loads config.yaml from the project directory
    """
    project_dir = str(Path(project_dir).resolve())
    cfg_path = Path(project_dir) / "configs" / "config.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError("config.yaml not found")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    return project_dir, cfg


def setup_logger(project_dir: str, name: str):
    """
    Creates a logger that prints to:
    - terminal
    - outputs/logs/<script>.log
    """
    logs_dir = Path(project_dir) / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # write logs to file
    file_handler = logging.FileHandler(logs_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    # print logs to terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def ensure_dirs(project_dir: str):
    """
    Makes sure required folders exist
    """
    Path(project_dir, "data", "processed").mkdir(parents=True, exist_ok=True)
    Path(project_dir, "data", "splits").mkdir(parents=True, exist_ok=True)
    Path(project_dir, "data", "augmented").mkdir(parents=True, exist_ok=True)
    Path(project_dir, "outputs", "logs").mkdir(parents=True, exist_ok=True)
    Path(project_dir, "outputs", "checkpoints").mkdir(parents=True, exist_ok=True)
    Path(project_dir, "outputs", "reports").mkdir(parents=True, exist_ok=True)
    Path(project_dir, "outputs", "plots").mkdir(parents=True, exist_ok=True)

