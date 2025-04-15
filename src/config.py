from dataclasses import dataclass
from pathlib import Path


@dataclass
class CFG:
    train_path: Path = Path("./data/train.csv")
    test_path: Path = Path("./data/test.csv")
    sub_path: Path = Path("./data/sample_submission.csv")

    num_fold: int = 5
    dev_mode: bool = False

    # Model parameters
    n_iter: int = 10000
    max_depth: int = -1
    num_leaves: int = 1024
    colsample_bytree: float = 0.7
    learning_rate: float = 0.04

    objective: str = "l2"
    metric: str = "rmse"
    verbosity: int = -1

    random_state: int = 42
    shuffle: bool = True
    encoded_columns_start: int = -91
    log_eval: int = 100
    early_stopping: int = 200


cfg = CFG()
