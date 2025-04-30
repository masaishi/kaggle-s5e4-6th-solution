from dataclasses import dataclass
from pathlib import Path


@dataclass
class CFG:
    train_path: Path = Path("./data/train.csv")
    test_path: Path = Path("./data/test.csv")
    sub_path: Path = Path("./data/sample_submission.csv")
    pltpd_path: Path = Path("./data/podcast_dataset.csv")

    model_path: Path = Path("./model/lgbm_model.txt")
    test_output_path: Path = Path("./data/submission.csv")

    # num_fold: int = 5
    num_fold: int = 7
    dev_mode: bool = False

    # Model parameters
    n_iter: int = 10000
    max_depth: int = -1

    # num_leaves: int = 2048
    # colsample_bytree: float = 0.5
    # learning_rate: float = 0.04
    # random_state: int = 242

    num_leaves: int = 4096
    colsample_bytree: float = 0.5
    learning_rate: float = 0.03
    random_state: int = 50

    objective: str = "l2"
    metric: str = "rmse"
    verbosity: int = -1

    shuffle: bool = True
    log_eval: int = 100
    early_stopping: int = 200

    # default_combinations: bool = True

    # debug: bool = True
    eval: bool = True
    predict: bool = True

    device: str = "cpu"


cfg = CFG()

if hasattr(cfg, "eval") and cfg.eval:
    cfg.n_iter = 2000
if hasattr(cfg, "debug") and cfg.debug:
    cfg.n_iter = 5
