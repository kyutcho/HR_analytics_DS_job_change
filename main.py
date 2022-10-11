import os
import json

import mlflow
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config.yaml")
def main(cfg: DictConfig) -> None:
    root_path = hydra.utils.get_original_cwd()

    print(root_path)


if __name__ == '__main__':
    main()