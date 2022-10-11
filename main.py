import os
import json
import tempfile

import mlflow
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    root_path = hydra.utils.get_original_cwd()

    print(root_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        run_info = mlflow.run(
            project_uri,
            parameters
        )


if __name__ == '__main__':
    main()