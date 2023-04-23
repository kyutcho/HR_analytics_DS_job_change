import os
import json
import logging
import tempfile
import pathlib
import argparse
import yaml

import mlflow
from mlflow.tracking import MlflowClient

import hydra
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

root_path = str(pathlib.Path(__file__).parents[1].resolve())

steps = [
    # "get_data",
    # "preprocess_data",
    "train_model"
]

def read_param(config_path):

    config_path = os.path.join(root_path, "config", config_path)

    with open(config_path) as fp:
        config = yaml.safe_load(fp)

    return config

# @hydra.main(version_base=None, config_path="../config", config_name="config")
# def main(config: DictConfig) -> None:
def main(args):

    # os.environ["MLFLOW_EXPERIMENT_NAME"] = config["main"]["project_name"]
    # os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db" #"http://127.0.0.1:5000"

    # mlflow.set_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    # mlflow.set_experiment = os.getenv("MLFLOW_EXPERIMENT_NAME")

    # print(os.getenv("MLFLOW_EXPERIMENT_NAME"))

    # client = MlflowClient()

    main_param = read_param(args.config)

    model_to_use = str(args.model)

    logger.info(f"Using {model_to_use} as a model")

    with tempfile.TemporaryDirectory() as tmp_dir:

        if "get_data" in steps:
        # Run MLflow Project - get_data.py
            run_info = mlflow.run(
                uri = os.path.join(root_path, "src"),
                entry_point = "get_data",
                parameters = {
                    "config": args.config
                }
            )

            logger.info("Process - get_data is complete")

            # prev_step_run = client.get_run(run_info.run_id)
            # print(prev_step_run.info)
            # print(prev_step_run.data)

        if "preprocess_data" in steps:
            # Run MLflow Project - preprocess_data.py
            run_info = mlflow.run(
                uri = os.path.join(root_path, "src"),
                entry_point = "preprocess_data",
                parameters = {
                    "config": args.config
                }
            )

        if "train_model" in steps:
            # Run MLflow Project - train_model.py
            run_info = mlflow.run(
                uri = os.path.join(root_path, "src"),
                entry_point = "train_model",
                parameters = {
                    "config": args.config,
                    "model": model_to_use
                }
            )


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="This is main")
    
    args.add_argument("--config", type=str, default="config.yaml")

    args.add_argument("--model", type=str, default="test_model")

    parsed_args = args.parse_args()

    main(parsed_args)