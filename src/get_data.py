import os
from io import StringIO
import pathlib
import logging
import yaml
import pandas as pd

import argparse
import mlflow
import dvc.api

from main import read_param

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

root_path = str(pathlib.Path(__file__).parents[1].resolve())

# Chagne directory to root
os.chdir('..')

def main(config):

    # Read config from arguments
    data_config = read_param(config)

    # Log data artifact
    with mlflow.start_run() as run:
        
        data_url = dvc.api.get_url(path=data_config["dvc"]["dvc_proj"], repo=root_path, remote=data_config["dvc"]["remote_name"])
        train_data = dvc.api.read(path=data_config["dvc"]["train_data"], repo=root_path, remote=data_config["dvc"]["remote_name"])
        train_df = pd.read_csv(StringIO(train_data), sep=',') 

        train_num_rows = f"train num rows: {str(train_df.shape[0])}"
        train_num_cols = f"train num cols: {str(train_df.shape[1])}"
        train_data_path = f"train path: {data_config['dvc']['train_data']}"

        # log train info
        with open(os.path.join(root_path, "artifacts", "data", "raw", "train_info.txt"), 'w') as f:
            f.write(data_url + '\n')
            f.write(train_num_rows + '\n')
            f.write(train_num_cols + '\n')
            f.write(train_data_path + '\n')

        # log train info as mlflow artifact
        mlflow.log_artifact(local_path = os.path.join(root_path, "artifacts", "data", "raw", "train_info.txt"), artifact_path="data/raw")

        test_data = dvc.api.read(path=data_config["dvc"]["test_data"], repo=root_path, remote=data_config["dvc"]["remote_name"])
        test_df = pd.read_csv(StringIO(test_data), sep=',') 

        test_num_rows = f"test num rows: {str(test_df.shape[0])}"
        test_num_cols = f"test num cols: {str(test_df.shape[1])}"
        test_data_path = f"test path: {data_config['dvc']['test_data']}"

        with open(os.path.join(root_path, "artifacts", "data", "raw", "test_info.txt"), 'w') as f:
            f.write(test_num_rows + '\n')
            f.write(test_num_cols + '\n')
            f.write(test_data_path + '\n')

        # log train info as mlflow artifact
        mlflow.log_artifact(local_path = os.path.join(root_path, "artifacts", "data", "raw", "test_info.txt"), artifact_path="data/raw")

        run = mlflow.active_run()

        logger.info(f"Train and test data information logged as artifacts in run {run.info.run_id}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="This steps is to train model")
    
    args.add_argument("--config", type=str, default="config.yaml")

    parsed_args = args.parse_args()

    main(config=parsed_args.config)