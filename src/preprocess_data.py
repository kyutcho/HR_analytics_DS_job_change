import os
from io import StringIO
import pathlib
import logging
import yaml
import numpy as np
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


def df_numeric_summary(df):
    stats = df.describe(include=np.number).loc[["mean", "25%", "50%", "75%", "std"], :]
    stats_dict = stats.to_dict(orient='list')

    return stats_dict


def main(config):

    # Read config from arguments
    data_config = read_param(config)

    # Remove features not needed - enrollee_id, city
    with mlflow.start_run() as run:
        col_to_drop = ["enrollee_id", "city"]

        data_url = dvc.api.get_url(path=data_config["dvc"]["dvc_proj"], repo=root_path, remote=data_config["dvc"]["remote_name"])
        train_data = dvc.api.read(path=data_config["dvc"]["train_data"], repo=root_path, remote=data_config["dvc"]["remote_name"])
        train_df = pd.read_csv(StringIO(train_data), sep=',')
        train_df.drop(col_to_drop, axis=1, inplace=True)

        # Save processed train df
        train_df.to_csv(os.path.join(root_path, "data", "processed", "train_processed.csv"), index=False)

        test_data = dvc.api.read(path=data_config["dvc"]["test_data"], repo=root_path, remote=data_config["dvc"]["remote_name"])
        test_df = pd.read_csv(StringIO(test_data), sep=',')
        test_df.drop(col_to_drop, axis=1, inplace=True)

        # Save processed test df
        test_df.to_csv(os.path.join(root_path, "data", "processed", "test_processed.csv"), index=False)

        with open(os.path.join(root_path, "artifacts", "data", "processed", "dropped_features.txt"), 'w') as f:
            for i, col in enumerate(col_to_drop):
                f.write(f"{i+1}: {col}" + '\n')

        # log processed data as artifact in data/processed
        mlflow.log_artifact(local_path = os.path.join(root_path, "artifacts", "data", "processed", "dropped_features.txt"), 
                            artifact_path="data/processed")

        logger.info(f"Dropped columns logged as artifacts in run {run.info.run_id}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="This steps is to preprocess data")
    
    args.add_argument("--config", type=str, default="config.yaml")

    parsed_args = args.parse_args()

    main(config=parsed_args.config)