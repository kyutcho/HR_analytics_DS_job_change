import os
import yaml
import pandas as pd
import json
import argparse
import mlflow

def read_param(config_path):
    with open(config_path) as fp:
        config = yaml.safe_load(fp)

    return config

def get_data(data_path):    
    df = pd.read_csv(data_path)

    return df

def main(config_path):
    config = read_param(config_path)

    train_data_path = config["data"]["train_path"]
    train_df = get_data(train_data_path)

    test_data_path = config["data"]["test_path"]
    test_df = get_data(test_data_path)

    with mlflow.start_run():
        mlflow.log_artifact




if __name__ == "__main__":
    args = argparse.ArgumentParser(description="This steps is to get data from specified path")
    args.add_argument("--config", default="config.yaml")

    parsed_args = args.parse_args()

    main(config_path=parsed_args.config)