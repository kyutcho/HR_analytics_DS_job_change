import os
from io import StringIO
import pathlib
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import mlflow
import dvc.api

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from main import read_param

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

root_path = str(pathlib.Path(__file__).parents[1].resolve())

# Chagne directory to root
os.chdir('..')


def exp_to_num_trans(experience_arr: pd.Series) -> np.array:
    """
    Function to convert categorical feature experience to numeric type
    """
    exp_trans = np.where(experience_arr == ">20", "25", np.where(experience_arr == "<1", "0", experience_arr))
    exp_to_num = pd.to_numeric(exp_trans, errors="raise").reshape(-1,1)

    return exp_to_num


def lnj_to_num_trans(last_new_job_arr: pd.Series) -> np.array:
    """
    Function to convert categorical feature last new job to numeric type
    """
    exp_trans = np.where(last_new_job_arr == ">4", "5", np.where(last_new_job_arr == "never", "0", last_new_job_arr))
    exp_to_num = pd.to_numeric(exp_trans, errors="raise").reshape(-1,1)

    return exp_to_num


def com_size_reformat_trans(com_size_arr: pd.Series) -> np.array:
    """
    Function to reformat one value contains '/' to '-'
    """
    reformat_com_size = np.array(com_size_arr.str.replace('/', '-')).reshape(-1,1)

    return reformat_com_size

def inference_pipeline(model: str=None, model_config: str=None):
    """
    Function to create inference pipeline
    Input:
        - model: model to be used in pipeline
        - model_config: parameters to be used in the model
    """
    model_config = list(model_config.values())[0]
    model_config = {k: None if v == "None" else v for k, v in model_config.items()}

    impute_na = model_config.pop("impute_na")

    if impute_na:

        numeric_cols = ["city_development_index", "experience", "last_new_job", "training_hours"]
        ordinal_cols = ["company_size"]
        categorical_cols = ["gender", "relevent_experience", "enrolled_university", "major_discipline", "education_level", "company_type"]

        to_numeric_transformer = ColumnTransformer([("experience_transformer", FunctionTransformer(exp_to_num_trans), "experience"), \
                                                    ("last_new_job_transformer", FunctionTransformer(lnj_to_num_trans), "last_new_job")],
                                                    remainder="passthrough")

        reformat_transformer = ColumnTransformer([("company_size_transformer", FunctionTransformer(com_size_reformat_trans), "company_size")],
                                                remainder="passthrough")                                             

        num_pipe = Pipeline([('to_numeric', to_numeric_transformer),
                            ('numeric_imputer', SimpleImputer())])

        ord_pipe = Pipeline([('reformat', reformat_transformer),
                            ('ordinal_imputer', SimpleImputer(strategy="constant", fill_value="Unknown")),
                            ('ordinal_encoder', OrdinalEncoder(
                                categories=[['<10', '10-49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+']], 
                                handle_unknown="use_encoded_value", unknown_value=-1))])
        

        cat_pipe = Pipeline([('categorical_imputer', SimpleImputer(strategy="constant", fill_value="Unknown")),
                            ('categorical_oh_encoder', OneHotEncoder(drop="first"))])       

    preprocessor = ColumnTransformer(
        [("num_pipe", num_pipe, numeric_cols), \
        ("ord_pipe", ord_pipe, ordinal_cols), \
        ("cat_pipe", cat_pipe, categorical_cols)],
        remainder="drop"
    )

    model_pipeline = make_pipeline(preprocessor, LogisticRegression(**model_config))

    return model_pipeline


def main(args):

    # Read config from arguments
    main_config = read_param(args.config)

    model_to_use = main_config["model"][str(args.model)]

    model_config = read_param(os.path.join("model", model_to_use))

    # # Remove features not needed - enrollee_id, city
    # with mlflow.start_run() as run:
    #     pass

    #  TO-USE-LATER
    # train_data = dvc.api.read(path=data_config["dvc"]["processed_train_data"], repo=root_path, remote=data_config["dvc"]["remote_name"])
    # train_df = pd.read_csv(StringIO(train_data), sep=',')

    train_df = pd.read_csv(os.path.join(root_path, "data", "processed", "train_processed.csv"))

    X_train = train_df
    y_train = X_train.pop("target")

    del train_df

    n_splits = 5
    kfold = StratifiedKFold(n_splits=n_splits)

    logger.info(f"Stratified {n_splits} sets completed")

    infer_pipeline = inference_pipeline(model=model_to_use, model_config=model_config)

    scores = cross_val_score(estimator=infer_pipeline, X=X_train, y=y_train, cv=kfold)

    print(f"cv scores: {scores}")

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="This steps is to train model")
    
    args.add_argument("--config", type=str, default="config.yaml")

    args.add_argument("--model", type=str, required=True)

    parsed_args = args.parse_args()

    main(parsed_args)