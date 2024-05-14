import os
import pickle
import tempfile
from typing import Optional

import mlflow
import pandas as pd


def log_table(data, artifact_file: str, run_id: str):
    directory, filename = os.path.split(artifact_file)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, filename)
        data.to_json(path, orient="split", index=False)

        mlflow.log_artifact(
            local_path=path,
            artifact_path=directory,
            run_id=run_id,
        )


def load_table(artifact_uri: str):
    try:
        dst_path = mlflow.artifacts.download_artifacts(artifact_uri)
    except OSError:
        return None

    return pd.read_json(dst_path, orient="split")


def log_pickle(obj, artifact_file: str, run_id: Optional[str] = None):
    directory, filename = os.path.split(artifact_file)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, filename)

        with open(path, "wb") as f:
            pickle.dump(obj, f)

        mlflow.log_artifact(
            local_path=path,
            artifact_path=directory,
            run_id=run_id,
        )


def load_pickle(artifact_uri: str):
    try:
        dst_path = mlflow.artifacts.download_artifacts(artifact_uri)
    except OSError:
        return None

    with open(dst_path, "rb") as f:
        return pickle.load(f)


def get_latest_run_id_by_name(name: str):
    runs = mlflow.search_runs(
        filter_string=f"run_name = '{name}'",
        order_by=["start_time DESC"],
    )

    if len(runs) == 0:
        return None

    return runs.iloc[0]["run_id"]


def get_nested_run_ids_by_parent_id(parent_run_id: str, name: str = None):
    filter_string = f"tags.mlflow.parentRunId = '{parent_run_id}'"

    if name:
        filter_string += f" AND run_name = '{name}'"

    runs = mlflow.search_runs(filter_string=filter_string)

    if len(runs) == 0:
        return None

    if name:
        return runs.iloc[0]["run_id"]

    return runs["run_id"].tolist()


def get_nested_runs_by_parent_id(parent_run_id: str, filter_string: str = None):
    fs = f"tags.mlflow.parentRunId = '{parent_run_id}'"

    if filter_string:
        fs += f" AND {filter_string}"

    runs = mlflow.search_runs(filter_string=fs)

    if len(runs) == 0:
        return None

    return runs
