import concurrent.futures
import os
import pickle
import tempfile
from concurrent.futures import ThreadPoolExecutor
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


def get_cv_metrics_by_model(task_name: str, model_name: str):
    experiments = mlflow.search_experiments(filter_string=f"name LIKE '{task_name}%'")
    experiments_map = {}

    for experiment in experiments:
        experiments_map[experiment.experiment_id] = experiment.name

    df_parent_runs = mlflow.search_runs(
        experiment_ids=experiments_map.keys(),
        filter_string=f"run_name = '{model_name}'",
    )

    futures = []
    dfs = []

    # SQL IN queries are not supported for tags.mlflow.parentRunId.
    # https://mlflow.org/docs/latest/search-runs.html#searching-over-a-set
    with ThreadPoolExecutor(max_workers=len(df_parent_runs)) as executor:
        for parent_run_id in df_parent_runs["run_id"]:
            fs = f"tags.mlflow.parentRunId = '{parent_run_id}'"
            fs += " AND run_name LIKE 'fold%'"

            future = executor.submit(
                mlflow.search_runs,
                experiment_ids=experiments_map.keys(),
                filter_string=fs,
            )

            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            df = future.result()
            dfs.append(df)

    df_agg = (
        pd.concat(dfs, axis=0)
        .rename(columns={"tags.mlflow.runName": "nested_run_name"})
        .groupby("tags.mlflow.parentRunId")
        .agg(
            {
                "experiment_id": "first",
                "nested_run_name": list,
                **{col: "mean" for col in df.columns if col.startswith("metrics")},
            },
        )
        .rename(lambda x: x.replace("metrics.", ""), axis=1)
        .assign(experiment_name=lambda x: x["experiment_id"].map(experiments_map))
    )

    return df_agg


def get_cv_metrics_by_task(task_name: str):
    experiments = mlflow.search_experiments(filter_string=f"name LIKE '{task_name}%'")
    experiments_map = {}

    for experiment in experiments:
        experiments_map[experiment.experiment_id] = experiment.name

    # This takes about 7 seconds.
    df = mlflow.search_runs(experiment_ids=experiments_map.keys())

    # https://github.com/mlflow/mlflow/issues/2922
    df_parent_runs = (
        df.query("`tags.mlflow.parentRunId`.isnull()")
        .query("`tags.mlflow.runName` != 'Encoders'")
        .sort_values(by=["experiment_id", "start_time"], ascending=[True, False])
        .drop_duplicates(subset=["experiment_id", "tags.mlflow.runName"], keep="first")
        .loc[:, ["experiment_id", "run_id", "tags.mlflow.runName"]]
    )

    df_agg = (
        # Select nested runs of selected parent runs.
        df.merge(
            df_parent_runs[["run_id", "tags.mlflow.runName"]],
            how="inner",
            left_on="tags.mlflow.parentRunId",
            right_on="run_id",
        )
        .rename(
            columns={
                "tags.mlflow.runName_x": "nested_run_name",
                "tags.mlflow.runName_y": "parent_run_name",
            }
        )
        .query("~nested_run_name.isin(['validation', 'testing'])")
        .groupby("tags.mlflow.parentRunId")
        .agg(
            {
                "experiment_id": "first",
                "parent_run_name": "first",
                "nested_run_name": list,
                **{col: "mean" for col in df.columns if col.startswith("metrics")},
            },
        )
        .rename(lambda x: x.replace("metrics.", ""), axis=1)
        .assign(experiment_name=lambda x: x["experiment_id"].map(experiments_map))
    )

    return df_agg
