import mlflow


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
