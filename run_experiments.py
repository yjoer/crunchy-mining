import logging
import os
import subprocess
import sys

from crunchy_mining.util import set_low_priority

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

experiments_clf = [
    # "sampling_v1",
    # "sampling_v2",
    # "preprocessing_v1",
    # "preprocessing_v2",
    # "preprocessing_v3",
    # "preprocessing_v4",
    # "preprocessing_v5",
    # "preprocessing_v6",
    # "preprocessing_v7",
    # "resampling_v1",
    # "resampling_v2",
    # "resampling_v3",
    # "resampling_v4",
    # "resampling_v5",
    # "resampling_v6",
    # "resampling_v7",
    # "resampling_v8",
]

experiments_bank = [
    # "bank/sampling_v1",
    # "bank/sampling_v2",
    # "bank/preprocessing_v1",
    # "bank/preprocessing_v2",
    # "bank/preprocessing_v3",
    # "bank/preprocessing_v4",
    # "bank/preprocessing_v5",
    # "bank/preprocessing_v6",
    # "bank/preprocessing_v7",
    # "bank/preprocessing_v8",
    # "bank/preprocessing_v9",
    # "bank/preprocessing_v10",
    # "bank/preprocessing_v11",
]

experiments_sba = [
    # "sba/sampling_v1",
    # "sba/sampling_v2",
    # "sba/preprocessing_v1",
    # "sba/preprocessing_v2",
    # "sba/preprocessing_v3",
    # "sba/preprocessing_v4",
    # "sba/preprocessing_v5",
    # "sba/preprocessing_v6",
    # "sba/preprocessing_v7",
    # "sba/preprocessing_v8",
    # "sba/preprocessing_v9",
    # "sba/preprocessing_v10",
    # "sba/preprocessing_v11",
]

for experiment in experiments_clf + experiments_bank + experiments_sba:
    env = os.environ.copy()
    env["CM_EXPERIMENT"] = experiment

    logger.info(f"Running experiment: {experiment}")
    task_name, experiment_file = experiment.split("/")

    match task_name:
        case "":
            file = "classification.py"
        case "bank":
            file = "regression.py"
        case "sba":
            file = "regression.py"
        case _:
            sys.exit()

    process = subprocess.Popen([sys.executable, file], env=env)
    set_low_priority(process.pid)
    process.wait()
