from __future__ import annotations

import typing
from abc import ABC
from abc import abstractmethod
from typing import Dict

import mlflow
import numpy as np

from .. import mlflow_util

if typing.TYPE_CHECKING:
    from omegaconf import DictConfig


class BasePreprocessor(ABC):
    def __init__(self, cfg: DictConfig):
        self.train_val_sets: Dict[str, list | tuple] = {}
        self.encoders = {}
        self.cfg = cfg

    @abstractmethod
    def fit(self):
        pass

    def get_train_val_sets(self):
        return self.train_val_sets

    def save_train_val_sets(self):
        experiment = self.cfg.mlflow.experiment_name.replace("/", "_")
        arrays = {}

        # Explode the tuple into a flat list.
        for name, (X_train, y_train, X_test, y_test) in self.train_val_sets.items():
            arrays[f"{name}.x_train"] = X_train
            arrays[f"{name}.y_train"] = y_train
            arrays[f"{name}.x_test"] = X_test
            arrays[f"{name}.y_test"] = y_test

        np.savez_compressed(f"data/{experiment}.npz", **arrays)

    def load_train_val_sets(self):
        experiment = self.cfg.mlflow.experiment_name.replace("/", "_")
        arrays = {}

        npz = np.load(f"data/{experiment}.npz")

        for file in npz.files:
            name = file.split(".")[0]

            arrays.setdefault(name, [])
            arrays[name].append(npz[file])

        self.train_val_sets = arrays

    def get_encoders(self):
        return self.encoders

    def save_encoders(self):
        with mlflow.start_run(run_name="Encoders"):
            mlflow_util.log_pickle(self.encoders, artifact_file="encoders.pkl")

    def load_encoders(self):
        run_id = mlflow_util.get_latest_run_id_by_name(name="Encoders")
        self.encoders = mlflow_util.load_pickle(f"runs:/{run_id}/encoders.pkl")
