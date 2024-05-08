from __future__ import annotations

import typing
from abc import ABC
from abc import abstractmethod
from typing import Dict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

if typing.TYPE_CHECKING:
    from omegaconf import DictConfig


class BaseSampler(ABC):
    def __init__(self, cfg: Optional[DictConfig] = None):
        self.train_val_sets: Dict[str, list | tuple] = {}
        self.cfg = cfg

    @abstractmethod
    def sample(self):
        pass

    def split(self, df: pd.DataFrame):
        df_train, df_test = train_test_split(
            df,
            test_size=0.15,
            random_state=12345,
            stratify=df[self.cfg.vars.target],
        )

        return df_train, df_test

    # 1. Holdout method
    def validation_split(self, df_train: pd.DataFrame):
        df_train_sm, df_val = train_test_split(
            df_train,
            test_size=0.15 / 0.85,
            random_state=12345,
            stratify=df_train[self.cfg.vars.target],
        )

        return df_train_sm, df_val

    # 2. Cross-validation
    def kfold_split(self, df_train: pd.DataFrame):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
        skf_indices = list(
            skf.split(
                X=np.zeros(len(df_train)),
                y=df_train[self.cfg.vars.target],
            )
        )

        return skf_indices
