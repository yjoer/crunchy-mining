import pandas as pd

from .base_sampler import BaseSampler


class SamplerV1(BaseSampler):
    def sample(self, df: pd.DataFrame):
        df_train, df_test = super().reg_split(df)
        df_train_sm, df_val = super().reg_validation_split(df_train)
        kf_indices = super().reg_kfold_split(df_train)

        self.train_val_sets["testing"] = (df_train, df_test)
        self.train_val_sets["validation"] = (df_train_sm, df_val)

        for i, (train_idx, val_idx) in enumerate(kf_indices):
            df_train_fold = df_train.iloc[train_idx]
            df_val_fold = df_train.iloc[val_idx]

            self.train_val_sets[f"fold_{i+1}"] = (df_train_fold, df_val_fold)


class SamplerV2(BaseSampler):
    def sample(self, df: pd.DataFrame):
        df_train, df_test = super().split(df)
        df_train_sm, df_val = super().validation_split(df_train)
        skf_indices = super().kfold_split(df_train)

        self.train_val_sets["testing"] = (df_train, df_test)
        self.train_val_sets["validation"] = (df_train_sm, df_val)

        for i, (train_idx, val_idx) in enumerate(skf_indices):
            df_train_fold = df_train.iloc[train_idx]
            df_val_fold = df_train.iloc[val_idx]

            self.train_val_sets[f"fold_{i+1}"] = (df_train_fold, df_val_fold)
