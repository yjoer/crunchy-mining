import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base_sampler import BaseSampler


class SamplerV1(BaseSampler):
    def sample(self, df: pd.DataFrame):
        df_train, df_test = super().split(df)
        df_train_sm, df_val = super().validation_split(df_train)
        skf_indices = super().kfold_split(df_train)

        self.train_val_sets["testing"] = (df_train, df_test)
        self.train_val_sets["validation"] = (df_train_sm, df_val)

        for i, (train_idx, val_idx) in enumerate(skf_indices):
            df_train_fold = df_train.iloc[train_idx]
            df_val_fold = df_train.iloc[val_idx]

            self.train_val_sets[f"fold_{i}"] = (df_train_fold, df_val_fold)


class SamplerV2(BaseSampler):
    def sample(self, df: pd.DataFrame):
        counts = df[self.variables["target"]].value_counts()
        majority_class = counts.index[np.argmax(counts)]
        minority_class = counts.index[np.argmin(counts)]
        n_minority_class = np.min(counts)

        # Align the rows of the majority class to the minority class.
        df_majority = df[df[self.variables["target"]] == majority_class]
        df_majority_sampled = df_majority.sample(n_minority_class, random_state=12345)

        # Save the remaining rows of the majority class.
        df_majority_remaining = df_majority.drop(df_majority_sampled.index)

        # Split the remaining rows of the majority class into validation and test sets.
        df_val_extra, df_test_extra = train_test_split(
            df_majority_remaining,
            test_size=0.5,
            random_state=12345,
        )

        df_sampled = pd.concat(
            [
                df_majority_sampled,
                df[df[self.variables["target"]] == minority_class],
            ]
        )

        df_train, df_test = super().split(df_sampled)
        df_train_sm, df_val = super().validation_split(df_train)
        skf_indices = super().kfold_split(df_train)

        df_val = pd.concat((df_val, df_val_extra))
        df_test = pd.concat((df_test, df_test_extra))

        self.train_val_sets["testing"] = (df_train, df_test)
        self.train_val_sets["validation"] = (df_train_sm, df_val)

        for i, (train_idx, val_idx) in enumerate(skf_indices):
            df_train_fold = df_train.iloc[train_idx]
            df_val_fold = pd.concat((df_train.iloc[val_idx], df_val_extra))

            self.train_val_sets[f"fold_{i}"] = (df_train_fold, df_val_fold)
