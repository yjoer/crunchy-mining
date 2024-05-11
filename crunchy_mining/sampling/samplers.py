import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

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

            self.train_val_sets[f"fold_{i+1}"] = (df_train_fold, df_val_fold)


class SamplerV2(BaseSampler):
    def sample(self, df: pd.DataFrame):
        counts = df[self.cfg.vars.target].value_counts()
        majority_class = counts.index[np.argmax(counts)]
        minority_class = counts.index[np.argmin(counts)]
        n_minority_class = np.min(counts)

        # Align the rows of the majority class to the minority class.
        df_majority = df[df[self.cfg.vars.target] == majority_class]
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
                df[df[self.cfg.vars.target] == minority_class],
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

            self.train_val_sets[f"fold_{i+1}"] = (df_train_fold, df_val_fold)


class ResamplerV0(BaseSampler):
    def sample(self, train_val_sets: dict):
        for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):
            self.train_val_sets[name] = (X_train, y_train, X_val, y_val)


class ResamplerV1(BaseSampler):
    def sample(self, train_val_sets: dict):
        for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):
            random = RandomOverSampler(random_state=12345)
            X_train_rs, y_train_rs = random.fit_resample(X_train, y_train)

            self.train_val_sets[name] = (X_train_rs, y_train_rs, X_val, y_val)


class ResamplerV2(BaseSampler):
    def sample(self, train_val_sets: dict):
        for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):
            nn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
            smote = SMOTE(random_state=12345, k_neighbors=nn)
            X_train_rs, y_train_rs = smote.fit_resample(X_train, y_train)

            self.train_val_sets[name] = (X_train_rs, y_train_rs, X_val, y_val)


class ResamplerV3(BaseSampler):
    def sample(self, train_val_sets: dict):
        for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):
            nn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
            adasyn = ADASYN(random_state=12345, n_neighbors=nn)
            X_train_rs, y_train_rs = adasyn.fit_resample(X_train, y_train)

            self.train_val_sets[name] = (X_train_rs, y_train_rs, X_val, y_val)


class ResamplerV4(BaseSampler):
    def sample(self, train_val_sets: dict):
        for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):
            k_nn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
            m_nn = NearestNeighbors(n_neighbors=10, n_jobs=-1)
            b_smote = BorderlineSMOTE(
                random_state=12345,
                k_neighbors=k_nn,
                m_neighbors=m_nn,
            )

            X_train_rs, y_train_rs = b_smote.fit_resample(X_train, y_train)

            self.train_val_sets[name] = (X_train_rs, y_train_rs, X_val, y_val)


class ResamplerV5(BaseSampler):
    def sample(self, train_val_sets: dict):
        for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):
            random = RandomUnderSampler(random_state=12345)
            X_train_rs, y_train_rs = random.fit_resample(X_train, y_train)

            self.train_val_sets[name] = (X_train_rs, y_train_rs, X_val, y_val)


class ResamplerV6(BaseSampler):
    def sample(self, train_val_sets: dict):
        for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):
            nm = NearMiss(n_jobs=-1)
            X_train_rs, y_train_rs = nm.fit_resample(X_train, y_train)

            self.train_val_sets[name] = (X_train_rs, y_train_rs, X_val, y_val)


# Oversampling the minority class to half of the majority class and undersampling the
# majority class to the minority class.
class ResamplerV7(BaseSampler):
    def sample(self, train_val_sets: dict):
        for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):
            over = RandomOverSampler(sampling_strategy=0.5, random_state=12345)
            under = RandomUnderSampler(sampling_strategy=1, random_state=12345)

            pipeline = Pipeline((("oversampling", over), ("undersampling", under)))
            X_train_rs, y_train_rs = pipeline.fit_resample(X_train, y_train)

            self.train_val_sets[name] = (X_train_rs, y_train_rs, X_val, y_val)


class ResamplerV8(BaseSampler):
    def sample(self, train_val_sets: dict):
        for name, (X_train, y_train, X_val, y_val) in tqdm(train_val_sets.items()):
            nn = NearestNeighbors(n_neighbors=5, n_jobs=-1)
            over = SMOTE(sampling_strategy=0.5, random_state=12345, k_neighbors=nn)
            under = RandomUnderSampler(sampling_strategy=1, random_state=12345)

            pipeline = Pipeline((("oversampling", over), ("undersampling", under)))
            X_train_rs, y_train_rs = pipeline.fit_resample(X_train, y_train)

            self.train_val_sets[name] = (X_train_rs, y_train_rs, X_val, y_val)
